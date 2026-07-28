#!/usr/bin/env python3
r"""A composite-structure version of the SnAr (nucleophilic aromatic
substitution) plug-flow-reactor benchmark from Summit
(`sustainable-processes/summit`, `summit/benchmarks/snar.py`, Felton et al.
2021), itself built on the kinetic model measured by Hone et al. (2017,
React. Chem. Eng., DOI: 10.1039/C6RE00109B).

The underlying simulator integrates a 5-species reaction-kinetics ODE
(concentrations of 2,4-dinitrofluorobenzene, pyrrolidine, the desired SnAr
product, an undesired regioisomer, and a bis-alkylation side product) over
the reactor's residence time, then reduces the final concentration vector to
two economic/environmental objectives via a known closed-form formula: space
time yield (STY, kg product per m^3 per h -- maximize) and E-factor (kg waste
per kg product -- minimize). Summit exposes only the two reduced objectives;
this module forks the ODE integration (ported from
`SnarBenchmark._integrate_equations`/`_integrand`) to additionally expose the
raw final-concentration vector (plus the flow rate needed to reduce it) as
the composite-GP raw response, and reimplements Summit's own STY/E-factor
formula as a `composite_snar_reduction` operating purely on that raw
response -- mirroring the `composite_penicillin.py` pattern already in this
package (fork the real simulator's *state*, expose it as `g(x)`, then apply
the *known* reduction `L(g)` separately) rather than depending on the
`summit` PyPI package itself, whose pinned `scikit-learn<0.25` dependency
does not build on modern Python/pip toolchains (verified 2026-07-28: `pip
install summit` fails at the scikit-learn 0.24.2 sdist build step). The
kinetic ODE and reduction formula below were extracted directly from the
`summit==0.8.4` wheel (`summit/benchmarks/snar.py`) and verified numerically
against that extracted source before porting (see methods.tex, Summit SnAr
subsection).

4 design variables (Summit's own bounds):
    tau: residence time, minutes, [0.5, 2]
    equiv_pldn: equivalents of pyrrolidine (nucleophile), [1.0, 5.0]
    conc_dfnb: inlet concentration of 2,4-dinitrofluorobenzene, M, [0.1, 0.5]
    temperature: reactor temperature, deg C, [30, 120]
"""
from typing import Tuple

import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch import Tensor

SNAR_BOUNDS = [(0.5, 2.0), (1.0, 5.0), (0.1, 0.5), (30.0, 120.0)]
SNAR_DIM = 4
_REACTOR_VOLUME_ML = 5.0
_MOLECULAR_WEIGHTS = [159.09, 71.12, 210.21, 210.21, 261.33]  # g/mol
_RHO_ETHANOL = 0.789  # g/mL, at 25C


def _snar_integrand(t: float, C: np.ndarray, T: float, C_i: np.ndarray) -> np.ndarray:
    r"""Fork of `SnarBenchmark._integrand`: reaction rates for the 5-species
    kinetic model at temperature `T` (deg C), given current concentrations
    `C` and initial concentrations `C_i` (used only to threshold near-zero
    reactant concentrations, exactly as upstream)."""
    R = 8.314 / 1000  # kJ/K/mol
    T_ref = 90 + 273.71
    Tk = T + 273.71

    def k(k_ref: float, E_a: float, temp: float) -> float:
        return 0.6 * k_ref * np.exp(-E_a / R * (1 / temp - 1 / T_ref))

    k_a = k(57.9, 33.3, Tk)
    k_b = k(2.70, 35.3, Tk)
    k_c = k(0.865, 38.9, Tk)
    k_d = k(1.63, 44.8, Tk)

    C = C.copy()
    for i in (0, 1):
        if C[i] < 1e-6 * C_i[i]:
            C[i] = 0.0

    r = np.zeros(5)
    r[0] = -(k_a + k_b) * C[0] * C[1]
    r[1] = -(k_a + k_b) * C[0] * C[1] - k_c * C[1] * C[2] - k_d * C[1] * C[3]
    r[2] = k_a * C[0] * C[1] - k_c * C[1] * C[2]
    r[3] = k_a * C[0] * C[1] - k_d * C[1] * C[3]
    r[4] = k_c * C[1] * C[2] + k_d * C[1] * C[3]
    return r


def _integrate_one(tau: float, equiv_pldn: float, conc_dfnb: float, temperature: float) -> Tuple[np.ndarray, float]:
    r"""Fork of `SnarBenchmark._integrate_equations` (noise-free, `V=5` mL
    reactor). Returns `(C_final, q_tot)`: the 5-species final-concentration
    vector and the total volumetric flow rate (mL/min) implied by `tau`."""
    C_i = np.zeros(5)
    C_i[0] = conc_dfnb
    C_i[1] = equiv_pldn * conc_dfnb
    q_tot = _REACTOR_VOLUME_ML / tau
    res = solve_ivp(_snar_integrand, [0, tau], C_i, args=(temperature, C_i))
    C_final = res.y[:, -1].copy()
    C_final[C_final < 0] = 0.0
    return C_final, q_tot


def get_composite_snar_fn(dtype=torch.double, device=None) -> Tuple[callable, Tensor]:
    r"""Construct the raw-response (final-concentration) function and bounds
    for composite SnAr.

    Returns:
        A tuple `(raw_response, bounds)`:
            raw_response: callable mapping an `n x 4`-dim tensor (raw,
                *unnormalized* problem-space SnAr inputs: tau, equiv_pldn,
                conc_dfnb, temperature) to an `n x 6`-dim tensor
                `[C_dfnb, C_pldn, C_product, C_regioisomer, C_bis, q_tot]`
                -- the 5 final concentrations plus the total flow rate
                `q_tot = 5 / tau` (mL/min), which `composite_snar_reduction`
                needs alongside the concentrations to reproduce Summit's own
                STY/E-factor formula (`tau` itself is a design input, not
                part of the reactor's chemical state, so it must be carried
                through explicitly rather than re-derived from `Y_raw`
                alone).
            bounds: `2 x 4`-dim tensor, Summit's own raw-space bounds.
    """
    def raw_response(X_input: Tensor) -> Tensor:
        X_flat = X_input.reshape(-1, SNAR_DIM)
        rows = []
        for row in X_flat.tolist():
            tau, equiv_pldn, conc_dfnb, temperature = row
            C_final, q_tot = _integrate_one(tau, equiv_pldn, conc_dfnb, temperature)
            rows.append(np.append(C_final, q_tot))
        out = torch.tensor(np.stack(rows), dtype=dtype, device=device)
        return out.view(*X_input.shape[:-1], 6)

    bounds = torch.tensor(SNAR_BOUNDS, dtype=dtype, device=device).t()
    return raw_response, bounds


def composite_snar_reduction(Y_raw: Tensor) -> Tensor:
    r"""Known reduction: Summit's own STY/E-factor formula, applied to the
    raw final-concentration vector plus flow rate.

    Args:
        Y_raw: `... x 6`-dim tensor `[C_dfnb, C_pldn, C_product,
            C_regioisomer, C_bis, q_tot]`, as produced by `raw_response`
            above (un-negated, true reactor-state concentrations and flow
            rate -- this evalfn should use `negate=False` in
            `BenchmarkFunction`, like `composite_dtlz2_reduction`, since this
            reduction performs its own maximize-convention sign flip below;
            there is no Penicillin-style pre-existing sign convention to
            preserve here, so folding the negation into the reduction is the
            simpler choice, matching `composite_dtlz2.py` rather than
            `composite_penicillin.py`).

    Returns:
        An `... x 2`-dim tensor `[sty, -e_factor]`: space-time yield
        (kg/m^3/h, maximized) and *negative* E-factor (maximized, i.e. true
        E-factor minimized) -- bit-identical to Summit's own
        `SnarBenchmark._integrate_equations` STY/E-factor computation
        (noise-free) on the same inputs.
    """
    C_product = Y_raw[..., 2]
    q_tot = Y_raw[..., 5]
    M = _MOLECULAR_WEIGHTS
    V = _REACTOR_VOLUME_ML

    sty = 6e4 / 1000 * M[2] * C_product * q_tot / V
    sty = torch.clamp(sty, min=1e-6)

    other_mass = sum(M[i] * Y_raw[..., i] for i in range(5) if i != 2) * q_tot * 1e-3
    product_mass = M[2] * C_product * q_tot * 1e-3
    e_factor = (q_tot * _RHO_ETHANOL + other_mass) / product_mass
    e_factor = torch.where(
        product_mass <= 1e-12, torch.full_like(e_factor, 1e3), e_factor
    )
    e_factor = torch.clamp(e_factor, max=1e3)

    return torch.stack([sty, -e_factor], dim=-1)
