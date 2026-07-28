#!/usr/bin/env python3
r"""A composite-structure, genuinely high-dimensional (64D) chemistry
benchmark: calibrating rate-constant multipliers in GRI-Mech 3.0
(Smith et al., `gri30.yaml`, shipped with Cantera -- \citet{goodwin2024cantera})
against synthetic ignition-delay-time targets, in the spirit of the
mechanism-optimization methodology that produced GRI-Mech itself
(\citet{frenklach1992solutionmapping}'s "solution mapping" method: rank
reactions by sensitivity, then tune a subset of rate parameters within
their physically plausible uncertainty range to match a suite of
experimental ignition-delay/flame-speed targets).

Why this benchmark, and why it is NOT literally a "mixture formulation"
problem: a survey of real chemistry composite-benchmark candidates (this
project's own Table~\ref{tab:all-candidates}, plus an external survey at
https://github.com/Sahas266/mixture-opt-bench/blob/codex/scientific-simulator-benchmarks/docs/research/scientific-simulator-benchmarks.md)
found no chemistry benchmark that is both genuinely composite AND exceeds
~15 real design dimensions -- that survey explicitly states "none of the
retained chemistry candidates spans low, medium, and high dimension
without... an artificial parameterization." Real published surrogate-fuel
*mixture* formulations (jet/diesel/gasoline surrogates) universally use
only 4--8 blend components (verified via literature search, e.g.
4-component jet-fuel and diesel surrogates), so padding a mixture-design
benchmark to 50+ dimensions would itself be the artificial parameterization
the survey warns against. Kinetic-mechanism rate-constant calibration, by
contrast, is a real, long-established, genuinely high-dimensional
combustion-chemistry research problem (GRI-Mech's own development tuned
of order 100 rate parameters against ~75 experimental targets), giving a
defensible >50D chemistry benchmark without stretching any domain's
natural scale.

Design variables (64 of them): log2-scale multipliers on the Arrhenius
pre-exponential factor of the 64 GRI-Mech reactions with the largest
temperature-sensitivity coefficient at a representative condition
(1200 K, 1 atm, phi=1, computed once via Cantera's built-in forward
sensitivity analysis on a 0D constant-pressure ignition simulation, via
`cantera.ReactorNet.sensitivities()` with all 325 reactions' rate
multipliers enabled as sensitivity parameters and temperature (state
index 1 in `IdealGasConstPressureReactor`) as the observable, ranked by
`|sensitivity|` at the simulation's peak-dT/dt (ignition) time step; this
one-time offline selection is not itself part of the benchmark's
per-evaluation code path, so it is not a function in this module --
`TOP_REACTION_INDICES` below is its result, hardcoded). Bounds are
[-1, 1] per dimension, i.e. each
selected reaction's rate constant is allowed to vary by a factor of 2 in
either direction -- a standard order-of-magnitude "uncertainty factor"
for combustion rate constants (e.g. the Baulch et al. evaluated-kinetics
convention of quoting a multiplicative uncertainty factor per reaction;
using a single common factor of 2 here rather than per-reaction values
from that database is a simplification, not a fabricated range).

Raw composite response (24-dim: 6 conditions x 4 checkpoints): for each of
6 fixed test conditions (3 "low-temperature" T0 in {1000, 1050, 1100} K,
3 "high-temperature" T0 in {1300, 1400, 1500} K; all at 1 atm, phi=1,
methane/air), a 0D constant-pressure adiabatic ignition simulation
(`cantera.IdealGasConstPressureReactor`) is integrated with the candidate
rate multipliers applied via `Solution.set_multiplier`, and the reactor
temperature is checkpointed at 4 fixed time fractions of that condition's
own max-time budget -- mirroring this package's `composite_penicillin.py`/
`composite_snar.py` pattern of exposing a checkpointed physical trajectory
as the GP's raw response, rather than only the final reduced objectives.

Known reduction: each condition's ignition-delay time is recovered from
its 4 checkpointed temperatures by linear interpolation to the time the
temperature first crosses `T0 + 400 K` (a fixed, deterministic
post-processing formula applied identically to every mechanism
configuration and to the synthetic targets below -- so any bias from the
checkpoint resolution being coarser than a full adaptive-step ODE trace
cancels between target and evaluation). The final two objectives are the
negative summed squared log-ratio ignition-delay error against synthetic
targets, aggregated separately over the low-T and high-T condition
groups -- a genuine two-way trade-off, since methane's dominant
chain-branching pathway at low temperature (CH3 + O2 -> CH3O + O, CH2O
oxidation) differs from its dominant pathway at high temperature
(H + O2 -> O + OH), so a rate-multiplier vector that best matches one
regime's targets need not best match the other's (confirmed empirically:
perturbing all 64 multipliers together shifts every condition's ignition
delay together, but per-reaction perturbations shift the two groups by
different amounts -- see this module's own verification script, not
committed, referenced in methods.tex).

Targets: rather than depend on historical experimental shock-tube data
(as GRI-Mech's own original calibration did), targets are the SAME
checkpoint-based ignition-delay reduction applied ONCE to the unmodified,
as-shipped GRI-Mech 3.0 mechanism (`TARGET_IGNITION_DELAYS` below) -- a
standard "synthetic ground truth recovery" benchmark-construction
convention (the real simulator, the real mechanism, and the real
sensitivity-based reaction selection are all genuine; only the specific
target values being matched are generated from the same real mechanism
rather than from historical experiments). This keeps the benchmark
self-contained and reproducible without redistributing third-party
experimental datasets.
"""
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

GRI_MECH_DIM = 64

# Top 64 GRI-Mech 3.0 reactions by |forward sensitivity coefficient| of
# temperature at the ignition point of a 1200 K / 1 atm / phi=1 methane-air
# 0D constant-pressure ignition (computed once via
# `cantera.ReactorNet.sensitivities()` with all 325 reactions enabled;
# indices are into `cantera.Solution("gri30.yaml").reactions()`, 0-indexed).
# All 64 have genuinely nonzero sensitivity (every reaction ranked below
# these 64 has sensitivity coefficient magnitude below 0.37, vs. these
# ranging 396.6 down to 0.59); the next ~200 reactions are irrelevant
# C3+/NOx chemistry that this pure-methane/air combustion never touches
# (their sensitivity is exactly 0.0), confirming the selection is a real
# subset of chemically active reactions, not an arbitrary cutoff.
TOP_REACTION_INDICES = [
    157, 155, 154, 31, 52, 37, 118, 160, 169, 56, 97, 100, 115, 117, 112,
    167, 158, 166, 120, 35, 164, 162, 10, 174, 14, 73, 45, 44, 57, 26, 83,
    51, 86, 77, 84, 32, 33, 111, 119, 96, 114, 283, 163, 144, 141, 286, 3,
    156, 9, 98, 311, 24, 161, 159, 74, 34, 67, 94, 293, 284, 165, 289, 2, 103,
]
assert len(TOP_REACTION_INDICES) == GRI_MECH_DIM == len(set(TOP_REACTION_INDICES))

# (T0 [K], group) -- 3 low-T + 3 high-T conditions, all at 1 atm, phi=1,
# methane/air (`CH4`, `O2:1.0, N2:3.76`).
CONDITIONS = [
    (1000.0, "low"), (1050.0, "low"), (1100.0, "low"),
    (1300.0, "high"), (1400.0, "high"), (1500.0, "high"),
]
# Per-condition integration horizon (s): generously above the unmodified
# mechanism's own ignition delay at that condition (see
# TARGET_IGNITION_DELAYS) so a factor-of-2 rate slowdown still ignites
# within budget; conditions that still don't ignite within their budget
# saturate to `max_time` in `_ignition_delay_from_checkpoints` below,
# a well-defined (not crashing) worst-case penalty.
MAX_TIME = {1000.0: 2.0, 1050.0: 1.0, 1100.0: 0.5, 1300.0: 0.02, 1400.0: 0.008, 1500.0: 0.004}
N_CHECKPOINTS = 4
IGNITION_THRESHOLD_RISE = 400.0  # K above T0
P0_PA = 101325.0  # 1 atm
PHI = 1.0
FUEL = "CH4"
OXIDIZER = "O2:1.0, N2:3.76"

# Ignition-delay targets (s), computed ONCE via this module's own
# checkpoint-based reduction applied to the unmodified `gri30.yaml`
# mechanism (multiplier = 1.0 for all 64 reactions) -- see module
# docstring's "Targets" paragraph for why these are the target values
# rather than historical experimental data.
TARGET_IGNITION_DELAYS = {
    1000.0: 1.125788,
    1050.0: 0.315814,
    1100.0: 0.158438,
    1300.0: 0.011382,
    1400.0: 0.002605,
    1500.0: 0.001297,
}


def _run_condition(T0: float, log2_mults: np.ndarray) -> np.ndarray:
    r"""Run one 0D constant-pressure ignition to `MAX_TIME[T0]`, applying
    `2**log2_mults` as the rate multiplier on each of `TOP_REACTION_INDICES`
    in turn. Returns the `N_CHECKPOINTS`-length temperature checkpoint
    vector (imported lazily so this module has no hard Cantera dependency
    at import time for code that never calls `get_composite_gri_mech_fn`,
    matching this package's other optional-simulator-dependency problems).
    """
    import cantera as ct

    gas = ct.Solution("gri30.yaml")
    for idx, log2_m in zip(TOP_REACTION_INDICES, log2_mults):
        gas.set_multiplier(float(2.0 ** log2_m), idx)
    gas.TP = T0, P0_PA
    gas.set_equivalence_ratio(PHI, FUEL, OXIDIZER)
    reactor = ct.IdealGasConstPressureReactor(gas)
    net = ct.ReactorNet([reactor])
    net.rtol, net.atol = 1e-10, 1e-20
    max_time = MAX_TIME[T0]
    checkpoint_times = np.linspace(max_time / N_CHECKPOINTS, max_time, N_CHECKPOINTS)
    temps = np.full(N_CHECKPOINTS, T0)
    for ci in range(N_CHECKPOINTS):
        net.advance(checkpoint_times[ci])
        temps[ci] = reactor.T
    return temps


def get_composite_gri_mech_fn(dtype=torch.double, device=None) -> Tuple[callable, Tensor]:
    r"""Construct the raw-response (checkpointed ignition trajectory)
    function and bounds for composite GRI-Mech calibration.

    Returns:
        A tuple `(raw_response, bounds)`:
            raw_response: callable mapping an `n x 64`-dim tensor (raw,
                *unnormalized* log2-multiplier inputs, one per
                `TOP_REACTION_INDICES` entry, each in `[-1, 1]`) to an
                `n x 24`-dim tensor: 6 conditions (in `CONDITIONS` order)
                x 4 temperature checkpoints each, flattened.
            bounds: `2 x 64`-dim tensor, `[-1, 1]` per dimension.
    """
    def raw_response(X_input: Tensor) -> Tensor:
        X_flat = X_input.reshape(-1, GRI_MECH_DIM)
        rows = []
        for row in X_flat.detach().cpu().numpy():
            per_condition = [_run_condition(T0, row) for T0, _ in CONDITIONS]
            rows.append(np.concatenate(per_condition))
        out = torch.tensor(np.stack(rows), dtype=dtype, device=device)
        return out.view(*X_input.shape[:-1], len(CONDITIONS) * N_CHECKPOINTS)

    bounds = torch.tensor([[-1.0] * GRI_MECH_DIM, [1.0] * GRI_MECH_DIM], dtype=dtype, device=device)
    return raw_response, bounds


def composite_gri_mech_reduction(Y_raw: Tensor) -> Tensor:
    r"""Known reduction: per-condition ignition delay via linear
    interpolation to the `T0 + 400 K` crossing, then negative summed
    squared log-ratio error against `TARGET_IGNITION_DELAYS`, aggregated
    separately over the low-T and high-T condition groups.

    Args:
        Y_raw: `... x 24`-dim tensor, 6 conditions x 4 checkpoints
            (flattened), as produced by `raw_response` above.

    Returns:
        An `... x 2`-dim tensor `[-low_T_error, -high_T_error]`
        (maximize convention -- both should be maximized, i.e. error
        minimized -- matching `composite_dtlz2_reduction`/
        `composite_snar_reduction`'s own sign convention, so this evalfn
        should use `negate=False` in `BenchmarkFunction`).
    """
    Y = Y_raw.view(*Y_raw.shape[:-1], len(CONDITIONS), N_CHECKPOINTS)
    group_errors = {"low": [], "high": []}
    for i, (T0, group) in enumerate(CONDITIONS):
        temps = Y[..., i, :]
        max_time = MAX_TIME[T0]
        checkpoint_times = torch.linspace(
            max_time / N_CHECKPOINTS, max_time, N_CHECKPOINTS,
            dtype=Y.dtype, device=Y.device,
        )
        threshold = T0 + IGNITION_THRESHOLD_RISE
        t_full = torch.cat([
            torch.zeros(*Y.shape[:-2], 1, dtype=Y.dtype, device=Y.device),
            checkpoint_times.expand(*Y.shape[:-2], N_CHECKPOINTS),
        ], dim=-1)
        T_full = torch.cat([
            torch.full((*Y.shape[:-2], 1), T0, dtype=Y.dtype, device=Y.device),
            temps,
        ], dim=-1)

        crossed = T_full >= threshold
        # first crossing index per row; if none crossed, argmax of an
        # all-False mask is 0, which `never_crossed` below overrides.
        idx = torch.argmax(crossed.to(Y.dtype), dim=-1)
        never_crossed = ~crossed.any(dim=-1)
        idx_prev = torch.clamp(idx - 1, min=0)

        t_prev = torch.gather(t_full, -1, idx_prev.unsqueeze(-1)).squeeze(-1)
        t_curr = torch.gather(t_full, -1, idx.unsqueeze(-1)).squeeze(-1)
        T_prev = torch.gather(T_full, -1, idx_prev.unsqueeze(-1)).squeeze(-1)
        T_curr = torch.gather(T_full, -1, idx.unsqueeze(-1)).squeeze(-1)

        denom = torch.where((T_curr - T_prev).abs() < 1e-8, torch.ones_like(T_curr), T_curr - T_prev)
        frac = (threshold - T_prev) / denom
        tau = t_prev + frac * (t_curr - t_prev)
        tau = torch.where(never_crossed | (idx == 0), torch.full_like(tau, max_time), tau)
        tau = torch.clamp(tau, min=1e-8)

        target = TARGET_IGNITION_DELAYS[T0]
        log_ratio_sq_error = torch.log(tau / target) ** 2
        group_errors[group].append(log_ratio_sq_error)

    low_error = torch.stack(group_errors["low"], dim=0).sum(dim=0)
    high_error = torch.stack(group_errors["high"], dim=0).sum(dim=0)
    return torch.stack([-low_error, -high_error], dim=-1)
