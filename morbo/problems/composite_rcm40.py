#!/usr/bin/env python3
r"""A composite-structure, 34-dimensional real-world benchmark: RCM40
Optimal Power Flow, from the CEC2021 Real-World Constrained
Multi-Objective Optimization suite \citep{kumar2021rwcmop}, itself a
real IEEE 14-bus test-system OPF instance \citep{biswas2020opf}.

Unlike SnAr/GRI-Mech (Sections 9.22.5/9.23 in methods.tex, whose "black
box" is a genuine ODE/reactor integration), this benchmark's raw
response is a closed-form linear-algebra computation (solve the grid's
admittance equations) rather than an iterative numerical simulation --
still the electrical-network analogue of the checkpointed-state
composite pattern used elsewhere in this package (each bus's power
injection is exposed, rather than only the two summed final
objectives), just cheaper to evaluate than a real ODE/PDE solve.

All numeric constants (the 14x14 admittance matrices $G$, $B$ and the
per-bus load vectors $P$, $Q$) were extracted directly from the
reference implementation \citep{kumar2021rwcmop}
(`CEC2021_func.m`, `case 40`, and `Cal_par.m`'s `xmin40`/`xmax40`;
both files are in
https://github.com/P-N-Suganthan/2021-RW-MOP/raw/main/CEC2021-RWCMOP.zip),
not retyped from a paper's typeset equations, to avoid the same class of
transcription error already caught once in this project's RCM17 survey
(see methods.tex \S\ref{sec:rcm17}).

Design variables (34, all bounds $\pm1$ except generation which is
$[0,1]$ -- verified against `Cal_par.m`'s `xmin40 = -1*ones(1,34);
xmin40(27:34)=0`, `xmax40 = +1*ones(1,34)`):
    x[0:13]   -- V_r, real part of complex bus voltage, buses 2-14
    x[13:26]  -- V_m, imaginary part of complex bus voltage, buses 2-14
    x[26:30]  -- P_g, active generation, buses {2,3,6,8}
    x[30:34]  -- Q_g, reactive generation, buses {2,3,6,8}
(bus 1 is the fixed slack bus, V_1 = 1 + 0j, not a design variable.)

Raw composite response $g(x)$: per-bus real/reactive power injection
$(P_{sp}, Q_{sp}) \in \mathbb{R}^{28}$ (computed by solving
$I = YV$, $S = V \odot \bar I$ -- one matrix-vector product and one
elementwise product, the entire "simulation" here), plus
$\operatorname{Im}(I_2) \in \mathbb{R}$ needed by the second objective's
slack-bus term (see "known discrepancy" below) -- 29 raw outputs total.

Known reduction $L(g)$: both final objectives are literal sums over the
per-bus raw response,
    f_1 = sum_{k=1}^{14} P_{sp,k}                       (total active power loss)
    f_2 = -Im(I_2) + sum_{k=2}^{14} Q_{sp,k}             (total reactive power loss)
reproducing `CEC2021_func.m`'s own
`f(i,1) = real(V(1)*conj(I(1))) + sum(Psp(2:14))` (algebraically
identical to `sum(Psp(1:14))` since $V_1=1$ makes
$\operatorname{Re}(V_1 \bar I_1) = \operatorname{Re}(\bar I_1) =
\operatorname{Re}(S_1) = P_{sp,1}$) and
`f(i,2) = imag(V(1)*conj(I(2))) + sum(Qsp(2:14))` exactly, INCLUDING the
reference code's own apparent index slip -- $f_2$ uses bus 2's current
$I_2$ multiplied against bus 1's voltage $V_1$, dimensionally
inconsistent with $f_1$'s matched-bus pattern ($V_1 \bar I_1$) and, on
inspection, most likely unintentional in the original paper's code
rather than a deliberate design choice. Reproduced here exactly as
published (not silently "corrected") since any comparison against the
suite's own reported baselines needs the same formula they actually
used -- see methods.tex \S\ref{sec:rcm40} for the full discussion.

Both objectives are minimized in the source; this module returns them
negated (`-f_1, -f_2`) to maximize, matching this project's own
maximize-everything convention elsewhere (`composite_snar_reduction`,
`composite_dtlz2_reduction`).
"""
from typing import Tuple

import torch
from torch import Tensor

RCM40_DIM = 34
RCM40_NUM_BUSES = 14
RCM40_GEN_BUSES = [1, 2, 5, 7]  # 0-indexed bus numbers {2,3,6,8} (1-indexed)

_G = [[6.025029055768224, -4.999131600798035, 0.0, 0.0, -1.025897454970189, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-4.999131600798035, 9.521323610814779, -1.1350191923073958, -1.686033150614943, -1.7011396670944048, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.1350191923073958, 3.1209949022329564, -1.9859757099255606, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -1.686033150614943, -1.9859757099255606, 10.512989522036175, -6.840980661495671, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.025897454970189, -1.7011396670944048, 0.0, -6.840980661495671, 9.568017783560265, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 6.5799234074662225, 0.0, 0.0, 0.0, 0.0, -1.9550285631772606, -1.525967440450974, -3.0989274038379877, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.3260550394673585, -3.9020495524474277, 0.0, 0.0, 0.0, -1.424005487019931], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.9020495524474277, 5.782934306147827, -1.8808847537003996, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -1.9550285631772606, 0.0, 0.0, 0.0, -1.8808847537003996, 3.8359133168776602, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -1.525967440450974, 0.0, 0.0, 0.0, 0.0, 0.0, 4.014992027272893, -2.4890245868219187, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, -3.0989274038379877, 0.0, 0.0, 0.0, 0.0, 0.0, -2.4890245868219187, 6.724946148466233, -1.1369941578063267], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.424005487019931, 0.0, 0.0, 0.0, -1.1369941578063267, 2.560999644826258]]
_B = [[-19.447070205514382, 15.263086523179553, 0.0, 0.0, 4.234983682334831, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [15.263086523179553, -30.272115398779064, 4.781863151757718, 5.115838325872083, 5.193927397969713, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 4.781863151757718, -9.82238012935164, 5.0688169775939205, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 5.115838325872083, 5.0688169775939205, -38.654171207607796, 21.578553981691588, 0.0, 4.889512660317341, 0.0, 1.8554995578159004, 0.0, 0.0, 0.0, 0.0, 0.0], [4.234983682334831, 5.193927397969713, 0.0, 21.578553981691588, -35.533639456044824, 4.257445335253384, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 4.257445335253384, -17.34073280991911, 0.0, 0.0, 0.0, 0.0, 4.0940743442404415, 3.1759639650294003, 6.102755448193116, 0.0], [0.0, 0.0, 0.0, 4.889512660317341, 0.0, 0.0, -19.549005948264654, 5.676979846721544, 9.09008271975275, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.676979846721544, -5.676979846721544, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.8554995578159004, 0.0, 0.0, 9.09008271975275, 0.0, -24.092506375267877, 10.365394127060915, 0.0, 0.0, 0.0, 3.0290504569306034], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.365394127060915, -14.768337876521436, 4.402943749460521, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 4.0940743442404415, 0.0, 0.0, 0.0, 4.402943749460521, -8.497018093700962, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 3.1759639650294003, 0.0, 0.0, 0.0, 0.0, 0.0, -5.427938591201612, 2.251974626172212, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 6.102755448193116, 0.0, 0.0, 0.0, 0.0, 0.0, 2.251974626172212, -10.66969354947068, 2.314963475105352], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0290504569306034, 0.0, 0.0, 0.0, 2.314963475105352, -5.344013932035955]]
_P = [0.0, 0.217, 0.9420000000000001, 0.478, 0.076, 0.11199999999999999, 0.0, 0.0, 0.295, 0.09, 0.035, 0.061, 0.135, 0.149]
_Q = [0.0, 0.127, 0.19, -0.039, 0.016, 0.075, 0.0, 0.0, 0.166, 0.057999999999999996, 0.018000000000000002, 0.016, 0.057999999999999996, 0.05]


def get_composite_rcm40_fn(dtype=torch.double, device=None) -> Tuple[callable, Tensor]:
    r"""Construct the raw-response (per-bus power injection) function
    and bounds for composite RCM40 Optimal Power Flow.

    Returns:
        A tuple `(raw_response, bounds)`:
            raw_response: callable mapping an `n x 34`-dim tensor (raw,
                *unnormalized* inputs -- `x[0:13]`=V_r, `x[13:26]`=V_m,
                `x[26:30]`=P_g, `x[30:34]`=Q_g, all buses 2-14 unless
                noted) to an `n x 29`-dim tensor
                `[Psp_1..Psp_14, Qsp_1..Qsp_14, Im(I_2)]`.
            bounds: `2 x 34`-dim tensor (`[-1,1]` except generation
                `[0,1]`, matching `Cal_par.m`'s `xmin40`/`xmax40`).
    """
    Y = torch.tensor(_G, dtype=dtype, device=device) + 1j * torch.tensor(_B, dtype=dtype, device=device)
    P = torch.tensor(_P, dtype=dtype, device=device)
    Q = torch.tensor(_Q, dtype=dtype, device=device)

    def raw_response(X_input: Tensor) -> Tensor:
        X_flat = X_input.reshape(-1, RCM40_DIM).to(dtype)
        n = X_flat.shape[0]
        V = torch.zeros(n, RCM40_NUM_BUSES, dtype=torch.complex128 if dtype == torch.double else torch.complex64, device=device)
        V[:, 0] = 1.0
        V[:, 1:14] = torch.complex(X_flat[:, 0:13], X_flat[:, 13:26])

        I = V @ Y.T  # (n, 14) complex, I_k = sum_j Y[k,j] V[j]
        S = V * I.conj()
        Psp = S.real  # (n, 14)
        Qsp = S.imag  # (n, 14)
        Im_I2 = I[:, 1].imag  # (n,)

        return torch.cat([Psp, Qsp, Im_I2.unsqueeze(-1)], dim=-1).to(dtype)

    lb = torch.cat([
        torch.full((26,), -1.0, dtype=dtype, device=device),
        torch.full((8,), 0.0, dtype=dtype, device=device),
    ])
    ub = torch.full((RCM40_DIM,), 1.0, dtype=dtype, device=device)
    bounds = torch.stack([lb, ub], dim=0)
    return raw_response, bounds


def composite_rcm40_reduction(Y_raw: Tensor) -> Tensor:
    r"""Known reduction: both objectives are literal sums over the
    per-bus raw response (`CEC2021_func.m`'s own formula, including its
    apparent I(2)/V(1) index slip in the second objective -- see module
    docstring).

    Args:
        Y_raw: `... x 29`-dim tensor
            `[Psp_1..Psp_14, Qsp_1..Qsp_14, Im(I_2)]`, as produced by
            `raw_response` above.

    Returns:
        An `... x 2`-dim tensor `[-f_1, -f_2]` (maximize convention --
        both minimized in the source, negated here to match this
        project's maximize-everything convention).
    """
    Psp = Y_raw[..., 0:14]
    Qsp = Y_raw[..., 14:28]
    Im_I2 = Y_raw[..., 28]

    f1 = Psp.sum(dim=-1)
    f2 = -Im_I2 + Qsp[..., 1:14].sum(dim=-1)
    return torch.stack([-f1, -f2], dim=-1)
