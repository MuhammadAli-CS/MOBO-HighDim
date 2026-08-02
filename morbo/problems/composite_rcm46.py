#!/usr/bin/env python3
r"""A composite-structure, 34-dimensional real-world benchmark: RCM46
Optimal Power Flow (fuel cost, voltage deviation, active and reactive
power loss), from the same CEC2021 Real-World Constrained
Multi-Objective Optimization suite \citep{kumar2021rwcmop} as RCM40
(`morbo/problems/composite_rcm40.py`) -- the *same* real IEEE 14-bus
test system \citep{biswas2020opf}, same 34 design variables, same
admittance matrices $G$, $B$ and load vectors $P$, $Q$ (all reused
verbatim from `composite_rcm40.py`, re-verified against `CEC2021_func.m`
`case 46` and `Cal_par.m`'s `xmin46`/`xmax46`, which are bit-identical to
`xmin40`/`xmax40`), but with 4 objectives instead of 2 -- fuel cost and
voltage deviation added on top of RCM40's active/reactive power loss.

Why this benchmark, not a bigger IEEE bus system: the CEC2021 RWCMOP
suite's only power-flow instances are this 14-bus system (cases 40-46,
same network, different objective subsets) and a separate 15-bus
distribution feeder (cases 36-39, a different domain -- single-phase DG
sizing, not transmission OPF); there is no larger transmission-OPF
instance in this specific suite to scale RCM40 up to. RCM46 instead
tests a different axis: whether *more, more-decoupled* objectives drawn
from the same real system amplify composite modeling's benefit (RCM40
already showed a modest but genuine win from per-bus raw-response
richness; see methods.tex \S\ref{sec:rcm40-implemented}).

Objectives (`CEC2021_func.m` `case 46`, verified directly against the
source, not retyped from a paper):
    f_1 = sum_{k in {2,3,6,8}} (b_k P_{g,k} + c_k P_{g,k}^2)   (fuel cost)
    f_2 = sum_{k=1}^{14} P_{sp,k}                              (active power loss, = RCM40's f_1)
    f_3 = sum_{k=1}^{14} Q_{sp,k}                              (reactive power loss)
    f_4 = sum_{k=2}^{14} (1 - |V_k|)^2                         (voltage deviation)
All four minimized in the source; this module negates them to match
this project's maximize-everything convention.

A genuine discrepancy with RCM40, confirmed by inspection: RCM40's
second objective (`composite_rcm40.py`'s $f_2$) uses
`imag(V(1)*conj(I(2)))`, an apparent index slip (bus 2's current against
bus 1's voltage) documented there as "most likely unintentional."
RCM46's independently-written $f_3$ here uses the dimensionally-
consistent `imag(V(1)*conj(I(1)))` for the analogous term -- the same
formula shape, written correctly. This is suggestive (not proof) that
RCM40's version really is a typo in the original suite rather than a
deliberate choice: two versions of essentially the same quantity, one
consistent, one not.

Two of the four objectives ($f_1$, fuel cost, and $f_4$, voltage
deviation) are pure closed-form functions of the design variables
directly ($P_g$ = `x[26:30]`, $|V|$ = a direct function of
`x[0:13]`/`x[13:26]`) with no dependency on the network solve at all --
unlike $f_2$/$f_3$, which need $I = YV$ (the genuine "black box" here,
identical to RCM40's). Since `composite_*_reduction` functions in this
package operate on the raw response alone (not on `x` directly, matching
every other composite evalfn's calling convention in
`run_one_replication.py`), $P_g$ and $|V|$ are carried through as
pass-through raw-response components (not derived from any simulation,
just relabeled) so the reduction can still be a pure function of
`Y_raw`. This mirrors RCM40's own raw response, which likewise mixes
directly-known linear quantities with the genuinely bus-coupled
$P_{sp}$/$Q_{sp}$ terms -- not a new untested composite mechanism.
"""
from typing import Tuple

import torch
from torch import Tensor

from morbo.problems.composite_rcm40 import _G, _B, _P, _Q

RCM46_DIM = 34
RCM46_NUM_BUSES = 14
RCM46_GEN_BUSES = [1, 2, 5, 7]  # 0-indexed bus numbers {2,3,6,8} (1-indexed)
# Fuel-cost coefficients (`CEC2021_func.m` case 46: b1, c1 for ng=[1,2,3,6,8],
# a1 is all-zero so omitted; bus 1's own coefficients are never used since
# Pg(1) is fixed at 0, not a design variable -- only the last 4 entries,
# matching RCM46_GEN_BUSES order, are used here).
_FUEL_B = [1.75, 1.0, 3.25, 3.0]
_FUEL_C = [0.0175, 0.0625, 0.00834, 0.025]

RCM46_RAW_DIM = RCM46_NUM_BUSES + RCM46_NUM_BUSES + len(RCM46_GEN_BUSES) + (RCM46_NUM_BUSES - 1)  # 45


def get_composite_rcm46_fn(dtype=torch.double, device=None) -> Tuple[callable, Tensor]:
    r"""Construct the raw-response function and bounds for composite
    RCM46 Optimal Power Flow.

    Returns:
        A tuple `(raw_response, bounds)`:
            raw_response: callable mapping an `n x 34`-dim tensor (raw,
                *unnormalized* inputs -- identical layout to RCM40:
                `x[0:13]`=V_r, `x[13:26]`=V_m, `x[26:30]`=P_g,
                `x[30:34]`=Q_g, buses 2-14 unless noted) to an
                `n x 45`-dim tensor
                `[Psp_1..Psp_14, Qsp_1..Qsp_14, Pg_1..Pg_4, |V|_2..|V|_14]`.
            bounds: `2 x 34`-dim tensor (`[-1,1]` except generation
                `[0,1]`, identical to RCM40's, verified against
                `Cal_par.m`'s `xmin46`/`xmax46`).
    """
    Y = torch.tensor(_G, dtype=dtype, device=device) + 1j * torch.tensor(_B, dtype=dtype, device=device)

    def raw_response(X_input: Tensor) -> Tensor:
        X_flat = X_input.reshape(-1, RCM46_DIM).to(dtype)
        n = X_flat.shape[0]
        V_r = X_flat[:, 0:13]
        V_m = X_flat[:, 13:26]
        V = torch.zeros(n, RCM46_NUM_BUSES, dtype=torch.complex128 if dtype == torch.double else torch.complex64, device=device)
        V[:, 0] = 1.0
        V[:, 1:14] = torch.complex(V_r, V_m)

        I = V @ Y.T
        S = V * I.conj()
        Psp = S.real
        Qsp = S.imag

        Pg = X_flat[:, 26:30]
        V_mag = torch.sqrt(V_r ** 2 + V_m ** 2)  # buses 2-14

        return torch.cat([Psp, Qsp, Pg, V_mag], dim=-1).to(dtype)

    lb = torch.cat([
        torch.full((26,), -1.0, dtype=dtype, device=device),
        torch.full((8,), 0.0, dtype=dtype, device=device),
    ])
    ub = torch.full((RCM46_DIM,), 1.0, dtype=dtype, device=device)
    bounds = torch.stack([lb, ub], dim=0)
    return raw_response, bounds


def composite_rcm46_reduction(Y_raw: Tensor) -> Tensor:
    r"""Known reduction: fuel cost + voltage deviation (pure functions of
    the pass-through `Pg`/`|V|` raw components) and active/reactive power
    loss (literal sums over the per-bus `Psp`/`Qsp` raw response) --
    `CEC2021_func.m` case 46's own formula, verified directly.

    Args:
        Y_raw: `... x 45`-dim tensor
            `[Psp_1..Psp_14, Qsp_1..Qsp_14, Pg_1..Pg_4, |V|_2..|V|_14]`,
            as produced by `raw_response` above.

    Returns:
        An `... x 4`-dim tensor `[-f_1, -f_2, -f_3, -f_4]` (maximize
        convention -- all four minimized in the source, negated here to
        match this project's maximize-everything convention).
    """
    Psp = Y_raw[..., 0:14]
    Qsp = Y_raw[..., 14:28]
    Pg = Y_raw[..., 28:32]
    V_mag = Y_raw[..., 32:45]

    fuel_b = torch.tensor(_FUEL_B, dtype=Y_raw.dtype, device=Y_raw.device)
    fuel_c = torch.tensor(_FUEL_C, dtype=Y_raw.dtype, device=Y_raw.device)
    f1 = (fuel_b * Pg + fuel_c * Pg ** 2).sum(dim=-1)
    f2 = Psp.sum(dim=-1)
    f3 = Qsp.sum(dim=-1)
    f4 = ((1.0 - V_mag) ** 2).sum(dim=-1)

    return torch.stack([-f1, -f2, -f3, -f4], dim=-1)
