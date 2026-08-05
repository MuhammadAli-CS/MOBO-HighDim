#!/usr/bin/env python3
r"""A composite-structure, 34-dimensional, FIVE-objective real-world
benchmark: many-objective Optimal Power Flow on the same real IEEE 14-bus
test system as RCM40/RCM46 (`composite_rcm40.py`/`composite_rcm46.py`,
CEC2021 RWCMOP suite \citep{kumar2021rwcmop}, \citet{biswas2020opf}).

Motivation. Across this project's benchmarks, composite modeling's
advantage scales with the number of genuinely *decoupled* objectives the
known reduction exposes: on the identical 14-bus system, RCM40's 2
objectives gave a modest win (sample-efficiency only) and RCM46's 4
objectives gave the strongest win in the project (+4.5% final HV, 5/5,
p=0.013; +4.6% and 17/17 on an independent pipeline). RCM46 is the
maximum objective count in the entire 50-problem CEC2021 suite, so this
module goes one further -- five objectives -- by adding the standard
voltage-stability objective from the many-objective OPF (MOOPF)
literature (\citealp{abido2003moopf}; the canonical fuel/emission/loss/
voltage-deviation/stability objective family), while staying on the
verified 14-bus admittance data (no larger transmission instance exists
in this suite, and a bigger network's per-bus raw response would exceed
this project's ~50-dimensional GP-fitting ceiling; see composite_oc20.py).

Objectives (all minimized in the source convention; negated here to
match this project's maximize-everything convention). The first four are
RCM46's, byte-for-byte:
    f_1 = sum_{k in {2,3,6,8}} (b_k P_{g,k} + c_k P_{g,k}^2)   (fuel cost)
    f_2 = sum_{k=1}^{14} P_{sp,k}                              (active power loss)
    f_3 = sum_{k=1}^{14} Q_{sp,k}                              (reactive power loss)
    f_4 = sum_{k=2}^{14} (1 - |V_k|)^2                         (voltage deviation)
    f_5 = max_{j in load buses} L_j                            (voltage stability)
The fifth is the Kessel--Glavitsch L-index \citep{kessel1986lindex}, the
standard static voltage-stability margin: partition the bus admittance
matrix Y into generator (PV + slack) and load (PQ) blocks, form
F = -Y_LL^{-1} Y_LG, and for each load bus j,
    L_j = | 1 - sum_{i in gen} F_{ji} V_i / V_j |,
with the system index max_j L_j (< 1 means stable; smaller is better).
It needs only the admittance matrix (already verified for RCM40/46) and
the bus voltages (design variables) -- no fabricated or transplanted
coefficients, unlike the emission objective sometimes used in MOOPF,
whose generator coefficients are not canonical for this 14-bus system.

Raw composite response g(x) in R^54: RCM46's 45-dim response
`[Psp(14), Qsp(14), Pg(4), |V|(13)]` plus the 9 per-load-bus L_j values,
so f_5's max-reduction reads a genuinely decoupled per-bus quantity (the
same checkpointed-per-component pattern the whole package uses), not a
pre-summarized scalar. 54 is under the ~60-100 GP-fit crash regime
established at OC20's 100-dim response (RCM46's 45 and photonic/topopt's
48 all ran cleanly).
"""
from typing import Tuple

import torch
from torch import Tensor

from morbo.problems.composite_rcm40 import _G, _B
from morbo.problems.composite_rcm46 import _FUEL_B, _FUEL_C, RCM46_GEN_BUSES

MOOPF_DIM = 34
MOOPF_NUM_BUSES = 14
MOOPF_GEN_BUSES = [1, 2, 5, 7]  # 0-indexed {2,3,6,8}, same as RCM46
# Generator (PV + slack) buses for the L-index partition: the slack bus 0
# plus the four generator buses. Load (PQ) buses are the rest.
_GEN_SET = [0] + MOOPF_GEN_BUSES              # {0,1,2,5,7}
_LOAD_SET = [b for b in range(MOOPF_NUM_BUSES) if b not in _GEN_SET]  # 9 buses
MOOPF_RAW_DIM = 14 + 14 + 4 + 13 + len(_LOAD_SET)  # 45 + 9 = 54


def get_composite_moopf_fn(dtype=torch.double, device=None) -> Tuple[callable, Tensor]:
    r"""Construct the raw-response function and bounds for composite
    5-objective MOOPF.

    Returns:
        A tuple `(raw_response, bounds)`:
            raw_response: callable mapping an `n x 34`-dim tensor (raw,
                *unnormalized* inputs -- identical layout to RCM40/46:
                `x[0:13]`=V_r, `x[13:26]`=V_m, `x[26:30]`=P_g,
                `x[30:34]`=Q_g, buses 2-14 unless noted) to an
                `n x 54`-dim tensor
                `[Psp(14), Qsp(14), Pg(4), |V|_2..14 (13), L_j (9)]`.
            bounds: `2 x 34`-dim tensor (identical to RCM40/46).
    """
    cdtype = torch.complex128 if dtype == torch.double else torch.complex64
    Y = torch.tensor(_G, dtype=dtype, device=device) + 1j * torch.tensor(_B, dtype=dtype, device=device)
    gen_idx = torch.tensor(_GEN_SET, device=device)
    load_idx = torch.tensor(_LOAD_SET, device=device)
    # F = -Y_LL^{-1} Y_LG  (load x gen), fixed by the network alone.
    Y_LL = Y[load_idx][:, load_idx]
    Y_LG = Y[load_idx][:, gen_idx]
    F_LG = -torch.linalg.solve(Y_LL, Y_LG)  # (9 x 5), complex

    def raw_response(X_input: Tensor) -> Tensor:
        X_flat = X_input.reshape(-1, MOOPF_DIM).to(dtype)
        n = X_flat.shape[0]
        V_r = X_flat[:, 0:13]
        V_m = X_flat[:, 13:26]
        V = torch.zeros(n, MOOPF_NUM_BUSES, dtype=cdtype, device=device)
        V[:, 0] = 1.0
        V[:, 1:14] = torch.complex(V_r, V_m)

        I = V @ Y.T
        S = V * I.conj()
        Psp = S.real
        Qsp = S.imag
        Pg = X_flat[:, 26:30]
        V_mag = torch.sqrt(V_r ** 2 + V_m ** 2)  # buses 2-14

        # L-index per load bus: L_j = |1 - sum_i F_ji V_i / V_j|.
        # (V_gen @ F_LG.T)[:, j] = sum_i F_LG[j, i] V_i, the generator-
        # voltage contribution to load bus j.
        V_gen = V[:, gen_idx]      # (n, 5)
        V_load = V[:, load_idx]    # (n, 9)
        sum_term = V_gen @ F_LG.T  # (n, 9)
        L_j = torch.abs(1.0 - sum_term / V_load)   # (n, 9), real

        return torch.cat([Psp, Qsp, Pg, V_mag, L_j], dim=-1).to(dtype)

    lb = torch.cat([
        torch.full((26,), -1.0, dtype=dtype, device=device),
        torch.full((8,), 0.0, dtype=dtype, device=device),
    ])
    ub = torch.full((MOOPF_DIM,), 1.0, dtype=dtype, device=device)
    bounds = torch.stack([lb, ub], dim=0)
    return raw_response, bounds


def composite_moopf_reduction(Y_raw: Tensor) -> Tensor:
    r"""Known reduction: RCM46's four objectives (fuel cost, active loss,
    reactive loss, voltage deviation) plus the L-index voltage-stability
    objective (max over the per-load-bus L_j raw components).

    Args:
        Y_raw: `... x 54`-dim tensor
            `[Psp(14), Qsp(14), Pg(4), |V|_2..14 (13), L_j (9)]`.

    Returns:
        An `... x 5`-dim tensor `[-f_1, -f_2, -f_3, -f_4, -f_5]`
        (maximize convention -- all five minimized in the source, negated
        here).
    """
    Psp = Y_raw[..., 0:14]
    Qsp = Y_raw[..., 14:28]
    Pg = Y_raw[..., 28:32]
    V_mag = Y_raw[..., 32:45]
    L_j = Y_raw[..., 45:54]

    fuel_b = torch.tensor(_FUEL_B, dtype=Y_raw.dtype, device=Y_raw.device)
    fuel_c = torch.tensor(_FUEL_C, dtype=Y_raw.dtype, device=Y_raw.device)
    f1 = (fuel_b * Pg + fuel_c * Pg ** 2).sum(dim=-1)
    f2 = Psp.sum(dim=-1)
    f3 = Qsp.sum(dim=-1)
    f4 = ((1.0 - V_mag) ** 2).sum(dim=-1)
    f5 = L_j.amax(dim=-1)

    return torch.stack([-f1, -f2, -f3, -f4, -f5], dim=-1)
