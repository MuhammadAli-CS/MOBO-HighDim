#!/usr/bin/env python3
r"""A standalone BoTier-style tiered composite-objective BO loop.

This is deliberately *not* built on MORBO's trust-region/HVI machinery --
BoTier (Haddadnia, Grashoff & Strieth-Kalthoff, arXiv:2501.15554) reduces M
objectives to a single scalar via a hierarchical, priority-ordered
scalarization and then runs ordinary single-objective BO on that scalar. No
Pareto front is mapped and no hypervolume is computed; the whole point is to
target one specific compromise point implied by a priority ordering and a
set of per-objective thresholds, rather than spend budget covering tradeoffs
nobody asked for. `morbo/utils.py::get_fitted_model` is reused as-is (one
independent GP per raw objective, exactly the composite-modeling primitive
Part 1 relies on) since it has no MORBO-specific assumptions.

All objective values are in the maximization convention (higher = better)
throughout, matching the rest of this codebase.
"""
from typing import Callable, Dict, List, Optional

import torch
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler
from botorch.utils.sampling import draw_sobol_samples
from torch import Tensor

from morbo.utils import get_fitted_model


def hierarchical_scalarization(
    Y: Tensor, thresholds: Tensor, order: Tensor, mu: float = 1e-2
) -> Tensor:
    r"""BoTier's tiered composite utility, with differentiable min/Heaviside.

    `Xi = sum_k [ min(Y_ordered[k], t_k) * prod_{j<k} H(Y_ordered[j] - t_j) ]`

    i.e. the k-th highest-priority objective only contributes once every
    higher-priority objective has cleared its own threshold. `min` is
    approximated via a log-sum-exp softmin and the Heaviside step `H` via a
    sigmoid, both controlled by `mu` (smaller `mu` = closer to the true,
    non-differentiable BoTier formula; this codebase's own convention for
    "smaller mu = sharper/more accurate" mirrors `approximate_hv_alpha`'s use
    elsewhere in `morbo/`).

    Args:
        Y: a `... x M`-dim tensor of maximization-convention objective
            values.
        thresholds: an `M`-dim tensor; `thresholds[k]` is the threshold for
            the k-th highest-priority objective (paired with `order[k]`).
        order: an `M`-dim `LongTensor`, a permutation of `range(M)` giving
            priority order (`order[0]` = highest priority).
        mu: softening parameter, > 0.

    Returns:
        A `...`-dim tensor of composite `Xi` values.
    """
    Y_ordered = Y.index_select(-1, order)
    thresholds_ordered = thresholds.expand_as(Y_ordered)
    soft_min = -mu * torch.logsumexp(
        torch.stack([-Y_ordered / mu, -thresholds_ordered / mu], dim=-1), dim=-1
    )
    margin = Y_ordered - thresholds_ordered
    soft_H = torch.sigmoid(margin / mu)
    cum_gate = torch.cumprod(soft_H, dim=-1)
    # gate for tier k excludes tier k's own Heaviside term -- only tiers
    # *before* k gate it.
    gate = torch.cat(
        [torch.ones_like(cum_gate[..., :1]), cum_gate[..., :-1]], dim=-1
    )
    return (soft_min * gate).sum(dim=-1)


def run_botier_bo(
    eval_fn: Callable[[Tensor], Tensor],
    bounds: Tensor,
    thresholds: Tensor,
    order: Tensor,
    n_initial_points: int,
    max_evals: int,
    batch_size: int = 1,
    mu: float = 1e-2,
    raw_samples: int = 512,
    num_restarts: int = 10,
    mc_samples: int = 128,
    seed: int = 0,
    verbose: bool = False,
) -> Dict[str, Tensor]:
    r"""Run a BoTier-style tiered single-objective BO loop.

    Args:
        eval_fn: callable mapping an `n x d`-dim tensor of designs to an
            `n x M`-dim tensor of maximization-convention objective values.
        bounds: `2 x d`-dim tensor of problem bounds.
        thresholds: `M`-dim tensor of per-tier thresholds (see
            `hierarchical_scalarization`).
        order: `M`-dim `LongTensor` priority ordering.
        n_initial_points: number of Sobol points to start with.
        max_evals: total evaluation budget (including initial points).
        batch_size: candidates per BO iteration.
        mu: softening parameter for `hierarchical_scalarization`.
        raw_samples, num_restarts, mc_samples: standard botorch acquisition
            optimization knobs.
        seed: random seed.
        verbose: print progress.

    Returns:
        A dict with `X`, `Y` (full history), `xi` (composite utility per
        observed point), `best_idx`, `best_X`, `best_Y` -- the single
        recovered compromise point implied by `thresholds`/`order`.
    """
    torch.manual_seed(seed)
    dtype, device = bounds.dtype, bounds.device
    dim = bounds.shape[-1]
    thresholds = thresholds.to(dtype=dtype, device=device)
    order = order.to(device=device, dtype=torch.long)

    X = draw_sobol_samples(bounds=bounds, n=n_initial_points, q=1).squeeze(1)
    Y = eval_fn(X)
    objective = GenericMCObjective(
        lambda samples, X=None: hierarchical_scalarization(samples, thresholds, order, mu)
    )

    n_evals = X.shape[0]
    while n_evals < max_evals:
        model = get_fitted_model(
            X=X,
            Y=Y,
            use_ard=True,
            max_cholesky_size=50_000,
            input_transform=Normalize(d=dim, bounds=bounds),
            outcome_transform=Standardize(m=Y.shape[-1]),
        )
        best_f = hierarchical_scalarization(Y, thresholds, order, mu).max()
        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
        acqf = qExpectedImprovement(
            model=model, best_f=best_f, sampler=sampler, objective=objective
        )
        q = min(batch_size, max_evals - n_evals)
        candidates, _ = optimize_acqf(
            acqf,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )
        Y_new = eval_fn(candidates)
        X = torch.cat([X, candidates], dim=0)
        Y = torch.cat([Y, Y_new], dim=0)
        n_evals = X.shape[0]
        if verbose:
            print(f"{n_evals}) best Xi so far: {best_f.item():.4f}")

    xi_all = hierarchical_scalarization(Y, thresholds, order, mu)
    best_idx = int(xi_all.argmax().item())
    return {
        "X": X,
        "Y": Y,
        "xi": xi_all,
        "best_idx": best_idx,
        "best_X": X[best_idx],
        "best_Y": Y[best_idx],
    }


def percentile_thresholds(
    Y_warm_start: Tensor, order: Tensor, percentile: float = 0.5
) -> Tensor:
    r"""Hand-specified-tier baseline: threshold each objective at a plain
    percentile of its warm-start values, in the given priority order.

    Args:
        Y_warm_start: `n x M`-dim tensor of warm-start (e.g. Sobol)
            objective observations, maximization convention.
        order: `M`-dim priority ordering (unused for the threshold values
            themselves, kept for signature symmetry with the LLM-proposed
            path -- thresholds are indexed the same way either way).
        percentile: quantile of each objective's warm-start distribution to
            use as its threshold (0.5 = median).

    Returns:
        An `M`-dim tensor of thresholds, ordered to match `order` (i.e.
        `thresholds[k]` is the threshold for objective `order[k]`).
    """
    q = torch.quantile(Y_warm_start, percentile, dim=0)
    return q.index_select(0, order.to(device=q.device))
