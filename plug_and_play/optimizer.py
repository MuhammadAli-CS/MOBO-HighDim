"""A minimal, self-contained trust-region Bayesian-optimization loop.

Single trust region (not this project's full coordinated multi-region
MORBO), built directly on BoTorch primitives -- independent per-objective
``SingleTaskGP``s, ``qLogExpectedHypervolumeImprovement`` scored on a
shape-adapted discrete candidate pool. This is deliberately simpler than
the multi-region engine this study's results were produced with (that
tradeoff is the whole point of this folder: a small, readable reference
implementation anyone can read start to finish, not a second copy of the
production system). What it keeps faithfully is the one thing this
project is actually about: candidates are drawn from a trust region whose
*shape* -- rotation ``R`` and per-axis widths ``axis_lengths`` -- is
recomputed every iteration by one of ``methods.py``'s functions, exactly
the same functions and the same rotated-box representation the full
system uses.

Everything here assumes ``evaluate(X)`` returns objectives to MAXIMIZE
(``benchmarks.py``'s convention) on inputs in ``[0, 1]^dim``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    NondominatedPartitioning,
)
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated
from gpytorch.mlls import SumMarginalLogLikelihood
from torch import Tensor

from methods import CMAState, MABShapeBandit, SHAPE_METHODS, extract_ard_lengthscale

Evaluator = Callable[[Tensor], Tensor]


@dataclass
class OptimizeResult:
    """Evaluated design/objective history and the running hypervolume,
    one entry per evaluation (so ``hv_history[i]`` is the hypervolume
    using the first ``i + 1`` rows of ``X``/``Y``)."""

    X: Tensor
    Y: Tensor
    hv_history: Tensor


def _sobol(n: int, d: int, seed: int) -> Tensor:
    return torch.quasirandom.SobolEngine(d, scramble=True, seed=seed).draw(n).double()


def _independent_gp(X: Tensor, Y: Tensor) -> ModelListGP:
    """Fit one ``SingleTaskGP`` per objective (independent, ARD Matern
    kernel by default) -- the per-dimension ARD lengthscale this returns
    is exactly what ``ard_box``/``ard_pca_ellipsoid``/``labcat_style``
    need from ``methods.py``."""
    models = [
        SingleTaskGP(X, Y[:, i : i + 1], outcome_transform=Standardize(m=1))
        for i in range(Y.shape[-1])
    ]
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def _hypervolume(Y: Tensor, ref_point: Tensor) -> float:
    pareto_mask = is_non_dominated(Y)
    pareto_Y = Y[pareto_mask]
    feasible = (pareto_Y >= ref_point).all(dim=-1)
    if not feasible.any():
        return 0.0
    return Hypervolume(ref_point=ref_point).compute(pareto_Y[feasible])


def _local_mask(X: Tensor, center: Tensor, length: Tensor) -> Tensor:
    """Which rows of ``X`` fall in the CURRENT ISOTROPIC box around
    ``center`` -- used only to select which data a shape method fits on.
    Deliberately isotropic regardless of the active shape method
    (mirrors this project's own design: trust-region *data selection*
    always uses the isotropic box, so shape adaptation only ever changes
    candidate sampling, never what the GP or a shape method is fit on in
    a way that could feed it badly-scaled coordinates)."""
    return ((X - center).abs() <= length / 2).all(dim=-1)


def _sample_candidates(
    center: Tensor, R: Tensor, axis_lengths: Tensor, dim: int, n: int, seed: int
) -> Tensor:
    """Draw ``n`` candidates uniformly from the rotated box
    ``{x : |(x - center) @ R| <= axis_lengths / 2}``, perturbing only a
    bounded subset of the ``dim`` rotated coordinates per candidate (this
    project's own subset-perturbation scheme: full-dimensional random
    perturbation degrades badly at high ``dim``, since almost every
    candidate then differs from the center in every coordinate at once).
    Perturbation happens IN THE ROTATED FRAME, then is rotated back --
    perturbing raw coordinates and rotating afterward would smear one
    "masked-in" raw direction across every principal axis, defeating the
    point of a sparse perturbation.
    """
    generator = torch.Generator().manual_seed(seed)
    n_pert = max(1, min(dim, math.ceil(dim * min(20.0 / dim, 1.0))))
    u = torch.rand(n, dim, generator=generator, dtype=center.dtype) - 0.5  # in [-0.5, 0.5)
    w = u * axis_lengths  # scale by rotated-frame axis widths
    mask = torch.zeros(n, dim, dtype=torch.bool)
    for i in range(n):
        idx = torch.randperm(dim, generator=generator)[:n_pert]
        mask[i, idx] = True
    w = torch.where(mask, w, torch.zeros_like(w))
    candidates = center + w @ R.t()
    return candidates.clamp(0.0, 1.0)


@dataclass
class TrustRegionState:
    """Persistent state for the single trust region, mutated across
    iterations by :func:`optimize`."""

    dim: int
    center: Tensor
    length: Tensor
    length_min: float = 0.01
    length_max: float = 1.6
    success_streak: int = 0
    failure_streak: int = 0
    success_tol: int = 3
    failure_tol: Optional[int] = None
    cma_state: CMAState = field(default_factory=CMAState)
    bandit: Optional[MABShapeBandit] = None

    def __post_init__(self) -> None:
        if self.failure_tol is None:
            self.failure_tol = self.dim

    def update_length(self, improved: bool) -> None:
        """Classic TuRBO doubling/halving rule: extend a streak of
        successes/failures, double the length after ``success_tol``
        consecutive improvements, halve it after ``failure_tol``
        consecutive non-improvements."""
        if improved:
            self.success_streak += 1
            self.failure_streak = 0
        else:
            self.failure_streak += 1
            self.success_streak = 0
        if self.success_streak >= self.success_tol:
            self.length = torch.clamp(self.length * 2.0, max=self.length_max)
            self.success_streak = 0
        elif self.failure_streak >= self.failure_tol:
            self.length = torch.clamp(self.length / 2.0, min=self.length_min / 2)
            self.failure_streak = 0


def _compute_shape(
    method: str,
    X_local: Tensor,
    Y_local: Tensor,
    center: Tensor,
    length: Tensor,
    lengthscale: Optional[Tensor],
    dim: int,
    tr_state: TrustRegionState,
) -> tuple[Tensor, Tensor]:
    """Dispatch to ``methods.SHAPE_METHODS[method]`` (plus ``"mab_shape"``,
    a meta-strategy over the others -- see ``methods.py``).

    This is a GENERIC dispatch, not a hardcoded per-method branch: it
    builds one shared context of everything a shape function could need
    (local data, center, current length, ARD lengthscale, objective
    values, dimension), then calls ``SHAPE_METHODS[method]`` with
    whichever of those keys that specific function's signature actually
    asks for (via `inspect.signature`). A new pure function added to
    ``SHAPE_METHODS`` in ``methods.py`` is runnable here immediately, with
    zero changes to this file, as long as it only needs arguments already
    in that shared context. ``cma_ellipsoid`` is the one exception -- it
    additionally needs elite (Pareto-improving) points and this trust
    region's persistent covariance state, both computed here since they
    aren't part of the shared per-iteration context.

    Falls back to the isotropic shape whenever a method needs the ARD
    lengthscale and the fitted GP doesn't have one yet (e.g. very early
    on, before enough data is collected for a stable fit).
    """
    import inspect

    from methods import isotropic_shape

    if method == "isotropic":
        return isotropic_shape(length, dim)
    if method == "mab_shape":
        if tr_state.bandit is None:
            tr_state.bandit = MABShapeBandit()
        arm = tr_state.bandit.select()
        return _compute_shape(arm, X_local, Y_local, center, length, lengthscale, dim, tr_state)
    if method not in SHAPE_METHODS:
        raise ValueError(f"Unknown method {method!r}. See methods.SHAPE_METHODS.")

    context = dict(
        X=X_local, X_center=center, Y_obj=Y_local, lengthscale=lengthscale,
        length=length, dim=dim,
    )
    if method == "cma_ellipsoid":
        pareto_mask = is_non_dominated(Y_local) if Y_local.shape[0] > 0 else torch.zeros(0, dtype=torch.bool)
        context["elites"] = X_local[pareto_mask] if Y_local.shape[0] > 0 else X_local[:0]
        context["state"] = tr_state.cma_state

    fn = SHAPE_METHODS[method]
    needed = inspect.signature(fn).parameters
    if "lengthscale" in needed and lengthscale is None:
        return isotropic_shape(length, dim)
    kwargs = {k: v for k, v in context.items() if k in needed}
    return fn(**kwargs)


def optimize(
    evaluate: Evaluator,
    dim: int,
    ref_point: Tensor,
    method: str,
    *,
    n_init: int = 20,
    n_iter: int = 40,
    batch_size: int = 5,
    seed: int = 0,
    length_init: float = 0.8,
    length_min: float = 0.01,
    length_max: float = 1.6,
    n_candidates: int = 512,
) -> OptimizeResult:
    """Run one trust-region BO replication.

    Args:
        evaluate: ``benchmarks.Benchmark.eval_fn`` (or any callable with
            the same signature/convention).
        dim: input dimension.
        ref_point: hypervolume reference point (``benchmarks.Benchmark.ref_point``).
        method: a key into ``methods.SHAPE_METHODS`` plus ``"mab_shape"``
            (see ``methods.py``'s module docstring for what each does).
        n_init: number of Sobol-initialization evaluations.
        n_iter: number of BO iterations (each proposing ``batch_size`` points).
        batch_size: candidates evaluated per iteration.
        seed: random seed (Sobol init, candidate sampling, and any
            benchmark-side randomness go through this).
        length_init, length_min, length_max: trust-region edge-length
            bounds (in normalized ``[0, 1]^d`` units).
        n_candidates: size of the discrete candidate pool scored by the
            acquisition function each iteration.

    Returns:
        ``OptimizeResult`` with the full evaluation history and a
        per-evaluation hypervolume trace.
    """
    torch.manual_seed(seed)
    ref_point = torch.as_tensor(ref_point, dtype=torch.double)

    X = _sobol(n_init, dim, seed)
    Y = evaluate(X).double().detach()

    tr = TrustRegionState(
        dim=dim,
        center=X[Y.sum(dim=-1).argmax() : Y.sum(dim=-1).argmax() + 1].clone(),
        length=torch.tensor(length_init, dtype=torch.double),
        length_min=length_min,
        length_max=length_max,
    )

    hv_history = [_hypervolume(Y[: i + 1], ref_point) for i in range(Y.shape[0])]
    best_hv = hv_history[-1]

    for it in range(n_iter):
        model = _independent_gp(X, Y)
        lengthscale = extract_ard_lengthscale(model, dim)

        local_mask = _local_mask(X, tr.center, tr.length)
        X_local, Y_local = X[local_mask], Y[local_mask]
        if X_local.shape[0] < dim + 1:
            X_local, Y_local = X, Y  # not enough local data yet: fit on everything

        R, axis_lengths = _compute_shape(
            method, X_local, Y_local, tr.center, tr.length, lengthscale, dim, tr,
        )

        candidates = _sample_candidates(
            center=tr.center.squeeze(0), R=R, axis_lengths=axis_lengths,
            dim=dim, n=n_candidates, seed=seed * 100_003 + it,
        )

        partitioning = NondominatedPartitioning(ref_point=ref_point, Y=Y)
        acq = qLogExpectedHypervolumeImprovement(
            model=model, ref_point=ref_point.tolist(), partitioning=partitioning,
        )
        with torch.no_grad():
            acq_values = acq(candidates.unsqueeze(1))  # q=1 per candidate
        top = torch.topk(acq_values, k=min(batch_size, n_candidates)).indices
        X_new = candidates[top].detach()
        Y_new = evaluate(X_new).double().detach()

        X, Y = torch.cat([X, X_new]), torch.cat([Y, Y_new])
        for i in range(Y_new.shape[0]):
            hv = _hypervolume(Y[: X.shape[0] - Y_new.shape[0] + i + 1], ref_point)
            hv_history.append(hv)

        improved = hv_history[-1] > best_hv
        best_hv = max(best_hv, hv_history[-1])
        tr.update_length(improved)
        if tr.bandit is not None:
            tr.bandit.update(success=improved)
        if improved:
            best_idx = Y.sum(dim=-1).argmax()
            tr.center = X[best_idx : best_idx + 1].clone()
        if tr.length <= tr.length_min:
            # Restart: recenter at the current best point, reset length.
            best_idx = Y.sum(dim=-1).argmax()
            tr.center = X[best_idx : best_idx + 1].clone()
            tr.length = torch.tensor(length_init, dtype=torch.double)
            tr.success_streak = tr.failure_streak = 0

    return OptimizeResult(X=X, Y=Y, hv_history=torch.tensor(hv_history, dtype=torch.double))
