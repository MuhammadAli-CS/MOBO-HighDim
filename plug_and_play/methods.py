r"""Trust-region SHAPE-ADAPTATION methods, as a standalone, plug-and-play
module.

This file is the "methods" half of a minimal two-file interface (see
``benchmarks.py`` for the other half): everything here is a pure function
(or, for ``mab_shape``, a small stateful class) that computes a trust
region's *shape* -- a rotation ``R`` (``d x d`` orthonormal matrix) and a
per-axis width vector ``axis_lengths`` (``d``-dim) -- from local
optimization data. None of these functions know anything about BoTorch
acquisition functions, GP fitting, or the rest of a BO loop; they are
swappable, testable in isolation, and safe to import into any other
project's trust-region-style optimizer.

**Shared representation.** Every method returns the SAME two objects:
``R`` (rotation) and ``axis_lengths`` (full edge length along each
rotated axis, so ``axis_lengths.prod() ** (1/d) == length`` for the
current isotropic edge length ``length`` -- shape methods only change
*where* the trust-region volume goes, not its total size, which stays
governed by whatever success/failure-streak logic the outer BO loop
already uses). Consuming code turns this into an actual sampling region
via:

    w = (x - x_center) @ R          # rotate into the trust region's own frame
    inside = (w.abs() <= axis_lengths / 2).all(dim=-1)

This is an :math:`L_\infty` rotated box, not a true ellipsoid -- uniform
rejection sampling from a true :math:`d`-dimensional ellipsoid has
acceptance probability :math:`\pi^{d/2}/(\Gamma(d/2+1) \cdot 2^d)`, which
underflows well before :math:`d=100`. ``R = I`` recovers an axis-aligned
box (this is exactly ``ard_box``, and the fallback every method uses when
it doesn't have enough local data yet).

**Method summary** (full mechanism in each function's own docstring):

- ``isotropic``: no adaptation. ``R = I``, uniform widths. The baseline
  every other method is compared against.
- ``ard_box``: axis-aligned, but each axis rescaled by the fitted GP's
  per-dimension ARD lengthscale (the original TuRBO paper's own
  technique). No rotation.
- ``pca_ellipsoid``: rotates to align with the principal components of
  the trust region's own local data -- lengthscale-blind, purely
  data-driven.
- ``ard_pca_ellipsoid``: ``pca_ellipsoid``'s rotation, with axis widths
  *additionally* reweighted by ARD lengthscale projected onto each
  (already-fixed) principal axis.
- ``cma_ellipsoid``: CMA-ES-style persistent covariance, updated from
  elite (Pareto-improving) points plus an evolution-path term -- unlike
  the one-shot methods above, this has state that must be carried across
  iterations by the caller (see ``CMAState``).
- ``labcat_style``: replicates LABCAT (Visser et al. 2023)'s own
  construction -- fitness-weighted PCA computed *inside* a
  lengthscale-whitened coordinate frame, the opposite order from
  ``ard_pca_ellipsoid``.
- ``mab_shape``: not a geometry itself, but a per-trust-region bandit
  (``MABShapeBandit``) that picks among the methods above using each
  region's own reward history, so no single geometry needs to be fixed
  in advance.

Usage:

    from methods import SHAPE_METHODS
    R, axis_lengths = SHAPE_METHODS["pca_ellipsoid"](
        X=local_X, X_center=center, length=length, dim=100,
    )
"""
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "isotropic_shape",
    "extract_ard_lengthscale",
    "compute_ard_box_shape",
    "compute_pca_ellipsoid_shape",
    "compute_ard_pca_ellipsoid_shape",
    "compute_cma_ellipsoid_shape",
    "compute_labcat_style_shape",
    "CMAState",
    "MABShapeBandit",
    "SHAPE_METHODS",
]


# ---------------------------------------------------------------------------
# Shared helper: extracting a usable per-dimension lengthscale from a GP.
# ---------------------------------------------------------------------------
def extract_ard_lengthscale(model, dim: int) -> Optional[Tensor]:
    r"""Pull a ``dim``-dim per-input ARD lengthscale vector out of a fitted
    GP, geometric-mean-averaging across output dimensions if ``model``
    bundles more than one (e.g. a ``ModelListGP`` over several objectives).

    Returns ``None`` if the model has no per-dimension lengthscale to give
    (a single shared lengthscale, a kernel with no ``lengthscale``
    attribute, or a jointly-modeled multi-task GP) -- callers should fall
    back to ``isotropic_shape`` in that case, which is exactly what
    ``ard_box``/``ard_pca_ellipsoid``/``labcat_style`` do below.

    Args:
        model: a fitted GP (single-output ``Model`` or ``ModelListGP``)
            exposing ``.covar_module.base_kernel.lengthscale`` per output,
            e.g. any BoTorch ``SingleTaskGP`` with a Matern/RBF kernel.
        dim: expected input dimension; a mismatch also returns ``None``.
    """
    from botorch.models.multitask import KroneckerMultiTaskGP

    if isinstance(model, KroneckerMultiTaskGP):
        return None  # jointly-modeled multi-task GP has no per-dim ARD lengthscale to read
    models = [model] if not hasattr(model, "models") else model.models
    log_ls_per_output = []
    for m in models:
        try:
            ls = m.covar_module.base_kernel.lengthscale
        except AttributeError:
            return None
        if ls is None:
            return None
        ls = ls.reshape(-1)
        if ls.numel() != dim:
            return None
        log_ls_per_output.append(ls.log())
    return torch.stack(log_ls_per_output, dim=0).mean(dim=0).exp()


# ---------------------------------------------------------------------------
# isotropic: the trivial baseline every other method is measured against.
# ---------------------------------------------------------------------------
def isotropic_shape(length: Tensor, dim: int) -> Tuple[Tensor, Tensor]:
    r"""No shape adaptation: identity rotation, uniform axis widths.

    Exactly reproduces an unrotated, axis-aligned trust-region cube of
    edge length ``length`` -- the default MORBO/TuRBO behavior. Every
    other method in this file degrades to this when it doesn't have
    enough local data to fit anything more informative yet.
    """
    return (
        torch.eye(dim, device=length.device, dtype=length.dtype),
        length.expand(dim).clone(),
    )


# ---------------------------------------------------------------------------
# ard_box: TuRBO's own per-dimension rescaling (Eriksson et al. 2019).
# ---------------------------------------------------------------------------
def compute_ard_box_shape(
    lengthscale: Tensor, length: Tensor, dim: int
) -> Tuple[Tensor, Tensor]:
    r"""Axis-aligned box, rescaled per-dimension by GP ARD lengthscales
    (the original TuRBO paper's technique). No rotation: this is a
    lengthscale-weighted special case of every other method here, using
    ``R = I``.

    Directions the GP thinks are smooth (large lengthscale -> safe to
    explore widely, uninformative) get wider axes; directions it thinks
    are sharp (small lengthscale -> risky to stray far, informative) get
    narrower ones. At high dimension with many noisy per-dimension
    lengthscale estimates, this can pathologically collapse the region's
    volume -- see ``ard_pca_ellipsoid``'s docstring for why rotating
    first, rather than reweighting axis-aligned dimensions directly, is
    often more robust.

    Args:
        lengthscale: ``dim``-dim ARD lengthscale vector, e.g. from
            ``extract_ard_lengthscale``.
        length: 0-dim tensor, the trust region's current isotropic edge
            length.
        dim: input dimension.

    Returns:
        ``(R, axis_lengths)`` with ``R = I`` and
        ``axis_lengths.prod() ** (1/d) == length``.
    """
    R = torch.eye(dim, device=length.device, dtype=length.dtype)
    log_ls = lengthscale.log()
    weights = (log_ls - log_ls.mean()).exp()  # geometric-mean-normalized, prod == 1
    axis_lengths = length * weights
    return R, axis_lengths


# ---------------------------------------------------------------------------
# pca_ellipsoid: rotate to the local data's own principal axes.
# ---------------------------------------------------------------------------
def compute_pca_ellipsoid_shape(
    X: Tensor, X_center: Tensor, length: Tensor, dim: int, eig_floor: float = 1e-8
) -> Tuple[Tensor, Tensor]:
    r"""Rotate the trust region to the principal axes of its own local
    accumulated data, lengthscale-blind. Falls back to ``isotropic_shape``
    if fewer than ``dim + 1`` local points are available (PCA is
    underdetermined otherwise).

    Args:
        X: ``n x d`` local data (normalized to ``[0, 1]^d``).
        X_center: ``1 x d`` trust-region center -- used as the fixed
            center for the second-moment matrix (not ``X``'s own mean),
            so the ellipsoid stays anchored at the same point every other
            trust-region operation uses.
        length: 0-dim tensor, current isotropic edge length.
        dim: input dimension.
        eig_floor: minimum eigenvalue before taking a square root, guards
            near-zero axis widths from a near-degenerate point cloud.

    Returns:
        ``R``: eigenvectors of the local second-moment matrix about
            ``X_center``. ``axis_lengths``: geometric-mean-normalized so
            ``axis_lengths.prod() ** (1/d) == length``.
    """
    if X.shape[0] < dim + 1:
        return isotropic_shape(length, dim)
    delta = X - X_center
    cov = (delta.t() @ delta) / X.shape[0]
    eigvals, eigvecs = torch.linalg.eigh(cov)
    eigvals = eigvals.clamp_min(eig_floor)
    scale = eigvals.sqrt()
    log_scale = scale.log()
    weights = (log_scale - log_scale.mean()).exp()
    axis_lengths = length * weights
    return eigvecs, axis_lengths


# ---------------------------------------------------------------------------
# ard_pca_ellipsoid: PCA rotation + lengthscale-reweighted widths.
# ---------------------------------------------------------------------------
def compute_ard_pca_ellipsoid_shape(
    X: Tensor,
    X_center: Tensor,
    lengthscale: Tensor,
    length: Tensor,
    dim: int,
    eig_floor: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    r"""``pca_ellipsoid``'s rotation, with axis widths additionally
    reweighted by the ARD lengthscale projected onto each already-fixed
    principal axis.

    IMPORTANT design note: this deliberately does NOT compute PCA on
    lengthscale-normalized coordinates (``X / lengthscale``) and map the
    result back -- that construction is a mathematical no-op, since
    undoing a diagonal whitening exactly cancels it:
    ``D @ (D^-1 @ Sigma @ D^-1) @ D == Sigma`` for any diagonal ``D``. So
    instead, the rotation ``R`` is computed exactly as in
    ``pca_ellipsoid`` (lengthscale never touches the rotation), and
    lengthscale only enters afterward as a per-axis reweighting of that
    already-fixed rotation's widths:
    ``lambda_eff_k = || diag(lengthscale) @ R[:, k] ||_2``.

    (Contrast with ``labcat_style`` below, which whitens by lengthscale
    *before* computing a genuinely different PCA -- the opposite,
    non-degenerate ordering.)

    Args:
        X, X_center, length, dim, eig_floor: as in ``compute_pca_ellipsoid_shape``.
        lengthscale: ``dim``-dim ARD lengthscale vector.

    Returns:
        ``R``: identical to ``compute_pca_ellipsoid_shape``'s.
        ``axis_lengths``: geometric-mean-normalized so
            ``axis_lengths.prod() ** (1/d) == length``.
    """
    R, axis_lengths_pca = compute_pca_ellipsoid_shape(
        X=X, X_center=X_center, length=length, dim=dim, eig_floor=eig_floor
    )
    ell_eff = (lengthscale.unsqueeze(-1) * R).norm(dim=0)
    log_eff = ell_eff.log()
    eff_weights = (log_eff - log_eff.mean()).exp()
    axis_lengths = axis_lengths_pca * eff_weights
    log_axis = axis_lengths.log()
    axis_lengths = axis_lengths * (length / log_axis.mean().exp())
    return R, axis_lengths


# ---------------------------------------------------------------------------
# cma_ellipsoid: persistent CMA-ES-style covariance adaptation.
# ---------------------------------------------------------------------------
@dataclass
class CMAState:
    r"""Persistent per-trust-region state ``compute_cma_ellipsoid_shape``
    needs across calls -- unlike every other method in this file, CMA's
    covariance is *not* recomputed from scratch each time; it's
    exponentially smoothed, so the caller must keep one ``CMAState`` per
    trust region and feed it back in on every call.

    Attributes:
        C: ``d x d`` covariance matrix, initialized to identity.
        path: ``d``-dim evolution path (tracks sustained center movement
            direction), initialized to zero.
        prev_center: last call's trust-region center, or ``None`` on the
            first call (the evolution-path term is skipped until there
            are two centers to compare).
    """
    C: Optional[Tensor] = None
    path: Optional[Tensor] = None
    prev_center: Optional[Tensor] = None

    def reset(self, dim: int, device, dtype) -> None:
        self.C = torch.eye(dim, device=device, dtype=dtype)
        self.path = torch.zeros(dim, device=device, dtype=dtype)
        self.prev_center = None


def compute_cma_ellipsoid_shape(
    elites: Tensor,
    X_center: Tensor,
    state: CMAState,
    length: Tensor,
    dim: int,
    c_mu: float = 0.3,
    c1: float = 0.1,
    c_p: float = 0.3,
    eig_floor: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    r"""CMA-ES-style covariance adaptation (cf. Wang et al. 2026's
    AS-SMEA), mutating ``state`` in place.

    Two differences from the one-shot ``pca_ellipsoid``-family methods:
    (1) the covariance is *persistent*, exponentially smoothed across
    iterations (``C_new = (1 - c_mu - c1) * C_old + c_mu * C_elites +
    c1 * (path outer path)``), so one noisy batch can't whipsaw the
    region's orientation; (2) the update is weighted toward *success*
    (elite/Pareto-improving points) plus a rank-one evolution-path term
    tracking sustained center movement, rather than toward wherever
    candidates happened to be sampled (which for one-shot PCA partly
    reflects the sampler's own previous shape -- a feedback loop this
    construction avoids).

    Multi-objective note: classical CMA weights elites by fitness rank;
    a Pareto set has no total order, so elites (already selected by
    non-dominated sorting upstream) are weighted equally.

    Args:
        elites: ``n_elite x d`` current Pareto-elite points (normalized
            ``[0,1]^d``). Falls back to identity-covariance update if empty.
        X_center: ``1 x d`` trust-region center.
        state: mutated in place; call ``state.reset(dim, ...)`` before the
            first use for a new trust region.
        length: 0-dim tensor, current isotropic edge length.
        dim: input dimension.
        c_mu: learning rate for the rank-mu covariance update from elites.
        c1: learning rate for the rank-one evolution-path term.
        c_p: decay rate of the evolution path itself.
        eig_floor: minimum eigenvalue before taking a square root.

    Returns:
        ``(R, axis_lengths)``, geometric-mean-normalized as in the other
        methods.
    """
    if state.C is None:
        state.reset(dim, X_center.device, X_center.dtype)

    if elites.numel() > 0:
        delta = elites - X_center
        C_elites = (delta.t() @ delta) / elites.shape[0]
    else:
        C_elites = torch.eye(dim, device=X_center.device, dtype=X_center.dtype)

    if state.prev_center is not None:
        step = (X_center - state.prev_center).squeeze(0)
        state.path = (1 - c_p) * state.path + math.sqrt(
            c_p * (2 - c_p)
        ) * step / step.norm().clamp_min(1e-12)
        rank_one = torch.outer(state.path, state.path)
    else:
        rank_one = torch.zeros(dim, dim, device=X_center.device, dtype=X_center.dtype)

    state.C = (1 - c_mu - c1) * state.C + c_mu * C_elites + c1 * rank_one
    state.C = 0.5 * (state.C + state.C.t())  # symmetrize against fp drift
    state.prev_center = X_center.detach().clone()

    eigvals, eigvecs = torch.linalg.eigh(state.C)
    eigvals = eigvals.clamp_min(eig_floor)
    log_scale = eigvals.sqrt().log()
    weights = (log_scale - log_scale.mean()).exp()
    axis_lengths = length * weights
    return eigvecs, axis_lengths


# ---------------------------------------------------------------------------
# labcat_style: LABCAT (Visser et al. 2023)'s own construction.
# ---------------------------------------------------------------------------
def compute_labcat_style_shape(
    X: Tensor,
    X_center: Tensor,
    Y_obj: Tensor,
    lengthscale: Tensor,
    length: Tensor,
    dim: int,
    eig_floor: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    r"""LABCAT-style shape (Visser et al. 2023): fitness-weighted PCA
    computed genuinely *inside* lengthscale-whitened coordinates, composed
    (not undone) with the whitening -- the opposite construction/order
    from ``ard_pca_ellipsoid`` above, and (per the no-op identity
    documented there) a non-degenerate one.

    Procedure: (1) whiten local data by lengthscale,
    ``X' = (X - X_center) / lengthscale``; (2) compute a fitness-weighted
    covariance of ``X'`` (better objective values get more weight); (3)
    eigendecompose *that* covariance directly -- lengthscale and fitness
    both shape the rotation itself, unlike ``ard_pca_ellipsoid`` where
    lengthscale only ever touches axis widths. Axis widths use the same
    lengthscale-proportional formula as ``ard_box``.

    Multi-objective note: LABCAT is single-objective and weights points
    by ``1 - y'`` (lower normalized loss = more weight). With no single
    scalar to rank by, this weights each point by the mean of its
    per-objective values, independently min-max normalized across the
    local data (higher = better) -- the natural multi-objective analogue,
    not a literal reimplementation of a mechanism that presupposes a
    total order.

    Falls back to ``isotropic_shape`` if fewer than ``dim + 1`` local
    points are available.

    Args:
        X, X_center, length, dim, eig_floor: as in ``compute_pca_ellipsoid_shape``.
        Y_obj: ``n x m`` already-selected objective values for the same
            points as ``X`` (row-aligned) -- i.e. objectives, not raw
            model outputs.
        lengthscale: ``dim``-dim ARD lengthscale vector.

    Returns:
        ``R``: eigenvectors of the fitness-weighted whitened covariance.
        ``axis_lengths``: geometric-mean-normalized so
            ``axis_lengths.prod() ** (1/d) == length``.
    """
    if X.shape[0] < dim + 1:
        return isotropic_shape(length, dim)
    Xw = (X - X_center) / lengthscale
    y_min = Y_obj.min(dim=0).values
    y_max = Y_obj.max(dim=0).values
    y_range = (y_max - y_min).clamp_min(1e-9)
    y_norm = (Y_obj - y_min) / y_range
    w = y_norm.mean(dim=-1).clamp_min(1e-6)
    w_sum = w.sum()
    cov = (Xw * w.unsqueeze(-1)).t() @ Xw / w_sum
    cov = 0.5 * (cov + cov.t())
    eigvals, eigvecs = torch.linalg.eigh(cov)
    R = eigvecs
    log_ls = lengthscale.log()
    weights = (log_ls - log_ls.mean()).exp()
    axis_lengths = length * weights
    return R, axis_lengths


# ---------------------------------------------------------------------------
# mab_shape: a per-trust-region bandit over all of the above.
# ---------------------------------------------------------------------------
class MABShapeBandit:
    r"""Per-trust-region multi-armed bandit over a set of shape methods
    (default: every method above except ``labcat_style``, which wasn't
    part of the original arm set this was designed around -- add it to
    ``arms`` freely if desired).

    Rather than fixing one geometry globally, each trust region learns
    from its own reward history which shape suits its own local
    landscape. Reward is binary: 1.0 if the region's success-streak
    counter was just incremented (i.e. the outer BO loop's own
    length-doubling logic already detected an improving step), else 0.0.

    Two selection policies:

    - ``"epsilon"``: epsilon-greedy over a per-arm exponential moving
      average of reward. Simple, but under non-stationary landscapes a
      stale arm's EMA is only corrected when that arm happens to be
      replayed, and a fixed ``epsilon`` taxes tight budgets forever.
    - ``"ducb"``: discounted UCB (Garivier & Moulines 2011). Keeps a
      discounted reward sum and pull count per arm, both decayed by
      ``gamma`` every decision, and selects
      ``argmax(S_a/N_a + c * sqrt(log(sum N) / N_a))``. A stale arm's
      count decays, so its exploration bonus *regrows* and it gets
      automatically replayed after the landscape shifts; bonuses anneal
      as counts grow, so there's no permanent exploration tax. This is
      the policy we found to be the more robust of the two in practice.

    Usage:

        bandit = MABShapeBandit(arms=["isotropic", "ard_box", "pca_ellipsoid"])
        arm = bandit.select()               # e.g. "pca_ellipsoid"
        # ... run one BO iteration using SHAPE_METHODS[arm] ...
        bandit.update(success=True)         # credit *last* iteration's arm
        arm = bandit.select()               # pick the next arm
    """

    def __init__(
        self,
        arms=("isotropic", "ard_box", "pca_ellipsoid", "ard_pca_ellipsoid", "cma_ellipsoid"),
        policy: str = "ducb",
        epsilon: float = 0.15,
        reward_ema_alpha: float = 0.3,
        ducb_gamma: float = 0.95,
        ducb_c: float = 1.0,
    ):
        assert policy in ("epsilon", "ducb")
        self.arms = list(arms)
        self.policy = policy
        self.epsilon = epsilon
        self.reward_ema_alpha = reward_ema_alpha
        self.ducb_gamma = ducb_gamma
        self.ducb_c = ducb_c

        n = len(self.arms)
        self.arm_values = torch.zeros(n)  # epsilon-greedy EMA
        self.arm_pulls = torch.zeros(n)
        self.ducb_counts = torch.zeros(n)
        self.ducb_rewards = torch.zeros(n)
        self.last_arm: Optional[int] = None

    def update(self, success: bool) -> None:
        r"""Credit the arm played on the *previous* call to ``select()``
        with a reward of 1.0 if ``success`` else 0.0. Call once per BO
        iteration, before ``select()``."""
        if self.last_arm is None:
            return
        reward = 1.0 if success else 0.0
        if self.policy == "ducb":
            self.ducb_counts.mul_(self.ducb_gamma)
            self.ducb_rewards.mul_(self.ducb_gamma)
            self.ducb_counts[self.last_arm] += 1.0
            self.ducb_rewards[self.last_arm] += reward
            self.arm_pulls[self.last_arm] += 1
        else:
            a = self.reward_ema_alpha
            self.arm_values[self.last_arm] = (
                1 - a
            ) * self.arm_values[self.last_arm] + a * reward
            self.arm_pulls[self.last_arm] += 1

    def select(self) -> str:
        r"""Pick the next arm (a key into ``SHAPE_METHODS``) and remember
        it so the next ``update()`` call knows who to credit."""
        n = len(self.arms)
        if self.policy == "ducb":
            never_pulled = (self.arm_pulls == 0).nonzero().view(-1)
            if never_pulled.numel() > 0:
                arm = int(never_pulled[0])  # round-robin init: play every arm once
            else:
                eps = 1e-9
                counts = self.ducb_counts.clamp_min(eps)
                mean = self.ducb_rewards / counts
                total = self.ducb_counts.sum().clamp_min(1.0)
                bonus = self.ducb_c * torch.sqrt(torch.log(total) / counts)
                arm = int((mean + bonus).argmax())
        else:
            if torch.rand(()).item() < self.epsilon:
                arm = torch.randint(0, n, ()).item()
            else:
                arm = int(self.arm_values.argmax())
        self.last_arm = arm
        return self.arms[arm]


# ---------------------------------------------------------------------------
# Registry: name -> shape function. See each function's docstring above for
# exactly which keyword arguments it needs (they aren't identical, since
# e.g. cma_ellipsoid needs persistent state and labcat_style needs
# objective values that the one-shot PCA methods don't).
# ---------------------------------------------------------------------------
SHAPE_METHODS: Dict[str, Callable] = {
    "isotropic": isotropic_shape,
    "ard_box": compute_ard_box_shape,
    "pca_ellipsoid": compute_pca_ellipsoid_shape,
    "ard_pca_ellipsoid": compute_ard_pca_ellipsoid_shape,
    "cma_ellipsoid": compute_cma_ellipsoid_shape,
    "labcat_style": compute_labcat_style_shape,
    # "mab_shape" is intentionally not a plain function here -- it's a
    # meta-strategy over the others, requiring per-trust-region state.
    # Use the MABShapeBandit class directly (see its docstring above).
}
