r"""Trust-region SHAPE-ADAPTATION methods, as a standalone, plug-and-play
module.

This file is the "methods" half of a minimal two-file interface (see
``benchmarks.py`` for the other half). It is a REGISTRY, not a
reimplementation: every ``compute_*_shape`` function below is imported
directly from this repo's own ``morbo/utils.py`` -- the exact code
``morbo/trust_region.py``'s ``TurboHParams(tr_shape=...)`` dispatches to
internally -- not a copy. (An earlier version of this file copied those
functions' bodies locally "for standalone-ness." One of those copies --
``compute_cma_ellipsoid_shape`` -- turned out to have a real,
non-cosmetic bug: it omitted CMA-ES's sigma/trust-region-length
normalization on both the evolution-path and elite-covariance updates,
and used the wrong formula in the no-elites branch. It was never invoked
by ``run.py``, which always called the real ``morbo`` engine, so it never
affected any recorded result -- but it did mean this file's own claim of
being "the same functions, not a copy" was false for that one function.
Importing instead of copying makes that class of bug structurally
impossible: there is nothing left here that could drift from the
original.)

Everything here is a pure function (or, for ``mab_shape``, a small
stateful class) that computes a trust region's *shape* -- a rotation
``R`` (``d x d`` orthonormal matrix) and a per-axis width vector
``axis_lengths`` (``d``-dim) -- from local optimization data. None of
these functions know anything about BoTorch acquisition functions, GP
fitting, or the rest of a BO loop; they are swappable, testable in
isolation, and safe to import into any other project's trust-region-style
optimizer (that part IS standalone -- ``morbo/utils.py`` itself has no
dependency on the rest of a running BO loop either, only ``torch``).

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

**Method summary** (full mechanism in each function's own docstring, in
``morbo/utils.py``):

- ``isotropic``: no adaptation. ``R = I``, uniform widths. The baseline
  every other method is compared against. (No dedicated ``morbo/utils.py``
  function -- inlined in ``TrustRegion._compute_shape_for_mode`` there;
  ``isotropic_shape`` below is new, trivial, 4-line code reproducing it.)
- ``ard_box``: axis-aligned, but each axis rescaled by the fitted GP's
  per-dimension ARD lengthscale (the original TuRBO paper's own
  technique). No rotation. -> ``morbo.utils.compute_ard_box_shape``.
- ``pca_ellipsoid``: rotates to align with the principal components of
  the trust region's own local data -- lengthscale-blind, purely
  data-driven. -> ``morbo.utils.compute_pca_ellipsoid_shape``.
- ``ard_pca_ellipsoid``: ``pca_ellipsoid``'s rotation, with axis widths
  *additionally* reweighted by ARD lengthscale projected onto each
  (already-fixed) principal axis. -> ``morbo.utils.compute_ard_pca_ellipsoid_shape``.
- ``cma_ellipsoid``: CMA-ES-style persistent covariance, updated from
  elite (Pareto-improving) points plus an evolution-path term -- unlike
  the one-shot methods above, this has state that must be carried across
  iterations by the caller (see ``CMAState``, which wraps
  ``morbo.utils.compute_cma_ellipsoid_shape``'s pure-functional
  state-in/state-out signature into a mutate-in-place object).
- ``labcat_style``: replicates LABCAT (Visser et al. 2023)'s own
  construction -- fitness-weighted PCA computed *inside* a
  lengthscale-whitened coordinate frame, the opposite order from
  ``ard_pca_ellipsoid``. -> ``morbo.utils.compute_labcat_style_shape``.
- ``mab_shape``: not a geometry itself, but a per-trust-region bandit
  (``MABShapeBandit``) that picks among the methods above using each
  region's own reward history, so no single geometry needs to be fixed
  in advance. (No standalone ``morbo/utils.py`` equivalent -- the real
  engine's version is inlined as ``TrustRegion._select_mab_arm``, coupled
  to that class's buffers; ``MABShapeBandit`` extracts the same
  epsilon-greedy/D-UCB selection logic into an object usable without a
  full ``TrustRegion``.)

Usage:

    from methods import SHAPE_METHODS
    R, axis_lengths = SHAPE_METHODS["pca_ellipsoid"](
        X=local_X, X_center=center, length=length, dim=100,
    )
"""
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

# The compute_*_shape functions below import directly from this repo's own
# morbo/utils.py -- make the repo root importable so that works even when
# this file is used standalone (i.e. without going through run.py, which
# does the same path fixup for the same reason).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from morbo.utils import (  # noqa: E402
    compute_ard_box_shape,
    compute_ard_pca_ellipsoid_shape,
    compute_cma_ellipsoid_shape as _morbo_compute_cma_ellipsoid_shape,
    compute_labcat_style_shape,
    compute_pca_ellipsoid_shape,
    extract_ard_lengthscale,
)

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
# isotropic: the trivial baseline every other method is measured against.
# No dedicated morbo/utils.py function exists for this (it's the
# do-nothing case, inlined directly in TrustRegion._compute_shape_for_mode
# rather than factored out) -- this is new, but trivial, code.
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
# cma_ellipsoid: persistent CMA-ES-style covariance adaptation.
#
# morbo.utils.compute_cma_ellipsoid_shape is pure-functional: it takes
# (elites, X_center, prev_center, C, path, ...) and RETURNS
# (R, axis_lengths, C_new, path_new), leaving the caller (normally
# TrustRegion, a stateful Module) responsible for storing C_new/path_new
# back into its own buffers for next time. CMAState below is a thin
# mutate-in-place wrapper around that exact function, for callers who'd
# rather keep one state object per trust region than thread C/path
# through their own loop by hand. The math is 100% morbo.utils's -- this
# adds no computation of its own.
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
    AS-SMEA) -- a thin mutate-``state``-in-place wrapper around
    ``morbo.utils.compute_cma_ellipsoid_shape``'s pure-functional
    state-in/state-out signature. See that function's docstring
    (``morbo/utils.py``) for the exact mechanism and math.

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

    R, axis_lengths, C_new, path_new = _morbo_compute_cma_ellipsoid_shape(
        elites=elites,
        X_center=X_center,
        prev_center=state.prev_center,
        C=state.C,
        path=state.path,
        length=length,
        dim=dim,
        c_mu=c_mu,
        c1=c1,
        c_p=c_p,
        eig_floor=eig_floor,
    )
    state.C = C_new
    state.path = path_new
    state.prev_center = X_center.detach().clone()
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

    Two selection policies (same logic as ``TrustRegion._select_mab_arm``
    in ``morbo/trust_region.py``, extracted here into a standalone object
    usable without a full ``TrustRegion``):

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
