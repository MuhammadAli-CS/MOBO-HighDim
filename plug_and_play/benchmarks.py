r"""Benchmark problems, as a standalone, plug-and-play module.

This file is the "benchmarks" half of a minimal two-file interface (see
``methods.py`` for the other half: the trust-region shape-adaptation
methods that get compared *on* these benchmarks). Every benchmark is
exposed through one function, ``get_benchmark``, returning a single
``Benchmark`` object with a uniform interface:

    bench = get_benchmark("dtlz2", dim=100, num_objectives=2)
    X = torch.rand(5, bench.dim, dtype=torch.double)   # bounds are always [0,1]^d
    Y = bench.eval_fn(X)                                # n x num_objectives, MAXIMIZE
    hv_ref_point = bench.ref_point                       # for hypervolume computation

**Convention**: every ``eval_fn`` here follows the MAXIMIZATION
convention (higher is always better, on every objective) with inputs
pre-normalized to the unit cube ``[0, 1]^d`` -- the actual problem's
native bounds are folded into ``eval_fn`` itself, so calling code never
needs to think about per-problem bound ranges. This matches what most BO
libraries (BoTorch included) expect a black-box multi-objective test
function to look like.

Where possible, the actual objective math is NOT hand-retyped here -- the
project-specific benchmarks (``sparse_dtlz2``, ``rotated_sparse_dtlz2``,
``time_varying_sparse_dtlz2``, ``rover``, ``sparse_rover``, ``bbob_biobj``,
``lasso_bench_mo``) live in ``problems/`` *inside this folder* (copied
from this project's own tested implementations, not imported across the
repo boundary), and the standard-benchmark ones (``dtlz1``...``dtlz7``,
``penicillin``, ``vehicle_safety``, ``welded_beam``) use BoTorch's
built-in synthetic problems directly. This whole folder -- ``methods.py``,
``benchmarks.py``, ``problems/``, ``optimizer.py``, ``run.py`` -- is
self-contained: nothing here imports from this repo's top-level ``morbo``
package. Only ``torch``, ``botorch``, and ``gpytorch`` are required
(plus, only if you use it, the third-party ``LassoBench`` package for
``lasso_bench_mo``).

**Benchmark families available** (pass the corresponding key as ``name``
to ``get_benchmark``; see each factory function's docstring for details
and extra kwargs):

- ``dtlz1`` / ``dtlz2`` / ``dtlz3`` / ``dtlz5`` / ``dtlz7``: the standard
  DTLZ multi-objective test family (Deb et al.), varying in landscape
  character (smooth vs. rugged, degenerate/disconnected Pareto fronts).
- ``composite_dtlz2``: DTLZ2 exposed as a COMPOSITE benchmark -- the raw
  intermediate quantities its formula is built from, plus the known
  reduction to final objectives, generalized (unlike this project's own
  ``composite_dtlz2.py``) to any number of objectives ``M``, not just
  ``M=2``. Defaults to ``dim=6, num_objectives=5``. See ``Benchmark``'s
  ``raw_eval_fn``/``composite_reduction`` fields.
- ``sparse_dtlz2``: DTLZ2 with a controllable, literal gap between
  nominal and *effective* input dimension (``k_eff`` of the informative
  dims are kept, the rest are pinned no-ops).
- ``rotated_sparse_dtlz2``: same, with the informative subspace rotated
  off the coordinate axes.
- ``time_varying_sparse_dtlz2``: same, with which dims are informative
  switching mid-optimization.
- ``rover`` / ``sparse_rover``: a real (non-algebraic) trajectory-planning
  cost simulator, optionally padded with literal no-op input dimensions.
- ``bbob_biobj``: pairs of BBOB-style landscape representatives (sphere,
  rosenbrock, ellipsoidal, rastrigin, a custom multi-peak function) --
  faithful-in-spirit reimplementations of representative BBOB functions,
  not the official ``cocoex`` package (see ``problems/bbob_style.py``
  for the exact honesty caveat).
- ``lasso_bench_mo``: a real bi-objective feature-selection benchmark
  built on the LassoBench package (requires ``pip install LassoBench``).
- ``penicillin``: a real bioprocess-simulation benchmark (BoTorch's
  ``Penicillin``).
- ``vehicle_safety`` / ``welded_beam``: real low-dimensional engineering
  design problems (unconstrained variant -- objectives only, no explicit
  outcome constraints, for simplicity).

Adding a new benchmark: write a ``_make_<name>(dim, **kwargs) ->
Benchmark`` function following the pattern below, then add it to the
``BENCHMARKS`` registry at the bottom of this file.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch
from torch import Tensor

__all__ = ["Benchmark", "get_benchmark", "BENCHMARKS"]


@dataclass
class Benchmark:
    r"""Uniform return type for every benchmark factory in this file.

    Attributes:
        eval_fn: callable, ``X`` (``n x dim``, values in ``[0, 1]^dim``)
            -> ``Y`` (``n x num_objectives``), MAXIMIZATION convention.
            For composite benchmarks (``raw_eval_fn`` is not ``None``),
            this is ``composite_reduction`` already composed with
            ``raw_eval_fn`` -- so ``eval_fn`` always gives you final,
            directly-hypervolume-able objectives regardless of whether a
            benchmark is composite, and every benchmark stays usable the
            same way.
        dim: input dimension.
        num_objectives: number of objectives.
        ref_point: length-``num_objectives`` list, a hypervolume
            reference point known to be dominated by the Pareto front
            (i.e. worse than the worst value achievable on every
            objective) -- required for computing hypervolume-based
            metrics, not used by ``eval_fn`` itself.
        raw_eval_fn: only set for COMPOSITE benchmarks (currently just
            ``composite_dtlz2``). ``X -> Y_raw`` (``n x num_raw``,
            un-negated/minimization convention), the raw intermediate
            response a composite GP would model directly -- richer than
            the final objectives, with a KNOWN deterministic reduction to
            them (see ``composite_reduction``). ``None`` for every other
            benchmark.
        composite_reduction: only set alongside ``raw_eval_fn``.
            ``Y_raw -> Y`` (``n x num_objectives``, MAXIMIZATION
            convention) -- the known reduction. ``eval_fn`` is exactly
            ``lambda X: composite_reduction(raw_eval_fn(X))``.
    """
    eval_fn: Callable[[Tensor], Tensor]
    dim: int
    num_objectives: int
    ref_point: List[float]
    raw_eval_fn: Optional[Callable[[Tensor], Tensor]] = None
    composite_reduction: Optional[Callable[[Tensor], Tensor]] = None


def _unit_cube_wrap(raw_f: Callable, bounds: Tensor) -> Callable[[Tensor], Tensor]:
    r"""Wrap a raw MINIMIZATION-convention function (defined on its own
    native ``bounds``) into a MAXIMIZATION-convention function defined on
    ``[0, 1]^dim`` -- the two conventions every ``Benchmark.eval_fn``
    shares, regardless of what convention the underlying problem
    implementation natively uses."""
    lb, ub = bounds[0], bounds[1]

    def eval_fn(X_unit: Tensor) -> Tensor:
        X_native = lb + X_unit * (ub - lb)
        return -raw_f(X_native)

    return eval_fn


def _unit_cube_wrap_raw(raw_f: Callable, bounds: Tensor) -> Callable[[Tensor], Tensor]:
    r"""Like ``_unit_cube_wrap``, but WITHOUT negating the output -- for
    composite benchmarks' raw response, which stays in its natural
    (un-negated) convention; only the final ``composite_reduction``
    negates, matching how this project's own composite problems work."""
    lb, ub = bounds[0], bounds[1]

    def raw_eval_fn(X_unit: Tensor) -> Tensor:
        X_native = lb + X_unit * (ub - lb)
        return raw_f(X_native)

    return raw_eval_fn


# ---------------------------------------------------------------------------
# Standard DTLZ family (BoTorch built-ins).
# ---------------------------------------------------------------------------
def _make_dtlz(name: str, dim: int, num_objectives: int = 2, **_) -> Benchmark:
    r"""Deb et al.'s DTLZ multi-objective test family. All share a
    position/distance-variable decomposition; they differ in the
    "distance" function ``g`` and how the Pareto front is shaped:

    - ``dtlz1``: linear Pareto front, ``g`` is highly multimodal
      (many local Pareto-optimal fronts) -- a genuinely hard search evem
      at moderate ``dim``.
    - ``dtlz2``: smooth, unimodal ``g`` (concave spherical front) -- the
      "easy" member of the family, most commonly used as a sanity check.
    - ``dtlz3``: like DTLZ2's front, but with DTLZ1's highly multimodal
      ``g`` grafted on -- tests robustness to a rugged local landscape.
    - ``dtlz5``: front degenerates to a curve (only 2 of the position
      variables actually vary the front's shape, regardless of
      ``num_objectives``).
    - ``dtlz7``: disconnected Pareto front (``2^(num_objectives-1)``
      separate regions).

    Args:
        dim: nominal input dimension.
        num_objectives: number of objectives ``M``. The number of
            "distance" variables is ``dim - M + 1``.
    """
    from botorch.test_functions.multi_objective import DTLZ1, DTLZ2, DTLZ3, DTLZ5, DTLZ7

    ctor = {"dtlz1": DTLZ1, "dtlz2": DTLZ2, "dtlz3": DTLZ3, "dtlz5": DTLZ5, "dtlz7": DTLZ7}[name]
    problem = ctor(dim=dim, num_objectives=num_objectives, negate=False)
    ref_points = {
        "dtlz1": -400.0, "dtlz2": -6.0, "dtlz3": -10000.0, "dtlz5": -10.0, "dtlz7": -15.0,
    }
    return Benchmark(
        eval_fn=_unit_cube_wrap(problem, problem.bounds.to(torch.double)),
        dim=dim,
        num_objectives=num_objectives,
        ref_point=[ref_points[name]] * num_objectives,
    )


# ---------------------------------------------------------------------------
# Composite-modeling DTLZ2 (project-specific; generalizes this project's
# own composite_dtlz2.py, which hardcodes num_objectives=2, to any M).
# ---------------------------------------------------------------------------
def _make_composite_dtlz2(dim: Optional[int] = None, num_objectives: int = 5, **_) -> Benchmark:
    r"""Composite-structure DTLZ2: the RAW intermediate quantities DTLZ2's
    own formula is built from --
    ``[g, cos(p_0), sin(p_0), ..., cos(p_{M-2}), sin(p_{M-2})]`` (``1 +
    2*(M-1)`` raw components) -- exposed as what a composite GP would
    model directly, together with the KNOWN deterministic reduction back
    to the ``M`` final objectives. Mathematically identical to direct
    DTLZ2 (verified numerically against BoTorch's own ``DTLZ2`` in
    ``problems/composite_dtlz2_general.py``'s tests) -- same Pareto front,
    same optimum -- so this is an apples-to-apples A/B against modeling
    the objectives directly (this project's own composite-modeling
    extension, generalized here from its original ``num_objectives=2``
    special case to arbitrary ``M``). Defaults to ``dim=6, M=5``: with
    ``k = dim - M + 1 = 2`` distance variables feeding ``g`` and ``M-1=4``
    position variables, the raw response is ``1 + 2*(M-1) = 9``-dim --
    richer than the 6-dim input, the interesting composite-modeling
    regime (contrast with high-``dim``/low-``M`` DTLZ2, where the raw
    response is much lower-dimensional than the input).

    Args:
        dim: input dimension. Must satisfy ``dim >= num_objectives``.
        num_objectives: ``M >= 2`` (unlike ``composite_dtlz2.py``, not
            restricted to 2).

    Note on running this at the default ``M=5`` through
    ``run.py``/``optimizer.py``: ``optimizer.py``'s acquisition function
    uses BoTorch's EXACT hypervolume box decomposition
    (``NondominatedPartitioning``). Confirmed by direct testing:
    ``M=2``/``3``/``4`` all run fine at this file's default settings, but
    ``M=5`` hits an out-of-memory error -- not specific to this benchmark
    (plain ``dtlz2`` at the same ``dim``/``M=5`` hits the identical
    blowup), so it's ``optimizer.py``'s exact-partitioning acquisition
    hitting its scaling wall around the Pareto-front size ``M=5``
    produces here, not a bug in this benchmark's construction (verified
    numerically identical to BoTorch's ``DTLZ2`` above). Workarounds: a
    smaller ``n_init``/evaluated-point count, or ``num_objectives=4`` or
    fewer.
    """
    from problems.composite_dtlz2_general import (
        composite_dtlz2_general_reduction,
        get_composite_dtlz2_general_fn,
    )

    dim = dim if dim is not None else 6
    raw_f, bounds = get_composite_dtlz2_general_fn(dim=dim, num_objectives=num_objectives)
    bounds = bounds.to(torch.double)
    raw_eval_fn = _unit_cube_wrap_raw(raw_f, bounds)

    def reduction(Y_raw: Tensor) -> Tensor:
        return composite_dtlz2_general_reduction(Y_raw, num_objectives=num_objectives)

    return Benchmark(
        eval_fn=lambda X: reduction(raw_eval_fn(X)),
        dim=dim,
        num_objectives=num_objectives,
        ref_point=[-6.0] * num_objectives,
        raw_eval_fn=raw_eval_fn,
        composite_reduction=reduction,
    )


# ---------------------------------------------------------------------------
# Effective-vs-nominal-dimension DTLZ2 variants (project-specific).
# ---------------------------------------------------------------------------
def _make_sparse_dtlz2(dim: int, num_objectives: int = 2, k_eff: int = 5, **_) -> Benchmark:
    r"""DTLZ2 with only ``k_eff`` of its "distance" input dimensions
    actually informative -- the rest are literal no-ops (pinned to their
    optimal value regardless of what the optimizer sets them to), so
    nominal dimension and effective dimension can be varied independently.
    See ``problems/sparse_dtlz2.py`` for the exact construction.
    """
    from problems.sparse_dtlz2 import get_sparse_dtlz2_fn

    f, bounds = get_sparse_dtlz2_fn(dim=dim, num_objectives=num_objectives, k_eff=k_eff)
    return Benchmark(
        eval_fn=_unit_cube_wrap(f, bounds.to(torch.double)),
        dim=dim, num_objectives=num_objectives, ref_point=[-6.0] * num_objectives,
    )


def _make_rotated_sparse_dtlz2(dim: int, num_objectives: int = 2, k_eff: int = 5, **_) -> Benchmark:
    r"""``sparse_dtlz2``, with the ``k_eff``-dim informative subspace
    rotated off the coordinate axes -- tests whether shape-adaptation
    methods that assume axis-alignment (e.g. ``ard_box``) lose their edge
    once the informative subspace isn't axis-aligned."""
    from problems.rotated_sparse_dtlz2 import get_rotated_sparse_dtlz2_fn

    f, bounds = get_rotated_sparse_dtlz2_fn(dim=dim, num_objectives=num_objectives, k_eff=k_eff)
    return Benchmark(
        eval_fn=_unit_cube_wrap(f, bounds.to(torch.double)),
        dim=dim, num_objectives=num_objectives, ref_point=[-6.0] * num_objectives,
    )


def _make_time_varying_sparse_dtlz2(
    dim: int, num_objectives: int = 2, k_eff: int = 49, switch_at_eval: int = 300, **_
) -> Benchmark:
    r"""``sparse_dtlz2``, where WHICH ``k_eff`` dims are informative
    switches to a different random subset partway through optimization
    (after ``switch_at_eval`` evaluations) -- tests robustness to a
    non-stationary effective subspace."""
    from problems.time_varying_sparse_dtlz2 import get_time_varying_sparse_dtlz2_fn

    f, bounds = get_time_varying_sparse_dtlz2_fn(
        dim=dim, num_objectives=num_objectives, k_eff=k_eff, switch_at_eval=switch_at_eval,
    )
    return Benchmark(
        eval_fn=_unit_cube_wrap(f, bounds.to(torch.double)),
        dim=dim, num_objectives=num_objectives, ref_point=[-6.0] * num_objectives,
    )


# ---------------------------------------------------------------------------
# Rover trajectory planning (real, non-algebraic).
# ---------------------------------------------------------------------------
def _make_rover(dim: int = 60, **_) -> Benchmark:
    r"""Trajectory-planning cost simulator: optimize the control points of
    a B-spline path through obstacles, trading off path cost against
    start/goal deviation. Bi-objective (path cost, goal deviation). ``dim``
    must be even and >= 20 (each pair of dims is one 2D waypoint)."""
    from problems.rover import get_rover_fn

    f, bounds = get_rover_fn(dim=dim, force_goal=False, force_start=True)
    return Benchmark(
        eval_fn=_unit_cube_wrap(f, bounds.to(torch.double)),
        dim=dim, num_objectives=2, ref_point=[0.0, -0.5],
    )


def _make_sparse_rover(dim: int, base_dim: int = 60, **_) -> Benchmark:
    r"""``rover`` padded with ``dim - base_dim`` literal no-op input
    dimensions -- tests whether input no-ops alone (without changing the
    underlying landscape's difficulty) are enough to unlock a
    shape-adaptation benefit."""
    from problems.sparse_rover import get_sparse_rover_fn

    f, bounds = get_sparse_rover_fn(dim=dim, base_dim=base_dim, force_goal=False, force_start=True)
    return Benchmark(
        eval_fn=_unit_cube_wrap(f, bounds.to(torch.double)),
        dim=dim, num_objectives=2, ref_point=[0.0, -0.5],
    )


# ---------------------------------------------------------------------------
# BBOB-style landscape taxonomy (project-specific; see honesty caveat in
# problems/bbob_style.py).
# ---------------------------------------------------------------------------
def _make_bbob_biobj(
    dim: int, f1_name: str = "sphere", f2_name: str = "sphere",
    k_eff: Optional[int] = None, ref_point: Optional[List[float]] = None, **_
) -> Benchmark:
    r"""Bi-objective BBOB-style function: pairs two independently-seeded
    base landscape representatives (``f1_name``, ``f2_name`` each one of
    ``"sphere"``, ``"rosenbrock"``, ``"ellipsoidal"``, ``"rastrigin"``,
    ``"peaks"``), the actual ``bbob-biobj`` construction method. Optional
    ``k_eff`` pins dims beyond it to their optimum for both objectives
    (literal no-ops), enabling the same effective-dimension dose-response
    test as ``sparse_dtlz2`` on a non-algebraic landscape. A reference
    point isn't derivable in closed form for these landscapes (unlike
    DTLZ's known geometry) -- pass one explicitly based on empirical
    scale, or accept the default which is unlikely to be tight."""
    from problems.bbob_style import get_bbob_biobj_fn

    f, bounds = get_bbob_biobj_fn(dim=dim, f1_name=f1_name, f2_name=f2_name, k_eff=k_eff)
    return Benchmark(
        eval_fn=_unit_cube_wrap(f, bounds.to(torch.double)),
        dim=dim, num_objectives=2,
        ref_point=ref_point or [-1e6, -1e6],
    )


# ---------------------------------------------------------------------------
# Real-world benchmarks.
# ---------------------------------------------------------------------------
def _make_lasso_bench_mo(bench_name: str = "synt_medium", **_) -> Benchmark:
    r"""Bi-objective LassoBench (Sehic et al. 2022): sparse linear
    regression hyperparameter tuning, trading off prediction MSE against
    an L1 sparsity-inducing regularization penalty coefficient's effect.
    Requires ``pip install LassoBench``. ``bench_name`` is one of the
    synthetic benchmarks (``"synt_simple"``, ``"synt_medium"``,
    ``"synt_high"``, ``"synt_hard"``) or a real dataset name (``"DNA"``,
    ``"Leukemia"``, ``"RCV1"``, ``"Breast_cancer"``, ``"Diabetes"``)."""
    from problems.lasso_bench_mo import get_lasso_bench_mo_fn

    f, bounds, dim = get_lasso_bench_mo_fn(bench_name=bench_name)
    return Benchmark(
        eval_fn=_unit_cube_wrap(f, bounds.to(torch.double)),
        dim=dim, num_objectives=2, ref_point=[-100.0, -1.0],
    )


def _make_penicillin(**_) -> Benchmark:
    r"""BoTorch's ``Penicillin``: a real bioprocess-simulation benchmark
    (fermentation yield vs. time vs. byproduct concentration), 7-dim,
    tri-objective."""
    from botorch.test_functions.multi_objective import Penicillin

    problem = Penicillin(negate=False)
    return Benchmark(
        eval_fn=_unit_cube_wrap(problem, problem.bounds.to(torch.double)),
        dim=problem.dim, num_objectives=problem.num_objectives,
        ref_point=[-1.85, -86.93, -514.70],
    )


def _make_vehicle_safety(**_) -> Benchmark:
    r"""BoTorch's ``VehicleSafety``: 5-dim, tri-objective real vehicle
    frontal-crash design problem (mass, toe-board intrusion, full-frontal
    acceleration -- all lower-raw-value-is-better). Unconstrained variant
    used here (objectives only, no explicit outcome constraints)."""
    from botorch.test_functions.multi_objective import VehicleSafety

    problem = VehicleSafety(negate=False)
    return Benchmark(
        eval_fn=_unit_cube_wrap(problem, problem.bounds.to(torch.double)),
        dim=5, num_objectives=3,
        ref_point=[-1698.549438, -11.205659393, -0.2864599382],
    )


def _make_welded_beam(**_) -> Benchmark:
    r"""BoTorch's ``WeldedBeam``: 4-dim, bi-objective real structural
    design problem (fabrication cost vs. end deflection). Unconstrained
    variant used here (objectives only)."""
    from botorch.test_functions.multi_objective import WeldedBeam

    problem = WeldedBeam(negate=False)
    return Benchmark(
        eval_fn=_unit_cube_wrap(problem, problem.bounds.to(torch.double)),
        dim=4, num_objectives=2, ref_point=[-40.0, -0.015],
    )


# ---------------------------------------------------------------------------
# Registry + entry point.
# ---------------------------------------------------------------------------
BENCHMARKS: dict = {
    "dtlz1": lambda dim, **kw: _make_dtlz("dtlz1", dim, **kw),
    "dtlz2": lambda dim, **kw: _make_dtlz("dtlz2", dim, **kw),
    "dtlz3": lambda dim, **kw: _make_dtlz("dtlz3", dim, **kw),
    "dtlz5": lambda dim, **kw: _make_dtlz("dtlz5", dim, **kw),
    "dtlz7": lambda dim, **kw: _make_dtlz("dtlz7", dim, **kw),
    "composite_dtlz2": _make_composite_dtlz2,
    "sparse_dtlz2": _make_sparse_dtlz2,
    "rotated_sparse_dtlz2": _make_rotated_sparse_dtlz2,
    "time_varying_sparse_dtlz2": _make_time_varying_sparse_dtlz2,
    "rover": _make_rover,
    "sparse_rover": _make_sparse_rover,
    "bbob_biobj": _make_bbob_biobj,
    "lasso_bench_mo": _make_lasso_bench_mo,
    "penicillin": _make_penicillin,
    "vehicle_safety": _make_vehicle_safety,
    "welded_beam": _make_welded_beam,
}


def get_benchmark(name: str, dim: Optional[int] = None, **kwargs) -> Benchmark:
    r"""Construct a ``Benchmark`` by name. See the module docstring for
    the full list of available ``name``s and this file's ``BENCHMARKS``
    dict for exactly which extra ``**kwargs`` each one accepts (e.g.
    ``num_objectives`` for the DTLZ family, ``k_eff`` for the sparse
    variants, ``f1_name``/``f2_name`` for ``bbob_biobj``).

    ``dim`` is optional because a few benchmarks (``penicillin``,
    ``vehicle_safety``, ``welded_beam``) have a fixed native dimension and
    ignore it.
    """
    if name not in BENCHMARKS:
        raise ValueError(f"Unknown benchmark {name!r}. Available: {sorted(BENCHMARKS)}")
    return BENCHMARKS[name](dim=dim, **kwargs)
