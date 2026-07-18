r"""Minimal driver tying ``benchmarks.py`` and ``methods.py`` together into
an actual, runnable multi-objective BO experiment.

**Design note, stated plainly**: this file does NOT reimplement a
Bayesian optimization loop from scratch. ``methods.py``'s shape functions
are already exactly what powers this project's full MORBO implementation
(``morbo/trust_region.py``'s ``TurboHParams(tr_shape=...)``) -- rewriting
a second, parallel BO loop here would be a lot of new, untested code for
no benefit, and would risk silently diverging from the validated engine
these methods were developed and measured against. Instead, ``run()``
below is a thin translation layer: it looks up a benchmark's spec (from
``benchmarks.py``) and a method's ``tr_shape`` (from ``methods.py``'s
naming), then calls the existing, tested
``morbo.run_one_replication.run_one_replication`` with the right
arguments. ``benchmarks.py``/``methods.py`` stay fully standalone and
importable into any other project regardless (they have zero dependency
on this file or the rest of this repo); ``run.py`` is just the
convenience path for running an experiment inside this one.

Usage (Python):

    from run import run
    result = run(
        benchmark="dtlz2", method="pca_ellipsoid",
        dim=100, seed=0, max_evals=600,
    )
    print(result["final_hypervolume"])

Usage (CLI):

    python run.py --benchmark dtlz2 --method pca_ellipsoid --dim 100 \
        --seed 0 --max-evals 600
"""
import argparse
import os
import sys
from typing import Any, Dict, Optional

# Make both this directory (for `benchmarks`/`methods`) and the repo root
# (for the `morbo` package) importable, regardless of the working
# directory `run.py` is launched from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from benchmarks import get_benchmark  # noqa: E402

# tr_shape adaptation methods this project implements, in one place --
# see methods.py for the actual mechanism behind each (this dict is only
# the name -> `tr_shape` string mapping `run_one_replication` expects;
# the real, standalone implementations live in methods.py).
METHODS = {
    "isotropic": {"tr_shape": "isotropic"},
    "ard_box": {"tr_shape": "ard_box"},
    "pca_ellipsoid": {"tr_shape": "pca_ellipsoid"},
    "ard_pca_ellipsoid": {"tr_shape": "ard_pca_ellipsoid"},
    "cma_ellipsoid": {"tr_shape": "cma_ellipsoid"},
    "labcat_style": {"tr_shape": "labcat_style"},
    "mab_shape": {"tr_shape": "mab_shape", "mab_policy": "ducb"},
}

# benchmark name (as in benchmarks.py) -> the `evalfn` string
# `run_one_replication` dispatches on, plus which of its own kwargs a
# benchmark's extra parameters (num_objectives, k_eff, ...) map to.
_EVALFN_MAP = {
    "dtlz1": "DTLZ1", "dtlz2": "DTLZ2", "dtlz3": "DTLZ3", "dtlz5": "DTLZ5", "dtlz7": "DTLZ7",
    "sparse_dtlz2": "SparseDTLZ2",
    "rotated_sparse_dtlz2": "RotatedSparseDTLZ2",
    "time_varying_sparse_dtlz2": "TimeVaryingSparseDTLZ2",
    "rover": "rover",
    "sparse_rover": "SparseRover",
    "bbob_biobj": "BBOBBiObj",
    "lasso_bench_mo": "LassoBenchMO",
    "penicillin": "Penicillin",
    "vehicle_safety": "VehicleSafety",
    "welded_beam": "WeldedBeam",
}


def run(
    benchmark: str,
    method: str,
    seed: int,
    max_evals: int,
    dim: Optional[int] = None,
    batch_size: int = 50,
    n_initial_points: int = 200,
    n_trust_regions: int = 3,
    min_tr_size: int = 200,
    benchmark_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    r"""Run one full BO replication: ``method`` (a key into ``METHODS``,
    see ``methods.py`` for what each one actually does) on ``benchmark``
    (a key into ``benchmarks.BENCHMARKS``).

    Args:
        benchmark: benchmark name, e.g. ``"dtlz2"``, ``"bbob_biobj"``.
        method: shape-adaptation method name, e.g. ``"pca_ellipsoid"``.
        seed: random seed (controls Sobol init and any stochastic
            benchmark construction, e.g. BBOB's random rotation/shift).
        max_evals: total evaluation budget.
        dim: input dimension (ignored by fixed-dimension benchmarks like
            ``"penicillin"``).
        batch_size, n_initial_points, n_trust_regions, min_tr_size:
            standard MORBO hyperparameters; the defaults match this
            project's own experiments.
        benchmark_kwargs: extra kwargs forwarded to the benchmark
            constructor (e.g. ``{"num_objectives": 4}`` for a DTLZ
            variant, ``{"k_eff": 20}`` for a sparse variant,
            ``{"f1_name": "rosenbrock", "f2_name": "peaks"}`` for
            ``bbob_biobj``).
        verbose: print per-iteration trust-region state.

    Returns:
        The raw result dict from ``run_one_replication`` -- notably
        ``result["true_hv"]`` (hypervolume trace) and
        ``result["n_evals"]`` (evaluation count at each trace point).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Available: {sorted(METHODS)}")
    if benchmark not in _EVALFN_MAP:
        raise ValueError(f"Unknown benchmark {benchmark!r}. Available: {sorted(_EVALFN_MAP)}")

    bench_kwargs = dict(benchmark_kwargs or {})
    spec = get_benchmark(benchmark, dim=dim, **bench_kwargs)

    from morbo.run_one_replication import run_one_replication

    extra_kwargs: Dict[str, Any] = {}
    if benchmark == "sparse_dtlz2" or benchmark == "rotated_sparse_dtlz2" or benchmark == "time_varying_sparse_dtlz2":
        extra_kwargs["sparse_dtlz2_k_eff"] = bench_kwargs.get("k_eff", 5)
    if benchmark == "sparse_rover":
        extra_kwargs["sparse_rover_base_dim"] = bench_kwargs.get("base_dim", 60)
    if benchmark == "lasso_bench_mo":
        extra_kwargs["lasso_bench_name"] = bench_kwargs.get("bench_name", "synt_medium")
    if benchmark == "bbob_biobj":
        extra_kwargs["bbob_f1"] = bench_kwargs.get("f1_name", "sphere")
        extra_kwargs["bbob_f2"] = bench_kwargs.get("f2_name", "sphere")
        extra_kwargs["bbob_k_eff"] = bench_kwargs.get("k_eff", None)

    outputs = []
    run_one_replication(
        seed=seed,
        label=method,
        max_evals=max_evals,
        evalfn=_EVALFN_MAP[benchmark],
        dim=spec.dim,
        batch_size=batch_size,
        n_initial_points=n_initial_points,
        n_trust_regions=n_trust_regions,
        min_tr_size=min_tr_size,
        max_reference_point=spec.ref_point,
        verbose=verbose,
        save_callback=lambda output: outputs.append(output),
        **METHODS[method],
        **extra_kwargs,
    )
    return outputs[-1]


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(_EVALFN_MAP))
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-evals", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--n-initial-points", type=int, default=200)
    parser.add_argument("--min-tr-size", type=int, default=200)
    parser.add_argument("--n-trust-regions", type=int, default=3)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    result = run(
        benchmark=args.benchmark,
        method=args.method,
        seed=args.seed,
        max_evals=args.max_evals,
        dim=args.dim,
        batch_size=args.batch_size,
        n_initial_points=args.n_initial_points,
        min_tr_size=args.min_tr_size,
        n_trust_regions=args.n_trust_regions,
        verbose=args.verbose,
    )
    final_hv = float(result["true_hv"][-1])
    print(f"{args.benchmark} / {args.method} (seed {args.seed}): "
          f"final hypervolume = {final_hv:.4f}")


if __name__ == "__main__":
    _cli()
