r"""Minimal driver tying ``benchmarks.py`` and ``methods.py`` together into
an actual, runnable multi-objective BO experiment -- fully self-contained
within this folder (no dependency on this repo's top-level ``morbo``
package; only ``torch``/``botorch``/``gpytorch`` are required).

Usage (Python):

    from run import run
    result = run(benchmark="dtlz2", method="pca_ellipsoid", dim=20, seed=0, n_iter=20)
    print(result["final_hypervolume"])

Usage (CLI):

    python run.py --benchmark dtlz2 --method pca_ellipsoid --dim 20 --seed 0 --n-iter 20
"""
import argparse
from typing import Any, Dict, Optional

from benchmarks import BENCHMARKS, get_benchmark
from methods import SHAPE_METHODS
from optimizer import optimize

# Every method this folder implements. "mab_shape" isn't a key in
# methods.SHAPE_METHODS (it's a meta-strategy, see methods.py), so it's
# added here explicitly.
METHODS = list(SHAPE_METHODS) + ["mab_shape"]


def run(
    benchmark: str,
    method: str,
    seed: int,
    dim: Optional[int] = None,
    n_init: int = 20,
    n_iter: int = 40,
    batch_size: int = 5,
    length_init: float = 0.8,
    length_min: float = 0.01,
    length_max: float = 1.6,
    n_candidates: int = 512,
    benchmark_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    r"""Run one full BO replication: ``method`` (see ``methods.py``) on
    ``benchmark`` (a key into ``benchmarks.BENCHMARKS``).

    Args:
        benchmark: benchmark name, e.g. ``"dtlz2"``, ``"bbob_biobj"``.
        method: shape-adaptation method name, e.g. ``"pca_ellipsoid"``.
        seed: random seed (Sobol init, candidate sampling, and any
            stochastic benchmark construction, e.g. BBOB's random
            rotation/shift).
        dim: input dimension (ignored by fixed-dimension benchmarks like
            ``"penicillin"``).
        n_init: Sobol-initialization evaluation count.
        n_iter: number of BO iterations, each proposing ``batch_size`` points
            (total evaluations = ``n_init + n_iter * batch_size``).
        batch_size: candidates evaluated per iteration.
        length_init, length_min, length_max: trust-region edge-length bounds.
        n_candidates: discrete candidate-pool size scored each iteration.
        benchmark_kwargs: extra kwargs forwarded to the benchmark
            constructor (e.g. ``{"num_objectives": 4}``, ``{"k_eff": 20}``,
            ``{"f1_name": "rosenbrock", "f2_name": "peaks"}``).

    Returns:
        ``{"X": ..., "Y": ..., "hv_history": ..., "final_hypervolume": ...}``.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Available: {sorted(METHODS)}")

    spec = get_benchmark(benchmark, dim=dim, **(benchmark_kwargs or {}))
    result = optimize(
        evaluate=spec.eval_fn,
        dim=spec.dim,
        ref_point=spec.ref_point,
        method=method,
        n_init=n_init,
        n_iter=n_iter,
        batch_size=batch_size,
        seed=seed,
        length_init=length_init,
        length_min=length_min,
        length_max=length_max,
        n_candidates=n_candidates,
    )
    return {
        "X": result.X,
        "Y": result.Y,
        "hv_history": result.hv_history,
        "final_hypervolume": float(result.hv_history[-1]),
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True, choices=sorted(BENCHMARKS))
    parser.add_argument("--method", required=True, choices=sorted(METHODS))
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=20)
    parser.add_argument("--n-iter", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    result = run(
        benchmark=args.benchmark,
        method=args.method,
        seed=args.seed,
        dim=args.dim,
        n_init=args.n_init,
        n_iter=args.n_iter,
        batch_size=args.batch_size,
    )
    n_evals = args.n_init + args.n_iter * args.batch_size
    print(f"{args.benchmark} / {args.method} (seed {args.seed}, {n_evals} evals): "
          f"final hypervolume = {result['final_hypervolume']:.4f}")


if __name__ == "__main__":
    _cli()
