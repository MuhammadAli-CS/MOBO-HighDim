r"""CLI: run one (benchmark, method, seed, budget) replication via
``run.py`` (the REAL ``morbo`` engine, not a simplified reimplementation)
and save the result to disk -- the ``plug_and_play`` analogue of the
top-level repo's ``run_comparison.py``.

Saves the raw ``run_one_replication`` output dict (``true_hv``,
``n_evals``, ``X_history``, ``objective_history``, etc. -- see
``morbo/run_one_replication.py``) plus this run's own config to
``results/<study>/<budget>ev/<method>/<seed>.pt``, so ``plot_study.py``
can aggregate across seeds afterward the same way ``plot_aggregate.py``
does for the top-level repo's experiments.

Usage:
    python run_and_save.py --study dtlz2_100d --benchmark dtlz2 --dim 100 \
        --method pca_ellipsoid --seed 0 --budget 600 \
        --benchmark-kwargs '{"num_objectives": 2}'
"""
import argparse
import json
import os

import torch

from benchmarks import get_benchmark
from run import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, help="Results subdirectory name.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True, help="Total evaluation budget (max_evals).")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--n-initial-points", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--n-trust-regions", type=int, default=3)
    parser.add_argument("--min-tr-size", type=int, default=200)
    parser.add_argument("--benchmark-kwargs", type=str, default="{}",
                         help="JSON dict forwarded to the benchmark constructor.")
    args = parser.parse_args()

    bench_kwargs = json.loads(args.benchmark_kwargs)
    ref_point = get_benchmark(args.benchmark, dim=args.dim, **bench_kwargs).ref_point

    result = run(
        benchmark=args.benchmark,
        method=args.method,
        seed=args.seed,
        max_evals=args.budget,
        dim=args.dim,
        batch_size=args.batch_size,
        n_initial_points=args.n_initial_points,
        n_trust_regions=args.n_trust_regions,
        min_tr_size=args.min_tr_size,
        benchmark_kwargs=bench_kwargs,
    )
    final_hv = float(result["true_hv"][-1])
    n_evals_final = int(result["n_evals"][-1])
    assert n_evals_final == args.budget, (
        f"evaluated {n_evals_final} points, expected budget={args.budget}"
    )

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", args.study, f"{args.budget}ev", args.method,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.seed:04d}.pt")
    result["run_config"] = {
        "benchmark": args.benchmark, "method": args.method, "seed": args.seed,
        "budget": args.budget, "dim": args.dim, "n_initial_points": args.n_initial_points,
        "batch_size": args.batch_size, "n_trust_regions": args.n_trust_regions,
        "min_tr_size": args.min_tr_size, "ref_point": ref_point,
    }
    torch.save(result, out_path)
    print(f"[{args.study}] {args.method} seed={args.seed} budget={args.budget}: "
          f"final HV = {final_hv:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
