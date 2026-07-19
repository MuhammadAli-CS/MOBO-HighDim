r"""CLI: run one (benchmark, method, seed, budget) replication via
``run.py`` and save the result to disk -- the ``plug_and_play`` analogue
of the top-level repo's ``run_comparison.py``.

Saves a dict with ``X``, ``Y``, ``hv_history`` (and the run's own config)
to ``results/<study>/<budget>ev/<method>/<seed>.pt``, so
``plot_study.py`` can aggregate across seeds afterward the same way
``plot_aggregate.py`` does for the top-level repo's experiments.

Usage:
    python run_and_save.py --study dtlz2_100d --benchmark dtlz2 --dim 100 \
        --method pca_ellipsoid --seed 0 --budget 600 \
        --benchmark-kwargs '{"num_objectives": 2}'
"""
import argparse
import json
import os

import torch

from run import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True, help="Results subdirectory name.")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, required=True, help="Total evaluation budget.")
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--n-init", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--benchmark-kwargs", type=str, default="{}",
                         help="JSON dict forwarded to the benchmark constructor.")
    args = parser.parse_args()

    if (args.budget - args.n_init) % args.batch_size != 0:
        raise ValueError(
            f"budget - n_init ({args.budget - args.n_init}) must be a multiple of "
            f"batch_size ({args.batch_size})."
        )
    n_iter = (args.budget - args.n_init) // args.batch_size

    result = run(
        benchmark=args.benchmark,
        method=args.method,
        seed=args.seed,
        dim=args.dim,
        n_init=args.n_init,
        n_iter=n_iter,
        batch_size=args.batch_size,
        benchmark_kwargs=json.loads(args.benchmark_kwargs),
    )
    assert result["X"].shape[0] == args.budget, (
        f"evaluated {result['X'].shape[0]} points, expected budget={args.budget}"
    )

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results", args.study, f"{args.budget}ev", args.method,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.seed:04d}.pt")
    torch.save(
        {
            "X": result["X"], "Y": result["Y"], "hv_history": result["hv_history"],
            "final_hypervolume": result["final_hypervolume"],
            "benchmark": args.benchmark, "method": args.method, "seed": args.seed,
            "budget": args.budget, "dim": args.dim, "n_init": args.n_init,
            "batch_size": args.batch_size,
        },
        out_path,
    )
    print(f"[{args.study}] {args.method} seed={args.seed} budget={args.budget}: "
          f"final HV = {result['final_hypervolume']:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
