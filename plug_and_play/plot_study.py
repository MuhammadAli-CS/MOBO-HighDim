r"""Aggregate (multi-seed) hypervolume comparison for a ``run_and_save.py``
study -- the ``plug_and_play`` analogue of the top-level repo's
``plot_aggregate.py``: mean HV-vs-evals curve per method, +/- 1 SEM band.

Recomputes hypervolume from ``objective_history`` at fixed evaluation-count
checkpoints (same approach as ``plot_aggregate.py``/``plot_comparison.py``'s
``hv_trace``, reimplemented here self-contained) rather than relying on
``true_hv``'s own checkpoint schedule -- ``objective_history`` always has
exactly one row per evaluation, so this sidesteps any risk of different
runs' checkpoint schedules not lining up.

Usage:
    python plot_study.py <study> <budget>
e.g.
    python plot_study.py dtlz2_100d 600
Writes results/<study>/<budget>ev/comparison.png
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning


def discover_method_seeds(budget_dir: str):
    """Map method -> sorted list of seeds with saved results."""
    out = {}
    for name in sorted(os.listdir(budget_dir)):
        sub = os.path.join(budget_dir, name)
        if not os.path.isdir(sub):
            continue
        seeds = [int(m.group(1)) for f in os.listdir(sub)
                 if (m := re.match(r"(\d+)\.pt$", f))]
        if seeds:
            out[name] = sorted(seeds)
    return out


def hv_trace(Y: torch.Tensor, ref_point: torch.Tensor, step: int):
    """Running hypervolume of Y[:k] for k = step, 2*step, ..., up to Y.shape[0]."""
    ns, hvs = [], []
    for k in range(step, Y.shape[0] + 1, step):
        Yk = Y[:k]
        better = (Yk > ref_point).all(dim=-1)
        if better.any():
            part = DominatedPartitioning(ref_point=ref_point, Y=Yk[better])
            hv = part.compute_hypervolume().item()
        else:
            hv = 0.0
        ns.append(k)
        hvs.append(hv)
    return ns, hvs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("study")
    parser.add_argument("budget", type=int)
    parser.add_argument("--band", choices=["sem", "std"], default="sem")
    parser.add_argument("--step", type=int, default=None,
                         help="Eval-count stride between plotted points (default: ~40/curve).")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    budget_dir = os.path.join(root, "results", args.study, f"{args.budget}ev")
    if not os.path.isdir(budget_dir):
        raise SystemExit(f"No results under {budget_dir}.")
    method_seeds = discover_method_seeds(budget_dir)
    if not method_seeds:
        raise SystemExit(f"No saved seeds under {budget_dir}.")

    step = args.step or max(1, args.budget // 40)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    summary = []
    n_init = None
    for method, seeds in method_seeds.items():
        traces = []
        ns_ref = None
        for s in seeds:
            path = os.path.join(budget_dir, method, f"{s:04d}.pt")
            out = torch.load(path, map_location="cpu", weights_only=False)
            Y = out["objective_history"].double()
            ref_point = torch.as_tensor(out["run_config"]["ref_point"], dtype=torch.double)
            n_init = out["run_config"]["n_initial_points"]
            ns, hvs = hv_trace(Y, ref_point, step=step)
            if ns_ref is None:
                ns_ref = ns
            traces.append(hvs)
        traces = np.array(traces)  # n_seeds x n_checkpoints
        mean = traces.mean(axis=0)
        spread = traces.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros_like(mean)
        if args.band == "sem" and len(seeds) > 1:
            spread = spread / np.sqrt(len(seeds))
        line, = ax.plot(ns_ref, mean, label=f"{method} (n={len(seeds)})")
        ax.fill_between(ns_ref, mean - spread, mean + spread, alpha=0.2, color=line.get_color())
        summary.append((method, mean[-1], spread[-1], len(seeds)))

    if n_init:
        ax.axvline(n_init, color="gray", linestyle=":", linewidth=1, label="End of initialization")

    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Hypervolume")
    ax.set_title(f"{args.study}: mean hypervolume over seeds (± 1 {args.band.upper()} band)\n"
                 f"plug_and_play (real morbo engine), {args.budget} evals")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(budget_dir, "comparison.png")
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}\n")

    print(f"{'method':22s} {'final HV':>12s} {'+/-':>10s} {'n':>4s}")
    for method, mean, spread, n in sorted(summary, key=lambda r: -r[1]):
        print(f"{method:22s} {mean:12.4f} {spread:10.4f} {n:4d}")


if __name__ == "__main__":
    main()
