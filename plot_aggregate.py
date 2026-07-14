#!/usr/bin/env python3
r"""Aggregate (multi-seed) hypervolume comparison: mean curve +/- uncertainty.

The single-seed `plot_comparison.py` panels were the right tool when
everything was seed 0; with 5-30 seeds per method the standard
presentation (as in the MORBO and LassoBench papers) is a mean
hypervolume-vs-evaluations curve per method with an uncertainty band.
This script produces exactly that, plus a final-HV summary table.

- Discovers every label with >= 1 saved seed (same convention as
  plot_comparison.py); uses however many seeds each label has and reports
  the count in the legend.
- Recomputes the HV trace per seed from `objective_history` with the same
  reference point for every method (identical convention to
  plot_comparison.py -- some labels don't track HV online).
- Band = +/- 1 SEM by default (`--band std` for +/- 1 std). SEM answers
  "how well is the MEAN pinned down" (the cross-method comparison
  question); std shows per-run spread.

Usage:
    python plot_aggregate.py <experiment_name> [--labels a b c] [--band std] [--step N]
e.g.
    python plot_aggregate.py tr_shape_dtlz2_100d
    python plot_aggregate.py lasso_dna_mo --band std
Writes experiments/<experiment_name>/comparison_aggregate.png
"""
import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_comparison import hv_trace, objective_Y, title_for


def discover_label_seeds(exp_dir: str):
    """Map label -> sorted list of seeds with saved results."""
    out = {}
    for name in sorted(os.listdir(exp_dir)):
        sub = os.path.join(exp_dir, name)
        if not os.path.isdir(sub):
            continue
        seeds = []
        for fname in os.listdir(sub):
            m = re.match(rf"(\d+)_{re.escape(name)}\.pt$", fname)
            if m:
                seeds.append(int(m.group(1)))
        if seeds:
            out[name] = sorted(seeds)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--band", choices=["sem", "std"], default="sem")
    parser.add_argument(
        "--step", type=int, default=None,
        help="Eval-count stride between HV checkpoints (default: ~40 points/curve).",
    )
    args = parser.parse_args()

    exp_name = args.experiment_name
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, "experiments", exp_name)
    with open(os.path.join(exp_dir, "config.json")) as f:
        config = json.load(f)
    ref_point = torch.tensor(config["max_reference_point"], dtype=torch.double)
    step = args.step or max(config.get("batch_size", 10), config["max_evals"] // 40)

    label_seeds = discover_label_seeds(exp_dir)
    if args.labels:
        label_seeds = {k: v for k, v in label_seeds.items() if k in args.labels}
    if not label_seeds:
        raise SystemExit(f"No saved results under {exp_dir}.")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    summary = []
    for label, seeds in label_seeds.items():
        traces = []
        ns_ref = None
        for s in seeds:
            path = os.path.join(exp_dir, label, f"{str(s).zfill(4)}_{label}.pt")
            out = torch.load(path, map_location="cpu", weights_only=False)
            Y = objective_Y(out)
            ns, hvs = hv_trace(Y, ref_point, step=step)
            if ns_ref is None:
                ns_ref = ns
            # Align on the shortest common prefix (a killed/requeued run can
            # be a few evals short; protocol is otherwise identical per exp).
            m = min(len(ns_ref), len(ns))
            ns_ref = ns_ref[:m]
            traces = [t[:m] for t in traces]
            traces.append(hvs[:m])
        A = np.array(traces)  # (n_seeds, n_checkpoints)
        mean = A.mean(axis=0)
        spread = A.std(axis=0, ddof=1) if A.shape[0] > 1 else np.zeros_like(mean)
        if args.band == "sem" and A.shape[0] > 1:
            spread = spread / np.sqrt(A.shape[0])
        (line,) = ax.plot(ns_ref, mean, label=f"{title_for(label)} (n={A.shape[0]})")
        ax.fill_between(ns_ref, mean - spread, mean + spread,
                        alpha=0.2, color=line.get_color(), linewidth=0)
        summary.append((label, A.shape[0], A[:, -1].mean(), A[:, -1].std(ddof=1) if A.shape[0] > 1 else 0.0))

    ax.axvline(config["n_initial_points"], color="gray", ls=":", lw=1,
               label="End of initialization")
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel(f"Hypervolume (ref point {config['max_reference_point']})")
    band_name = "±1 SEM" if args.band == "sem" else "±1 std"
    ax.set_title(
        f"{exp_name}: mean hypervolume over seeds ({band_name} band)\n"
        f"d={config['dim']}, {config['max_evals']} evals, batch {config['batch_size']}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(exp_dir, "comparison_aggregate.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}\n")

    print(f"{'method':<26} {'n':>3} {'final HV mean':>14} {'std':>8}")
    for label, n, m, s in sorted(summary, key=lambda r: -r[2]):
        print(f"{label:<26} {n:>3} {m:>14.4f} {s:>8.4f}")
