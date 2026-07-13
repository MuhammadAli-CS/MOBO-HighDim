#!/usr/bin/env python3
r"""Compute-efficiency comparison: optimizer wall-time vs. hypervolume.

Answers "how much computational power does each method spend for its
hypervolume improvement" using timing data every saved run already
records -- `fit_times` (model fitting per BO iteration) and `gen_times`
(candidate generation per BO iteration) -- so this is pure post-processing
of committed `.pt` results; nothing needs to be rerun.

Produces, for each experiment/seed:
  1. Cumulative optimizer wall-time (s) vs. hypervolume -- the efficiency
     trace. A method that reaches high HV further LEFT is cheaper per unit
     of hypervolume, regardless of how many evaluations it used.
  2. Total optimizer time vs. final HV scatter (one point per method) --
     the "efficiency frontier" summary view.
  3. A printed per-method table: total fit time, total gen time, final HV,
     and HV per optimizer-minute.

Time here is OPTIMIZER overhead only (GP fitting + candidate generation),
not function-evaluation time -- on synthetic benchmarks like DTLZ2 the
evaluation itself is microseconds, so optimizer overhead is the real
compute cost; on expensive real problems evaluation cost would dominate
and equal-eval comparisons (the existing plots) are the right lens instead.

CAVEAT: wall-times are only comparable across runs executed on the same
hardware. All cluster runs here used one B200 GPU each, so within-experiment
comparisons are valid; do not mix laptop-CPU-era results into the same plot.

Usage:
    python plot_efficiency.py <experiment_name> <seed> [--labels a b c] [--log-time]
e.g.
    python plot_efficiency.py tr_shape_dtlz2_100d 0
    python plot_efficiency.py tr_shape_methods_dtlz2_100d 0 --log-time
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)

from plot_comparison import discover_labels, objective_Y, title_for


def hv_at_checkpoints(Y: torch.Tensor, ref_point: torch.Tensor, checkpoints):
    """Hypervolume of Y[:k] at each eval-count checkpoint k.

    Recomputed identically for every label (some, e.g. scalarized runs,
    don't track HV online; sobol tracks it but through its own path) --
    same convention as plot_comparison.py's hv_trace.
    """
    hvs = []
    for k in checkpoints:
        Yk = Y[:k]
        better = (Yk > ref_point).all(dim=-1)
        if better.any():
            part = DominatedPartitioning(ref_point=ref_point, Y=Yk[better])
            hvs.append(part.compute_hypervolume().item())
        else:
            hvs.append(0.0)
    return hvs


def cumulative_optimizer_time(out: dict):
    """Cumulative fit+gen wall-time (s) at each BO iteration.

    `fit_times` and `gen_times` are appended once per BO iteration in
    run_one_replication (matching `n_evals`); the sobol label records only
    gen_times (its fit_times is empty -- it fits nothing), so the shorter
    list is zero-padded to the longer one's length.
    """
    fit = list(out.get("fit_times", []) or [])
    gen = list(out.get("gen_times", []) or [])
    n = max(len(fit), len(gen))
    fit += [0.0] * (n - len(fit))
    gen += [0.0] * (n - len(gen))
    per_iter = [f + g for f, g in zip(fit, gen)]
    cum, total = [], 0.0
    for t in per_iter:
        total += t
        cum.append(total)
    return cum, sum(fit), sum(gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", nargs="?", default="tr_shape_dtlz2_100d")
    parser.add_argument("seed", nargs="?", type=int, default=0)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels to plot. Defaults to auto-discovery like plot_comparison.py.",
    )
    parser.add_argument(
        "--log-time",
        action="store_true",
        help="Log-scale the time axis (useful when methods' costs span orders "
        "of magnitude, e.g. sobol vs. GP-based methods).",
    )
    args = parser.parse_args()

    exp_name = args.experiment_name
    seed = args.seed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, "experiments", exp_name)
    with open(os.path.join(exp_dir, "config.json")) as f:
        config = json.load(f)
    ref_point = torch.tensor(config["max_reference_point"], dtype=torch.double)

    requested = args.labels if args.labels is not None else discover_labels(exp_dir, seed)
    if not requested:
        raise SystemExit(f"No saved results found under {exp_dir} for seed {seed}.")

    rows = []  # (label, cum_times, hvs, total_fit, total_gen, final_hv)
    for label in requested:
        path = os.path.join(exp_dir, label, f"{str(seed).zfill(4)}_{label}.pt")
        if not os.path.exists(path):
            print(f"Skipping '{label}': no saved result at {path}")
            continue
        out = torch.load(path, map_location="cpu", weights_only=False)
        cum_times, total_fit, total_gen = cumulative_optimizer_time(out)
        if not cum_times:
            print(f"Skipping '{label}': no fit_times/gen_times recorded.")
            continue
        n_evals = list(out["n_evals"])
        # Older TRBO saves can have one fewer timing entry than n_evals
        # checkpoints (or vice versa) if a run was killed mid-iteration;
        # align on the shorter of the two.
        m = min(len(cum_times), len(n_evals))
        cum_times, n_evals = cum_times[:m], n_evals[:m]
        Y = objective_Y(out)
        hvs = hv_at_checkpoints(Y, ref_point, n_evals)
        rows.append((label, cum_times, hvs, total_fit, total_gen, hvs[-1]))

    if not rows:
        raise SystemExit("None of the requested labels have usable timing data.")

    fig, (trace_ax, frontier_ax) = plt.subplots(1, 2, figsize=(12, 5))

    # --- 1. Efficiency trace: cumulative optimizer time vs. HV ---
    for label, cum_times, hvs, *_ in rows:
        trace_ax.plot(cum_times, hvs, label=title_for(label), marker=".", ms=3)
    trace_ax.set_xlabel("Cumulative optimizer wall-time (s): GP fitting + candidate generation")
    trace_ax.set_ylabel(f"Hypervolume (ref point {config['max_reference_point']})")
    trace_ax.set_title("Efficiency trace: compute spent vs. hypervolume reached")
    if args.log_time:
        trace_ax.set_xscale("log")
    trace_ax.legend(fontsize=8)

    # --- 2. Efficiency frontier: total time vs. final HV, one point/method ---
    for label, cum_times, hvs, total_fit, total_gen, final_hv in rows:
        total = cum_times[-1]
        frontier_ax.scatter(total, final_hv, s=60)
        frontier_ax.annotate(
            title_for(label),
            (total, final_hv),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )
    frontier_ax.set_xlabel("Total optimizer wall-time (s)")
    frontier_ax.set_ylabel("Final hypervolume")
    frontier_ax.set_title("Efficiency frontier (up-left is better)")
    if args.log_time:
        frontier_ax.set_xscale("log")

    fig.suptitle(
        f"{exp_name} (seed {seed}): d={config['dim']}, "
        f"{config['max_evals']} evals, batch {config['batch_size']} -- "
        "optimizer overhead only; evaluation time excluded"
    )
    fig.tight_layout()
    out_path = os.path.join(exp_dir, f"efficiency_seed{seed}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}\n")

    # --- 3. Summary table ---
    print(f"{'method':<28} {'fit (s)':>10} {'gen (s)':>10} {'total (s)':>10} {'final HV':>10} {'HV/min':>10}")
    for label, cum_times, hvs, total_fit, total_gen, final_hv in rows:
        total = cum_times[-1]
        hv_per_min = final_hv / (total / 60.0) if total > 0 else float("inf")
        print(
            f"{label:<28} {total_fit:>10.1f} {total_gen:>10.1f} {total:>10.1f} "
            f"{final_hv:>10.3f} {hv_per_min:>10.3f}"
        )
