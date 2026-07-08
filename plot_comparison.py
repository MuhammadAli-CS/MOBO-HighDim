#!/usr/bin/env python3
r"""Plot an N-way MORBO variant comparison (Figure-2-style).

Produces, for each experiment:
  1. Objective-space scatter per method (all evaluated points, colored by
     trust region when attribution is available) with the identified Pareto
     frontier -- the paper's Figure 2 layout, generalized to N methods.
  2. Hypervolume vs. function evaluations, computed post-hoc from
     `metric_history` with the same reference point for every method (some
     labels, e.g. scalarized runs, don't track hypervolume online).

Labels are auto-discovered from whichever `<seed>_<label>.pt` files already
exist under `experiments/<experiment_name>/` -- running a new method never
requires rerunning or touching previously-saved results, so this script
picks up new labels automatically on the next invocation.

Usage:
    python plot_comparison.py <experiment_name> <seed> [--labels a b c]
e.g.
    python plot_comparison.py fig2_dtlz2_100d 0
    python plot_comparison.py fig2_dtlz2_100d 0 --labels morbo composite_morbo
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated

# Hand-written display names for known labels; anything else falls back to a
# title-cased version of the label itself (see `title_for`).
TITLES = {
    "morbo": "MORBO",
    "turbo_scalarized": "TuRBO + Chebyshev scalarizations",
    "composite_morbo": "Composite MORBO",
}


def title_for(label: str) -> str:
    return TITLES.get(label, label.replace("_", " ").title())


def objective_Y(out: dict) -> torch.Tensor:
    r"""Objective-space observations for a saved run.

    Prefers `objective_history` (always in objective space, i.e. already
    reduced for composite runs). Falls back to `metric_history` for runs
    saved before that field existed, where the two are identical anyway
    (non-composite runs have no reduction to apply).
    """
    key = "objective_history" if "objective_history" in out else "metric_history"
    return out[key].double()


def discover_labels(exp_dir: str, seed: int):
    """Find every label with a saved `<seed>_<label>.pt` file under exp_dir."""
    labels = []
    for name in sorted(os.listdir(exp_dir)):
        sub = os.path.join(exp_dir, name)
        if not os.path.isdir(sub):
            continue
        if os.path.exists(os.path.join(sub, f"{str(seed).zfill(4)}_{name}.pt")):
            labels.append(name)
    return labels


def hv_trace(Y: torch.Tensor, ref_point: torch.Tensor, step: int = 1):
    """Running hypervolume of Y[:k] for k = 1..n (every `step` evals)."""
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name", nargs="?", default="fig2_dtlz2_100d")
    parser.add_argument("seed", nargs="?", type=int, default=0)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels to plot. Defaults to auto-discovering every label with "
        "a saved result for this seed under experiments/<experiment_name>/.",
    )
    args = parser.parse_args()

    exp_name = args.experiment_name
    seed = args.seed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, "experiments", exp_name)
    with open(os.path.join(exp_dir, "config.json")) as f:
        config = json.load(f)
    ref_point = torch.tensor(config["max_reference_point"], dtype=torch.double)
    n_init = config["n_initial_points"]

    requested = args.labels if args.labels is not None else discover_labels(exp_dir, seed)
    if not requested:
        raise SystemExit(f"No saved results found under {exp_dir} for seed {seed}.")

    data = {}
    for label in requested:
        path = os.path.join(exp_dir, label, f"{str(seed).zfill(4)}_{label}.pt")
        if not os.path.exists(path):
            print(f"Skipping '{label}': no saved result at {path}")
            continue
        data[label] = torch.load(path, weights_only=False)
    labels = list(data.keys())
    if not labels:
        raise SystemExit("None of the requested labels have saved results.")

    n = len(labels)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    if n + 1 == 1:
        axes = [axes]

    # --- Objective-space scatter (Figure 2 style), one panel per method ---
    tr_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    scatter_axes = axes[:n]
    for ax, label in zip(scatter_axes, labels):
        out = data[label]
        Y = objective_Y(out)
        tr_idx = out.get("tr_indices")
        ax.scatter(
            Y[:n_init, 0],
            Y[:n_init, 1],
            c="lightgray",
            s=12,
            label="Initial points",
        )
        post = Y[n_init:]
        if tr_idx is not None and len(tr_idx) == Y.shape[0]:
            tr_idx_post = torch.tensor(tr_idx[n_init:])
            for t in sorted(set(tr_idx_post.tolist())):
                mask = tr_idx_post == t
                name = f"Trust region {t + 1}" if t >= 0 else "Restart point"
                color = tr_colors[t % len(tr_colors)] if t >= 0 else "black"
                ax.scatter(
                    post[mask, 0], post[mask, 1], c=color, s=12, label=name
                )
        else:
            ax.scatter(post[:, 0], post[:, 1], c="tab:blue", s=12, label="BO points")
        pareto_mask = is_non_dominated(Y)
        pf = Y[pareto_mask]
        pf = pf[pf[:, 0].argsort()]
        # Draw the frontier as a step function (right-angle staircase), not a
        # diagonal interpolation: this is the actual boundary of the
        # dominated region for a 2-objective (maximization) Pareto set.
        ax.step(pf[:, 0], pf[:, 1], where="post", color="k", lw=1.5, label="Pareto frontier")
        ax.set_title(title_for(label))
        ax.set_xlabel("Objective 1")
        ax.set_ylabel("Objective 2")
        ax.legend(fontsize=7)

    # Use the same axis limits across all scatter plots
    xlims = [a.get_xlim() for a in scatter_axes]
    ylims = [a.get_ylim() for a in scatter_axes]
    for a in scatter_axes:
        a.set_xlim(min(x[0] for x in xlims), max(x[1] for x in xlims))
        a.set_ylim(min(y[0] for y in ylims), max(y[1] for y in ylims))

    # --- Hypervolume traces (computed identically for every method) ---
    hv_ax = axes[n]
    for label in labels:
        Y = objective_Y(data[label])
        ns, hvs = hv_trace(Y, ref_point, step=10)
        hv_ax.plot(ns, hvs, label=title_for(label))
    hv_ax.axvline(n_init, color="gray", ls=":", lw=1, label="End of initialization")
    hv_ax.set_xlabel("Function evaluations")
    hv_ax.set_ylabel(f"Hypervolume (ref point {config['max_reference_point']})")
    hv_ax.set_title("Hypervolume vs. evaluations")
    hv_ax.legend(fontsize=8)

    fig.suptitle(
        f"{exp_name} (seed {seed}): d={config['dim']}, "
        f"{config['max_evals']} evals, batch {config['batch_size']}, "
        f"{config.get('n_trust_regions', 5)} TRs"
    )
    fig.tight_layout()
    out_path = os.path.join(exp_dir, f"comparison_seed{seed}.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    # Print summary numbers
    for label in labels:
        Y = objective_Y(data[label])
        better = (Y > ref_point).all(dim=-1)
        part = DominatedPartitioning(ref_point=ref_point, Y=Y[better])
        pf_size = int(is_non_dominated(Y).sum())
        print(
            f"{title_for(label)}: final HV = {part.compute_hypervolume().item():.4f}, "
            f"Pareto set size = {pf_size}, evals = {Y.shape[0]}"
        )
