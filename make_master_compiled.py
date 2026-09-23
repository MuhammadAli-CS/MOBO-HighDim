#!/usr/bin/env python3
r"""Master compiled-results grid: every composite-vs-direct benchmark in one
figure, in the established hypervolume-vs-evaluations trajectory format.

Two blocks:
  A. This project's own composite benchmarks -- run through the real MORBO
     engine. Each panel plots the direct `morbo` baseline against the
     primary `composite_*` label, mean +/- 1 SEM over seeds, recomputing the
     HV trace per seed with a shared reference point (same convention as
     plot_aggregate.py).
  B. Collaborator (tau315/composite-mobo) ablation benchmarks -- each panel
     plots every solver pair's direct (solid) vs composite (dashed) mean
     +/- 1 SEM from the saved .npz HV traces.

Regenerates writeup/figures/master_compiled_results.png. Rerun after pulling
new cluster results; it discovers whatever is present on disk.
"""
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_comparison import hv_trace, objective_Y

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.join(HERE, "experiments")
TAU = os.path.join(HERE, "composite_ablation", "results_tau_repo")

# --- Block A: own composite experiments, in narrative order -----------------
OWN = [
    ("moopf_composite", "composite_moopf", "MOOPF 5-objective OPF  (34D, 5 obj)"),
    ("rcm46_composite", "composite_rcm46", "RCM46 Optimal Power Flow  (34D, 4 obj)"),
    ("rcm40_composite", "composite_rcm40", "RCM40 Optimal Power Flow  (34D, 2 obj)"),
    ("gri_mech_composite", "composite_gri_mech", "GRI-Mech 3.0 calibration  (64D, 2 obj)"),
    ("snar_composite", "composite_snar", "Summit SnAr reactor  (4D, 2 obj)"),
    ("penicillin_composite", "composite_penicillin", "Penicillin fermentation  (7D, 3 obj)"),
    ("oc20_composite", "composite_oc20", "OC20 catalyst relaxation  (57D, 2 obj)"),
    ("topopt_composite", "composite_topopt", "2D SIMP topology opt.  (1152D, 2 obj)"),
    ("photonic_composite", "composite_photonic", "Photonic-crystal bandgap  (1024D, 2 obj)"),
]

# --- Block B: tau-repo benchmarks (auto-discovered .npz solver pairs) --------
TAU_ORDER = [
    # This project's own benchmarks, run through tau315's multi-method
    # pipeline (not just our own MORBO engine) -- listed first.
    ("moopf_5obj_34d", "MOOPF-5 OPF, tau pipeline  (34D, 5 obj)"),
    ("rcm40_2obj_34d", "RCM40 OPF, tau pipeline  (34D, 2 obj)"),
    ("rcm46_4obj_34d", "RCM46 OPF, tau pipeline  (34D, 4 obj)"),
    ("penicillin_3obj_7d", "Penicillin, tau pipeline  (7D, 3 obj)"),
    ("dtlz2_2obj_100d", "DTLZ2  (100D, 2 obj)"),
    ("dtlz2_2obj_600d", "DTLZ2  (600D, 2 obj)"),
    ("cort_tg119_3obj_418d", "CORT TG119 radiotherapy  (418D, 3 obj)"),
    ("projected_langermann_2obj_500d", "Projected Langermann  (500D, 2 obj)"),
    ("ackley_griewank_2obj_50d", "Ackley-Griewank  (50D, 2 obj)"),
    ("nanoparticle_rgb_3obj_6d", "Nanoparticle RGB  (6D, 3 obj)"),
    ("summit_snar_2obj_4d", "Summit SnAr (tau port)  (4D, 2 obj)"),
    ("dtlz2_2obj_6d", "DTLZ2  (6D, 2 obj)"),
    ("dtlz2_5obj_6d_ours", "DTLZ2  (6D, 5 obj)"),
    ("five_ackley_5obj_6d", "Five-Ackley  (6D, 5 obj)"),
    ("langermann3_ackley_2obj_6d", "Langermann-Ackley  (6D, 2 obj)"),
    ("ackley_griewank_2obj_6d", "Ackley-Griewank  (6D, 2 obj)"),
]

DIRECT_C = "#c44"
COMP_C = "#2a78d6"
# distinct color per tau solver pair
PAIR_C = {
    "standard_mobo": "#8e44ad",
    "chebyshev_bo": "#e67e22",
    "batched_morbo": "#2a78d6",
    "spherical_chebyshev_bo": "#16a085",
}


def own_panel(ax, exp, comp_label, title):
    exp_dir = os.path.join(EXP, exp)
    with open(os.path.join(exp_dir, "config.json")) as f:
        cfg = json.load(f)
    ref = torch.tensor(cfg["max_reference_point"], dtype=torch.double)
    step = max(cfg.get("batch_size", 10), cfg["max_evals"] // 40)

    def curve(label, color, ls):
        sub = os.path.join(exp_dir, label)
        if not os.path.isdir(sub):
            return None
        seeds = sorted(
            int(m.group(1))
            for f in os.listdir(sub)
            if (m := re.match(rf"(\d+)_{re.escape(label)}\.pt$", f))
        )
        traces, ns_ref = [], None
        for s in seeds:
            out = torch.load(os.path.join(sub, f"{s:04d}_{label}.pt"),
                             map_location="cpu", weights_only=False)
            ns, hvs = hv_trace(objective_Y(out), ref, step=step)
            if ns_ref is None:
                ns_ref = ns
            m = min(len(ns_ref), len(ns))
            ns_ref = ns_ref[:m]
            traces = [t[:m] for t in traces]
            traces.append(hvs[:m])
        A = np.array(traces)
        mean = A.mean(0)
        sem = A.std(0, ddof=1) / np.sqrt(A.shape[0]) if A.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(ns_ref, mean, color=color, ls=ls, lw=1.8,
                label=f"{'composite' if ls=='--' else 'direct'} (n={A.shape[0]})")
        ax.fill_between(ns_ref, mean - sem, mean + sem, color=color, alpha=0.18, lw=0)
        return A[:, -1]

    d = curve("morbo", DIRECT_C, "-")
    c = curve(comp_label, COMP_C, "--")
    ax.axvline(cfg["n_initial_points"], color="gray", ls=":", lw=0.9)
    annotate(ax, d, c)
    finish(ax, title)


def tau_panel(ax, slug, title):
    best = None  # (n_seeds, direct_finals, comp_finals, pair_name) with the most seeds
    for fname in sorted(os.listdir(os.path.join(TAU, slug))):
        if not fname.endswith(".npz"):
            continue
        stem = fname[:-4]
        direct_name = stem.split("_-_")[0]
        color = PAIR_C.get(direct_name, "#555")
        arr = np.load(os.path.join(TAU, slug, fname), allow_pickle=True)
        for key, ls in (("direct_traces", "-"), ("composite_traces", "--")):
            tr = arr[key]
            n = len(tr)
            L = min(len(t) for t in tr)
            A = np.array([t[:L] for t in tr], dtype=float)
            mean = A.mean(0)
            sem = A.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
            xs = np.arange(1, L + 1)
            ax.plot(xs, mean, color=color, ls=ls, lw=1.5)
            ax.fill_between(xs, mean - sem, mean + sem, color=color, alpha=0.13, lw=0)
        # Annotate the pair with the MOST seeds (a just-started STCH pair at
        # n=1 must not shadow a complete n=20 morbo pair's headline delta/p).
        n_this = min(len(arr["direct_traces"]), len(arr["composite_traces"]))
        if best is None or n_this > best[0]:
            d = np.array([t[-1] for t in arr["direct_traces"]], dtype=float)
            c = np.array([t[-1] for t in arr["composite_traces"]], dtype=float)
            best = (n_this, d, c, direct_name)
    if best is not None:
        _, d, c, pair = best
        annotate(ax, d, c, n_hint=f"{pair} n={len(d)}")
    finish(ax, title)


def annotate(ax, direct_finals, comp_finals, n_hint=""):
    if direct_finals is None or comp_finals is None:
        return
    from scipy import stats
    m = min(len(direct_finals), len(comp_finals))
    d, c = direct_finals[:m], comp_finals[:m]
    delta = (c.mean() - d.mean()) / abs(d.mean()) * 100
    try:
        _, p = stats.ttest_rel(c, d)
    except Exception:
        p = float("nan")
    sig = p < 0.05
    col = "#146b3a" if delta >= 0 else "#a13c1c"
    txt = f"Δ {delta:+.2f}%   p={p:.3f}" + ("  *" if sig else "")
    if n_hint:
        txt = f"{n_hint}   " + txt
    ax.text(0.03, 0.96, txt, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, color=col, fontweight="bold" if sig else "normal",
            family="DejaVu Sans Mono")


def finish(ax, title):
    ax.set_title(title, fontsize=9.5, fontweight="bold", pad=6)
    ax.set_xlabel("Function evaluations", fontsize=8)
    ax.set_ylabel("Hypervolume", fontsize=8)
    ax.tick_params(labelsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, color="#eceeec", lw=0.6)
    ax.set_axisbelow(True)


def main():
    tau_present = [(s, t) for s, t in TAU_ORDER if os.path.isdir(os.path.join(TAU, s))]
    ncols = 4
    a_rows = (len(OWN) + ncols - 1) // ncols
    b_rows = (len(tau_present) + ncols - 1) // ncols
    SPACER = a_rows  # index of the empty divider row
    nrows = a_rows + 1 + b_rows  # +1 spacer row between blocks

    fig = plt.figure(figsize=(4.1 * ncols, 3.1 * nrows))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.55, wspace=0.32,
                          height_ratios=[1] * a_rows + [0.32] + [1] * b_rows)

    # Block A
    for i, (exp, comp, title) in enumerate(OWN):
        ax = fig.add_subplot(gs[i // ncols, i % ncols])
        own_panel(ax, exp, comp, title)
        if i == 0:
            ax.legend(fontsize=7, loc="lower right", frameon=False)

    # Spacer row: a full-width invisible axes carrying the B-block header
    spacer = fig.add_subplot(gs[SPACER, :])
    spacer.axis("off")
    spacer.axhline(0.5, color="#c4cac6", lw=1.0)
    spacer.text(0.0, 0.08,
                "B · Collaborator repo tau315/composite-mobo  "
                "(n up to 20 · solid = direct, dashed = composite · color = solver pair)",
                fontsize=11, fontweight="bold", color="#1c2321",
                va="bottom", ha="left", transform=spacer.transAxes)

    # Block B
    for j, (slug, title) in enumerate(tau_present):
        r = SPACER + 1 + j // ncols
        ax = fig.add_subplot(gs[r, j % ncols])
        tau_panel(ax, slug, title)

    fig.subplots_adjust(left=0.05, right=0.985, top=0.93, bottom=0.02)

    # figure title + block-A header, placed in the reserved top band so
    # nothing collides with the first row of panels
    fig.text(0.5, 0.988,
             "Composite vs. direct modeling — hypervolume trajectories across every benchmark",
             ha="center", va="top", fontsize=16, fontweight="bold")
    fig.text(0.012, 0.958,
             "A · Own pipeline  (real MORBO engine · n=5 seeds · mean ±1 SEM · red/solid = direct, blue/dashed = composite)",
             fontsize=11, fontweight="bold", color="#1c2321", va="top")

    out = os.path.join(HERE, "writeup", "figures", "master_compiled_results.png")
    fig.savefig(out, dpi=170, facecolor="white")
    print("saved", out)


if __name__ == "__main__":
    main()
