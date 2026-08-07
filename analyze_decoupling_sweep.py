#!/usr/bin/env python3
r"""Aggregate the controlled decoupling sweep and make the causal figure.

Reads experiments/decoupling_b{N}/{dtlz2_blocked,composite_dtlz2_blocked}/*.pt,
computes the composite-vs-direct final-HV advantage per block size, and plots
advantage against the effective input-dimension (block_size / dim). A
monotone-decreasing curve is the causal claim: with input-dimension, objective
count, and the exact objective all held fixed, only the raw-response coupling
changes -- so any trend is attributable to decoupling alone.

Usage: python analyze_decoupling_sweep.py
"""
import glob
import json
import os

import numpy as np
import torch
from scipy import stats


def finals(exp, label):
    out = []
    for f in sorted(glob.glob(f"experiments/{exp}/{label}/*_{label}.pt")):
        d = torch.load(f, map_location="cpu", weights_only=False)
        out.append(np.array(d["true_hv"])[-1])
    return np.array(out)


rows = []
for d in sorted(glob.glob("experiments/decoupling_b*/config.json")):
    exp = os.path.basename(os.path.dirname(d))
    cfg = json.load(open(d))
    b, dim = cfg["block_size"], cfg["dim"]
    comp, direct = finals(exp, "composite_dtlz2_blocked"), finals(exp, "dtlz2_blocked")
    if len(comp) == 0 or len(direct) == 0:
        print(f"{exp}: incomplete (composite={len(comp)}, direct={len(direct)}) -- skipped")
        continue
    n = min(len(comp), len(direct))
    delta = (comp[:n].mean() - direct[:n].mean()) / abs(direct[:n].mean()) * 100
    try:
        _, p = stats.ttest_rel(comp[:n], direct[:n])
    except Exception:
        p = float("nan")
    rows.append({"block_size": b, "eff_dim": b / dim, "advantage_pct": float(delta),
                 "p": float(p), "wins": int((comp[:n] > direct[:n]).sum()), "n": n})

rows.sort(key=lambda r: r["eff_dim"])
print(f"{'block':>5} {'eff_dim':>8} {'advantage%':>11} {'win':>6} {'p':>7}")
for r in rows:
    print(f"{r['block_size']:>5} {r['eff_dim']:>8.3f} {r['advantage_pct']:>+11.2f} "
          f"{r['wins']}/{r['n']:<4} {r['p']:>7.3f}")

if len(rows) >= 3:
    E = np.array([r["eff_dim"] for r in rows]); A = np.array([r["advantage_pct"] for r in rows])
    rho, pr = stats.spearmanr(E, A)
    print(f"\nSpearman(advantage, eff_dim) = {rho:+.3f}  p={pr:.4f}  (causal claim: strong NEGATIVE)")
    json.dump(rows, open("experiments/decoupling_sweep_summary.json", "w"), indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ax.axhline(0, color="#999", lw=1, ls=":")
        ax.plot(E, A, "o-", color="#2a78d6", lw=2.2, ms=9, zorder=3)
        for r in rows:
            ax.annotate(f"b={r['block_size']}", (r["eff_dim"], r["advantage_pct"]),
                        textcoords="offset points", xytext=(7, 6), fontsize=8.5, color="#555")
        ax.set_xlabel("Effective input-dimension of raw components  (block size / dim)", fontsize=11)
        ax.set_ylabel("Composite advantage:  final-HV Δ vs. direct  (%)", fontsize=11)
        ax.set_title("Controlled decoupling sweep — DTLZ2 objective held FIXED\n"
                     f"only raw-response coupling varies  (Spearman ρ={rho:+.2f}, p={pr:.3f})",
                     fontsize=12, fontweight="bold")
        ax.grid(True, color="#eee")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        fig.tight_layout()
        os.makedirs("writeup/figures", exist_ok=True)
        fig.savefig("writeup/figures/decoupling_sweep.png", dpi=170, facecolor="white")
        print("saved writeup/figures/decoupling_sweep.png")
    except Exception as e:
        print("plot skipped:", e)
