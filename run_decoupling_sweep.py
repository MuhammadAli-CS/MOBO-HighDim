#!/usr/bin/env python3
r"""Controlled decoupling experiment driver.

Sweeps the block-size knob of morbo/problems/composite_dtlz2_blocked.py --
holding the input dimension, objective count, and exact objective function
fixed -- and measures the composite-vs-direct hypervolume advantage at each
block size. If the advantage falls monotonically as block size (effective
input-dimension) rises, decoupling *causes* the composite advantage rather
than merely correlating with it across heterogeneous benchmarks.

Usage:
    python run_decoupling_sweep.py [--dim 25] [--seeds 3] [--evals 150]
Writes experiments/decoupling_sweep/results.json and (if matplotlib is
available) experiments/decoupling_sweep/advantage_vs_effdim.png.
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from morbo.run_one_replication import run_one_replication
from morbo.problems.composite_dtlz2_blocked import (
    dtlz2_blocked_objective,
    get_composite_dtlz2_blocked_fn,
    composite_dtlz2_blocked_reduction,
)

REF = [-1.1, -1.1]  # maximize convention; DTLZ2 front on the unit quarter-circle


def _run(dim, block_size, seed, evals, n_init, batch, composite):
    captured = {}
    if composite:
        raw_fn, _ = get_composite_dtlz2_blocked_fn(dim, block_size)
        kwargs = dict(
            raw_evaluate_components=raw_fn,
            raw_compose=lambda Y: -composite_dtlz2_blocked_reduction(Y),  # minimize convention
        )
    else:
        kwargs = dict(raw_evaluate=lambda X: -dtlz2_blocked_objective(X, dim))  # minimize
    run_one_replication(
        seed=seed, label="morbo", max_evals=evals, evalfn="Callable",
        batch_size=batch, dim=dim, n_initial_points=n_init,
        n_trust_regions=3, min_tr_size=n_init, max_reference_point=REF,
        save_during_opt=False,
        save_callback=lambda out: captured.update(out), **kwargs,
    )
    return np.array(captured["true_hv"])[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--evals", type=int, default=150)
    args = ap.parse_args()
    dim, k = args.dim, args.dim - 1
    n_init, batch = max(2 * dim, 30), 20
    # block sizes: divisors-ish spanning 1..k
    bset = sorted({1, 2, 4, 8, max(1, k // 2), k})
    bset = [b for b in bset if b <= k]

    print(f"dim={dim} k={k} block_sizes={bset} seeds={args.seeds} evals={args.evals}")
    # direct baseline (identical objective for every block size): run once per seed
    direct = np.array([_run(dim, k, s, args.evals, n_init, batch, composite=False) for s in range(args.seeds)])
    print(f"direct baseline final HV: mean={direct.mean():.4f}")

    rows = []
    for b in bset:
        comp = np.array([_run(dim, b, s, args.evals, n_init, batch, composite=True) for s in range(args.seeds)])
        delta = (comp.mean() - direct.mean()) / abs(direct.mean()) * 100
        eff = b / dim
        rows.append({"block_size": b, "eff_dim": eff, "advantage_pct": delta,
                     "composite_hv": comp.mean(), "wins": int((comp > direct).sum()), "n": args.seeds})
        print(f"  b={b:3d}  eff_dim={eff:.3f}  advantage={delta:+.2f}%  ({int((comp>direct).sum())}/{args.seeds})")

    os.makedirs("experiments/decoupling_sweep", exist_ok=True)
    out = {"dim": dim, "seeds": args.seeds, "evals": args.evals,
           "direct_hv": float(direct.mean()), "rows": rows}
    with open("experiments/decoupling_sweep/results.json", "w") as f:
        json.dump(out, f, indent=2)

    E = np.array([r["eff_dim"] for r in rows]); A = np.array([r["advantage_pct"] for r in rows])
    from scipy import stats
    if len(rows) >= 3:
        rho, p = stats.spearmanr(E, A)
        print(f"\nSpearman(advantage, eff_dim) = {rho:+.3f}  p={p:.4f}  (expect strong NEGATIVE)")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(E, A, "o-", color="#2a78d6", lw=2, ms=8)
        ax.axhline(0, color="#888", lw=1, ls=":")
        ax.set_xlabel("Effective input-dimension of raw components  (block size / dim)")
        ax.set_ylabel("Composite advantage: final-HV Δ vs direct (%)")
        ax.set_title(f"Controlled decoupling sweep — DTLZ2, dim={dim} & objective FIXED\n"
                     f"only raw-response granularity varies (n={args.seeds} seeds)")
        for r in rows:
            ax.annotate(f"b={r['block_size']}", (r["eff_dim"], r["advantage_pct"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.grid(True, color="#eee")
        fig.tight_layout()
        fig.savefig("experiments/decoupling_sweep/advantage_vs_effdim.png", dpi=150)
        print("saved experiments/decoupling_sweep/advantage_vs_effdim.png")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
