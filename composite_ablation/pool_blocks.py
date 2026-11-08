r"""Pool several seed blocks of the same benchmark arm into one paired sample.

Each block is a `run_ablation.py --out-dir` directory.  Blocks are disjoint
slices of the SAME seed sequence (`seed = seed0 + 10_007 * trial`), so
concatenating them is exactly what one longer run would have produced.

If a block stored `direct_Y`/`composite_Y` (runs from the patch that added
them), `--ref-point` recomputes hypervolume at any reference point instead of
re-running.  Older blocks hold only the hypervolume traces, computed at
whatever reference the benchmark shipped, and cannot be converted -- pooling
those with a re-referenced block would silently mix two scales, so this
refuses to do it.

Usage:
    python -m composite_ablation.pool_blocks results_tau_repo/dtlz2_2obj_6d \
        results_blocks/dtlz2_2obj_6d_b1 results_blocks/dtlz2_2obj_6d_b2
    python -m composite_ablation.pool_blocks --ref-point 1.1 results_blocks/*
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import ttest_rel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from composite_ablation.run_ablation import dominated_hypervolume_trace  # noqa: E402


def _finals(z, arm, ref):
    """Final hypervolume per trial, recomputed at `ref` when Y is available."""
    if ref is None:
        return [float(t[-1]) for t in z[f"{arm}_traces"].tolist()], False
    key = f"{arm}_Y"
    if key not in z.files:
        return None, False
    out = []
    for Y in z[key].tolist():
        T = torch.as_tensor(np.asarray(Y, dtype=float), dtype=torch.double)
        out.append(float(dominated_hypervolume_trace(T, ref)[-1]))
    return out, True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--ref-point", type=float, nargs="+", default=None,
                    help="recompute hypervolume at this reference point "
                         "(needs blocks that stored raw objective values)")
    a = ap.parse_args()

    ref = None
    if a.ref_point is not None:
        ref = torch.as_tensor(a.ref_point, dtype=torch.double)

    pooled = defaultdict(lambda: {"d": [], "c": [], "src": []})
    for d in a.dirs:
        for f in sorted(os.listdir(d)):
            if not f.endswith(".npz"):
                continue
            z = np.load(os.path.join(d, f), allow_pickle=True)
            if ref is not None and ref.numel() == 1:
                m = np.asarray(z["direct_Y"].tolist()[0]).shape[-1] \
                    if "direct_Y" in z.files else None
                ref_use = ref.expand(m).clone() if m else ref
            else:
                ref_use = ref
            dd, rec = _finals(z, "direct", ref_use)
            cc, _ = _finals(z, "composite", ref_use)
            if dd is None or cc is None:
                print(f"  SKIP {d}/{f}: no raw Y stored, cannot re-reference")
                continue
            n = min(len(dd), len(cc))
            p = pooled[f[:-4]]
            p["d"] += dd[:n]; p["c"] += cc[:n]; p["src"].append((d, n, rec))

    for arm, p in sorted(pooled.items()):
        d, c = np.array(p["d"]), np.array(p["c"])
        n = len(d)
        pct = 100.0 * (c - d) / np.abs(d)
        mu, sd = pct.mean(), pct.std(ddof=1)
        se = sd / math.sqrt(n)
        _, tp = ttest_rel(c, d)
        tp = tp / 2 if c.mean() > d.mean() else 1 - tp / 2
        print(f"\n{arm}   n={n}   blocks: "
              + ", ".join(f"{os.path.basename(s)}({k})" for s, k, _ in p["src"]))
        print(f"  direct  {d.mean():.6f}   composite {c.mean():.6f}")
        print(f"  gain    {mu:+.2f} +- {sd:.2f} (sd)   +- {1.96 * se:.2f} (95% CI)"
              f"   win {int((c > d).sum())}/{n}   p={tp:.4g}")


if __name__ == "__main__":
    main()
