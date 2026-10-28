"""Pool the per-job two-objective results into one table.

cluster/submit_two_stage.sh writes one file per (g1 map, inner function draw).
This pools them and recomputes the paired statistics over all (draw, seed)
pairs.  Pairing stays valid because both arms within a file saw the same inner
functions and the same initial designs.

    python composite_ablation/collect_two_stage.py \
        --out composite_ablation/results/two_stage_grid
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _betacf(a, b, x, it=300):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, it):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 / ((1.0 + aa * d) if abs(1.0 + aa * d) > 1e-300 else 1e-300)
        c = 1.0 + aa / (c if abs(c) > 1e-300 else 1e-300)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 / ((1.0 + aa * d) if abs(1.0 + aa * d) > 1e-300 else 1e-300)
        c = 1.0 + aa / (c if abs(c) > 1e-300 else 1e-300)
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-16:
            break
    return h


def _ibeta(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def one_sided_p(diffs):
    """H1: mean > 0, paired t-test."""
    n = len(diffs)
    if n < 2:
        return float("nan")
    m, s = st.mean(diffs), st.stdev(diffs)
    if s == 0:
        return 0.5
    t = m / (s / math.sqrt(n))
    two = _ibeta((n - 1) / 2, 0.5, (n - 1) / ((n - 1) + t * t))
    return two / 2 if t > 0 else 1 - two / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="composite_ablation/results/two_stage_grid")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.out, "*.json")))
    if not files:
        ap.error(f"no results in {a.out}")

    pooled, draws, inj, cfgs = {}, {}, {}, []
    for fp in files:
        b = json.load(open(fp))
        cfgs.append((b["config"].get("kernel"), b["config"].get("g2"),
                     b["config"].get("n_iter"), b["config"].get("dim")))
        for name, r in b["results"].items():
            acc = pooled.setdefault(name, {"direct": [], "composite": []})
            acc["direct"] += r["direct"]
            acc["composite"] += r["composite"]
            inj[name] = r["injective"]
            draws[name] = draws.get(name, 0) + 1

    if len(set(cfgs)) > 1:
        print("WARNING: files disagree on kernel/g2/n_iter/dim:")
        for c in sorted(set(cfgs)):
            print("   ", c)
    k, g2, n_iter, dim = cfgs[0]
    print(f"{len(files)} files, g2={g2}, kernel={k}, dim={dim}, n_iter={n_iter}\n")

    print(f"{'g1':10s} {'bijective':>10s} {'HV direct':>11s} {'HV composite':>13s} "
          f"{'gain %':>16s} {'p':>8s} {'wins':>8s} {'runs':>5s} {'drw':>4s}")
    print("-" * 94)
    for name in sorted(pooled, key=lambda n: -st.mean(
            [100.0 * (c - d) / abs(d) for d, c in
             zip(pooled[n]["direct"], pooled[n]["composite"])])):
        d, c = pooled[name]["direct"], pooled[name]["composite"]
        pct = [100.0 * (cv - dv) / abs(dv) for dv, cv in zip(d, c)]
        m = st.mean(pct)
        se = st.stdev(pct) / math.sqrt(len(pct)) if len(pct) > 1 else float("nan")
        p = one_sided_p(pct)
        wins = sum(1 for dv, cv in zip(d, c) if cv > dv)
        ps = "<0.001" if p < 0.001 else f"{p:.3f}"
        print(f"{name:10s} {str(inj[name]):>10s} {st.mean(d):11.4f} "
              f"{st.mean(c):13.4f} {m:+9.2f} +/-{se:5.2f} {ps:>8s} "
              f"{wins:4d}/{len(d):<3d} {len(d):5d} {draws[name]:4d}")
    print("\ngain is the per-run percentage difference in final hypervolume,")
    print("paired over the shared initial design; p is one-sided for gain > 0.")


if __name__ == "__main__":
    main()
