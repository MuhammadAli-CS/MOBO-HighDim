#!/usr/bin/env python3
r"""Paired significance testing for headline tr_shape claims.

Every headline percentage in RESULTS.md's Executive Summary (+66.6%,
+75.7%, +78.9%, ...) was, until now, reported as a paired mean delta plus
a win-rate (e.g. "5/5", "20/20") -- no formal significance test or
confidence interval. That's thin evidence for a reviewer once effect
sizes get large: "why is it THIS big" deserves a p-value and a CI, not
just a mean and a win count.

For a given (experiment, label) vs. a baseline (experiment, label) run at
the same seeds, this computes, on the PAIRED per-seed final-hypervolume
deltas (same seed = same problem instance/initial data, so pairing removes
seed-to-seed variance rather than comparing independent samples):

  - Wilcoxon signed-rank test (two-sided): the primary test. Distribution-
    free, doesn't assume the paired differences are normal -- appropriate
    here since n is often small (5) and HV deltas are not obviously
    Gaussian. Exact p-value used for n <= 25 (scipy's default switches to
    a normal approximation above that, which is what we want anyway -- the
    20-seed program lands right at that boundary).
  - Paired t-test (two-sided): reported alongside as the more familiar/
    conventional check; the two rarely disagree here but showing both
    forecloses "well the test you picked was cherry-picked."
  - A t-based 95% CI on the mean paired percent delta.

Usage:
    python compute_significance.py <exp> <label> [--baseline-exp EXP]
        [--baseline-label LABEL] [--seeds N [N ...]]
e.g.
    python compute_significance.py tr_shape_dtlz2_100d pca_ellipsoid
    python compute_significance.py tr_shape_methods_dtlz2_100d \
        mab_shape_ducb_shared --baseline-exp tr_shape_dtlz2_100d
"""
import argparse
import os
import re

import numpy as np
import torch
from scipy import stats


def _seeds_available(exp_dir: str, label: str):
    label_dir = os.path.join(exp_dir, label)
    if not os.path.isdir(label_dir):
        return []
    seeds = []
    for fname in os.listdir(label_dir):
        m = re.match(rf"(\d+)_{re.escape(label)}\.pt$", fname)
        if m:
            seeds.append(int(m.group(1)))
    return sorted(seeds)


def _final_hv(exp_dir: str, label: str, seed: int) -> float:
    p = os.path.join(exp_dir, label, f"{seed:04d}_{label}.pt")
    d = torch.load(p, map_location="cpu", weights_only=False)
    return float(d["true_hv"][-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("label")
    parser.add_argument("--baseline-exp", default=None)
    parser.add_argument("--baseline-label", default="morbo")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(root, "experiments", args.experiment_name)
    base_exp_dir = os.path.join(
        root, "experiments", args.baseline_exp or args.experiment_name
    )

    avail = set(_seeds_available(exp_dir, args.label))
    base_avail = set(_seeds_available(base_exp_dir, args.baseline_label))
    seeds = sorted(avail & base_avail)
    if args.seeds is not None:
        seeds = sorted(set(seeds) & set(args.seeds))
    if len(seeds) < 2:
        raise SystemExit(
            f"Need >=2 paired seeds, found {len(seeds)} "
            f"(label seeds={sorted(avail)}, baseline seeds={sorted(base_avail)})"
        )

    vals = np.array([_final_hv(exp_dir, args.label, s) for s in seeds])
    base_vals = np.array(
        [_final_hv(base_exp_dir, args.baseline_label, s) for s in seeds]
    )
    raw_delta = vals - base_vals
    pct_delta = raw_delta / base_vals * 100.0

    wilcoxon = stats.wilcoxon(raw_delta, alternative="two-sided")
    ttest = stats.ttest_rel(vals, base_vals)

    n = len(seeds)
    mean_pct = pct_delta.mean()
    sem_pct = pct_delta.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci_lo, ci_hi = mean_pct - tcrit * sem_pct, mean_pct + tcrit * sem_pct

    wins = int((raw_delta > 0).sum())

    print(f"=== {args.experiment_name}/{args.label} vs "
          f"{args.baseline_exp or args.experiment_name}/{args.baseline_label} "
          f"(n={n}, seeds={seeds}) ===")
    print(f"paired mean delta: {mean_pct:+.1f}%  "
          f"95% CI [{ci_lo:+.1f}%, {ci_hi:+.1f}%]  win-rate {wins}/{n}")
    print(f"Wilcoxon signed-rank: statistic={wilcoxon.statistic:.3f}  "
          f"p={wilcoxon.pvalue:.4g}")
    print(f"Paired t-test:        t={ttest.statistic:.3f}  "
          f"p={ttest.pvalue:.4g}  df={n - 1}")


if __name__ == "__main__":
    main()
