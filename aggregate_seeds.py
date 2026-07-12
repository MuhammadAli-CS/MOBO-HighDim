#!/usr/bin/env python3
r"""Aggregate multi-seed results for an experiment: mean +/- std of final
true hypervolume per label, plus per-seed values so outliers are visible.

Everything reported so far in this project is seed 0 only; the multi-seed
cluster sweep (cluster/submit_tr_shape_multiseed.sh) fills in seeds 1-4.
This script makes the aggregate table once those land.

Usage:
    python aggregate_seeds.py <experiment_name> [--labels a b c]
e.g.
    python aggregate_seeds.py tr_shape_dtlz2_100d
"""
import argparse
import os
import re

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_name")
    parser.add_argument("--labels", nargs="+", default=None)
    args = parser.parse_args()

    exp_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "experiments", args.experiment_name
    )
    labels = args.labels
    if labels is None:
        labels = sorted(
            d
            for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d)) and d != "__pycache__"
        )

    print(f"=== {args.experiment_name} ===")
    for label in labels:
        label_dir = os.path.join(exp_dir, label)
        if not os.path.isdir(label_dir):
            continue
        finals = {}
        for fname in sorted(os.listdir(label_dir)):
            m = re.match(rf"(\d+)_{re.escape(label)}\.pt$", fname)
            if not m:
                continue
            seed = int(m.group(1))
            d = torch.load(
                os.path.join(label_dir, fname), map_location="cpu", weights_only=False
            )
            finals[seed] = float(d["true_hv"][-1])
        if not finals:
            print(f"{label:28s}: no results")
            continue
        vals = torch.tensor(list(finals.values()), dtype=torch.double)
        mean = vals.mean().item()
        std = vals.std().item() if len(vals) > 1 else float("nan")
        per_seed = "  ".join(f"s{s}={v:.3f}" for s, v in sorted(finals.items()))
        print(
            f"{label:28s}: mean={mean:10.3f}  std={std:8.3f}  n={len(vals)}   [{per_seed}]"
        )


if __name__ == "__main__":
    main()
