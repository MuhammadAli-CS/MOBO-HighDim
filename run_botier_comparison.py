#!/usr/bin/env python3
r"""Compare LLM-proposed vs. hand-specified BoTier tiers on DTLZ5 (M=2).

Note: `experiments/dtlz5_m2/config.json` in this repo is mislabeled (its
`evalfn` is actually "DTLZ7", matching a swap with `experiments/dtlz7_m2/`,
which is itself mislabeled "DTLZ5" -- a pre-existing bug in the archived
upstream repo, not touched here). This script constructs the DTLZ5 problem
directly instead of loading that config.

Usage:
    python run_botier_comparison.py [dim] [max_evals] [seed]
e.g.
    python run_botier_comparison.py 10 60 0
"""
import json
import os
import sys

import torch
from botorch.test_functions.multi_objective import DTLZ5

from botier_llm.llm_tiers import propose_tiers
from botier_llm.solver import percentile_thresholds, run_botier_bo

OBJECTIVE_NAMES = ["objective_1", "objective_2"]
PROBLEM_DESCRIPTION = (
    "DTLZ5 is a standard 2-objective synthetic benchmark with a curved "
    "(degenerate, roughly 1-dimensional) Pareto front. Both objectives are "
    "maximized (higher is better) and range roughly between 0 and 1.1 at the "
    "Pareto front, trading off against each other."
)


def main():
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    max_evals = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    n_initial_points = max(2 * dim, 10)

    problem = DTLZ5(dim=dim, num_objectives=2, negate=True)  # maximize convention
    bounds = problem.bounds.to(torch.double)

    def eval_fn(X):
        return problem(X)

    torch.manual_seed(seed)
    X_warm = torch.rand(n_initial_points, dim, dtype=torch.double)
    Y_warm = eval_fn(X_warm)

    order_hand = torch.tensor([0, 1], dtype=torch.long)
    thresholds_hand = percentile_thresholds(Y_warm, order_hand, percentile=0.5)
    print(f"Hand-specified (median warm-start) thresholds: {thresholds_hand.tolist()}")

    order_llm, thresholds_llm, rationale = propose_tiers(
        problem_description=PROBLEM_DESCRIPTION,
        objective_names=OBJECTIVE_NAMES,
        warm_start_Y=Y_warm,
    )
    print(f"LLM-proposed order: {[OBJECTIVE_NAMES[i] for i in order_llm.tolist()]}")
    print(f"LLM-proposed thresholds: {thresholds_llm.tolist()}")
    print(f"LLM rationale: {rationale}")

    results = {}
    for name, order, thresholds in [
        ("hand_specified", order_hand, thresholds_hand),
        ("llm_proposed", order_llm, thresholds_llm),
    ]:
        print(f"\n=== Running BoTier BO with {name} tiers ===")
        result = run_botier_bo(
            eval_fn=eval_fn,
            bounds=bounds,
            thresholds=thresholds,
            order=order,
            n_initial_points=n_initial_points,
            max_evals=max_evals,
            batch_size=5,
            seed=seed,
            verbose=True,
        )
        cleared = (result["best_Y"] >= thresholds[order.argsort()]).tolist()
        results[name] = {
            "order": order.tolist(),
            "thresholds": thresholds.tolist(),
            "best_Y": result["best_Y"].tolist(),
            "best_xi": result["xi"][result["best_idx"]].item(),
            "thresholds_cleared": cleared,
        }
        print(f"{name}: best_Y={results[name]['best_Y']}, "
              f"best_xi={results[name]['best_xi']:.4f}, cleared={cleared}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(current_dir, "experiments", "botier_dtlz5")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{str(seed).zfill(4)}_comparison.json")
    with open(out_path, "w") as f:
        json.dump(
            {"dim": dim, "max_evals": max_evals, "seed": seed, **results,
             "llm_rationale": rationale},
            f,
            indent=2,
        )
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
