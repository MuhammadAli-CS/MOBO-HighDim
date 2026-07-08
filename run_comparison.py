#!/usr/bin/env python3
r"""Run a MORBO vs. scalarized-TuRBO comparison replication.

This mirrors the paper's Figure 2 comparison (Daulton et al., UAI 2022):
a straightforward multi-objective extension of TuRBO where each trust
region independently optimizes its own random (augmented) Chebyshev
scalarization, versus full MORBO with coordinated hypervolume-based
selection and data sharing.

The "turbo_scalarized" label reuses MORBO's own trust-region machinery
with the coordination turned off:
  - hypervolume=False        -> each TR optimizes a random Chebyshev
                                scalarization (kept for the TR's lifetime;
                                a new one is drawn on restart), success /
                                failure counted on the TR's own objective.
  - track_history=False      -> no data sharing across trust regions.
  - restart_hv_scalarizations=False -> no coordinated global restart search.

The "composite_morbo" label runs ordinary (coordinated, hypervolume-based)
MORBO, but swaps `evalfn` to "CompositeDTLZ2": the local GPs model DTLZ2's
raw intermediate response instead of the final 2 objectives directly, with
the known reduction applied before anything Pareto/HV-shaped sees it. Since
"CompositeDTLZ2" is mathematically identical to plain "DTLZ2" (same true
objectives, same Pareto front, same reference point), it's a controlled A/B
against the "morbo" label on the composite-modeling question specifically --
the "morbo" label is untouched and can be reused from a prior run in the
same experiment directory without rerunning it.

The "llm_morbo" label runs ordinary MORBO with an added candidate source: an
LLM proposes a handful of candidate points per trust region once per BO
iteration (not per batch slot), concatenated into the existing
Thompson-sampling candidate pool and screened by the same HVI/scalarization
scoring as every other candidate before acceptance. The number of candidates
requested decays as each trust region accumulates its own local data (reuses
`decay_function`, see `morbo/gen.py`). Requires `ANTHROPIC_API_KEY`.

The "independent_gp_composite" / "kronecker_gp_composite" labels both run
composite MORBO on "CompositeDTLZ2Curve" (a genuinely *correlated* raw
response -- a discretized curve whose endpoints are exactly DTLZ2's own 2
objectives, see `morbo/problems/composite_dtlz2_curve.py`), differing only
in whether the local raw-response GP is `n_curve_points` independent
single-task GPs (`get_fitted_model`, same as `composite_morbo`) or one
Kronecker-structured multi-task GP that models correlation across the raw
dimensions (`get_fitted_kronecker_model`). This isolates whether
correlation-aware modeling (the actual contribution of Maddox, Feng &
Balandat 2021's HOGP-based composite MORBO) matters, or whether simple
per-dimension decoupling is enough.

The "composite_penicillin" label runs composite MORBO on "CompositePenicillin"
-- Penicillin's raw ~2500-step fermentation trajectory, checkpointed at 10
points (5 state variables each, see `morbo/problems/composite_penicillin.py`),
reduced to the same 3 objectives (yield, CO2, time) plain Penicillin exposes.
Compare against the "morbo" label on the same experiment directory
(`evalfn: "Penicillin"` in the base config) for a controlled A/B.

Usage (same CLI shape as experiments/main.py):
    python run_comparison.py <experiment_name> <label> <seed>
e.g.
    python run_comparison.py fig2_dtlz2_100d morbo 0
    python run_comparison.py fig2_dtlz2_100d turbo_scalarized 0
    python run_comparison.py fig2_dtlz2_100d composite_morbo 0
    python run_comparison.py llm_morbo_vehicle_safety morbo 0
    python run_comparison.py llm_morbo_vehicle_safety llm_morbo 0
    python run_comparison.py correlation_ablation_dtlz2curve morbo 0
    python run_comparison.py correlation_ablation_dtlz2curve independent_gp_composite 0
    python run_comparison.py correlation_ablation_dtlz2curve kronecker_gp_composite 0
    python run_comparison.py penicillin_composite morbo 0
    python run_comparison.py penicillin_composite composite_penicillin 0
"""
import json
import os
import sys

import torch
from morbo.run_one_replication import run_one_replication

LABEL_OVERRIDES = {
    "morbo": {},
    "turbo_scalarized": {
        "hypervolume": False,
        "scalarization_type": "chebyshev",
        "track_history": False,
        "restart_hv_scalarizations": False,
    },
    "composite_morbo": {
        "evalfn": "CompositeDTLZ2",
    },
    "llm_morbo": {
        "use_llm_candidates": True,
        "llm_candidates_per_tr": 4,
        "llm_problem_description": (
            "Vehicle frontal-crash safety design: 5 design variables control the "
            "widths of structural components of the vehicle's frame. The 3 "
            "objectives (all maximized, i.e. more negative raw mass/intrusion/"
            "acceleration is worse) are: vehicle mass (lower is better, correlated "
            "with fuel economy), toe-board intrusion (lower is better, less "
            "passenger-compartment damage), and full-frontal-collision "
            "acceleration (lower is better, less passenger injury)."
        ),
    },
    "independent_gp_composite": {
        "evalfn": "CompositeDTLZ2Curve",
        "use_kronecker_gp": False,
    },
    "kronecker_gp_composite": {
        "evalfn": "CompositeDTLZ2Curve",
        "use_kronecker_gp": True,
    },
    "composite_penicillin": {
        "evalfn": "CompositePenicillin",
    },
}

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, "experiments", sys.argv[1])
    config_path = os.path.join(exp_dir, "config.json")
    label = sys.argv[2]
    seed = int(sys.argv[3])
    if label not in LABEL_OVERRIDES:
        raise ValueError(f"label must be one of {list(LABEL_OVERRIDES)}")

    with open(config_path, "r") as f:
        kwargs = json.load(f)
    kwargs.update(LABEL_OVERRIDES[label])

    output_dir = os.path.join(exp_dir, label)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{str(seed).zfill(4)}_{label}.pt")
    save_callback = lambda data: torch.save(data, output_path)

    run_one_replication(
        seed=seed,
        label=label,
        save_callback=save_callback,
        **kwargs,
    )
    print(f"Saved output to {output_path}")
