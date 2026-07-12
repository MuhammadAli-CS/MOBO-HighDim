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

The "ard_box" / "pca_ellipsoid" / "ard_pca_ellipsoid" labels run ordinary
MORBO (direct DTLZ2 objectives, same as the "morbo" label) but replace the
trust region's isotropic hypercube with an alternative shape, adapted to
each TR's own local data every time its model refits (see `tr_shape` on
`TurboHParams`, `morbo/trust_region.py`): "ard_box" rescales the box
per-dimension by the TR's fitted GP ARD lengthscales (the original TuRBO
paper's technique, which this fork's isotropic-only default doesn't
implement); "pca_ellipsoid" instead rotates the box into the PCA frame of
the TR's local data (no lengthscale involvement); "ard_pca_ellipsoid"
combines both -- the PCA rotation, with per-axis widths additionally
reweighted by lengthscales projected onto each principal axis. All three
are controlled A/Bs against the "morbo" label (identical `evalfn`, only TR
geometry differs) and can reuse an existing "morbo" result from the same
experiment directory without rerunning it. Not supported together with
`use_kronecker_gp`.

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
    python run_comparison.py tr_shape_dtlz2_100d ard_box 0
    python run_comparison.py tr_shape_dtlz2_100d pca_ellipsoid 0
    python run_comparison.py tr_shape_dtlz2_100d ard_pca_ellipsoid 0
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
    "ard_box": {
        "tr_shape": "ard_box",
    },
    "pca_ellipsoid": {
        "tr_shape": "pca_ellipsoid",
    },
    "ard_pca_ellipsoid": {
        "tr_shape": "ard_pca_ellipsoid",
    },
    # Composite modeling x shape adaptation: composes with zero new code.
    # tr_shape only changes candidate sampling/containment in design space;
    # composite modeling only changes what the GP models (the raw response)
    # and how objectives are reconstructed. extract_ard_lengthscale already
    # geometric-means across however many outputs the local ModelListGP has
    # (K raw dims for composite, M objectives for direct), so the
    # lengthscale-based variants work unchanged too.
    "composite_penicillin_ard_pca": {
        "evalfn": "CompositePenicillin",
        "tr_shape": "ard_pca_ellipsoid",
    },
    "composite_penicillin_pca": {
        "evalfn": "CompositePenicillin",
        "tr_shape": "pca_ellipsoid",
    },
    # CMA-ES-style covariance adaptation (AS-SMEA, Wang et al. 2026):
    # persistent per-TR covariance updated from Pareto-elite points plus an
    # evolution-path term -- success-weighted and temporally smoothed, vs.
    # the one-shot data-covariance PCA variants above.
    "cma_ellipsoid": {
        "tr_shape": "cma_ellipsoid",
    },
    # Spherically-projected linear kernel (Doumont et al. 2026, "We Still
    # Don't Understand High-Dimensional BO"): the challenge baseline -- in
    # the N ~ d regime a cosine-similarity linear kernel reportedly matches
    # TuRBO-class methods. Run alone and crossed with the shape variants
    # (shape adaptation is model-agnostic: it only consumes the model via
    # posterior sampling and, for ARD variants, lengthscales -- which a
    # linear kernel doesn't have, hence no linear_gp_ard_* labels).
    "linear_gp": {
        "use_linear_kernel": True,
    },
    "linear_gp_pca": {
        "use_linear_kernel": True,
        "tr_shape": "pca_ellipsoid",
    },
    "linear_gp_cma": {
        "use_linear_kernel": True,
        "tr_shape": "cma_ellipsoid",
    },
    # Dimension-scaled lengthscale prior (Hvarfner et al. 2024): does the
    # sqrt(d)-scaled LogNormal prior -- which removes the Interval(0.05, 4.0)
    # ceiling that ~99/100 fitted lengthscales pin against at d=100 --
    # rescue ard_box from its constraint-compounding region collapse?
    "ard_box_dimprior": {
        "tr_shape": "ard_box",
        "use_dim_scaled_ls_prior": True,
    },
    "ard_pca_dimprior": {
        "tr_shape": "ard_pca_ellipsoid",
        "use_dim_scaled_ls_prior": True,
    },
}

if __name__ == "__main__":
    experiment_name = sys.argv[1]
    # Thermal throttle: penicillin_composite's morbo/composite_penicillin runs
    # were running the CPU hot (~95C) via torch/MKL saturating most threads.
    # Cap threads for just this experiment rather than everywhere, since the
    # fig2/correlation-ablation runs already ran fine at full thread count.
    if experiment_name == "penicillin_composite":
        torch.set_num_threads(8)
        torch.set_num_interop_threads(4)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    exp_dir = os.path.join(current_dir, "experiments", experiment_name)
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
