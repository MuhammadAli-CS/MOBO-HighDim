[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces

This is the code associated with the paper "[Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces](https://arxiv.org/abs/2109.10964)."

Please cite our work if you find it useful.

    @InProceedings{pmlr-v162-daulton22a,
    title = 	 {Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces},
    author =       {Daulton, Samuel and Eriksson, David and Balandat, Maximilian and Bakshy, Eytan},
    booktitle = 	 {Proceedings of the Thirty-Eighth Conference on Uncertainty in Artificial Intelligence},
    year = 	 {2022},
    series = 	 {Proceedings of Machine Learning Research},
    publisher =    {PMLR},
    }


## Getting started

From the base `morbo` directory run:

`pip install -e .`

## Structure

The code is structured in three parts.
- The utilities for constructing the acquisition functions and other helper methods are defined in `morbo/`.
- The experiments are found in and ran from within `experiments/`. The `main.py` is used to run the experiments, and the experiment configurations are found in the `config.json` file of each sub-directory.

The individual experiment outputs were left out to avoid inflating the file size.

## Running Experiments

To run a basic benchmark based on the `config.json` file in `experiments/<experiment_name>` using `<algorithm>`:

```
cd experiments
python main.py <experiment_name> <algorithm> <seed>
```

The code refers to the algorithms using the following labels:
```
algorithms = [
    ("morbo", "MORBO"),
]
```

Each folder under `experiments/` corresponds to the experiments in the paper according to the following mapping:
```
experiments = {
    "dtlz2_10d": "DTLZ2 (d=10)",
    "dtlz2_30d": "DTLZ2 (d=30)",
    "dtlz2_100d": "DTLZ2 (d=100)",
    "dtlz3_m2": "DTLZ3 (M=2)",
    "dtlz5_m2": "DTLZ5 (M=2)",
    "dtlz7_m2": "DTLZ7 (M=2)",
    "dtlz3_m4": "DTLZ3 (M=4)",
    "dtlz5_m4": "DTLZ5 (M=4)",
    "dtlz7_m4": "DTLZ7 (M=4)",
    "rover": "Rover",
    "vehicle_safety": "Vehicle Safety",
    "welded_beam": "Welded Beam",
}
```
Note: this code can heavily exploit a GPU if available.

## Fork notes: modern-botorch port + scalarized-TuRBO baseline

The upstream repo was archived in Oct 2023 and targeted botorch ~0.6. This fork
runs on current versions (tested: torch 2.12, botorch 0.9.5, gpytorch 1.11,
Python 3.11). Port changes: removed/renamed botorch imports
(`AcquisitionObjective`, `fit_gpytorch_model`, `fit_gpytorch_torch`, sampler
`num_samples` API), a `DeterministicSampler` for the dummy qEHVI sampler, and a
fix for a latent crash in `ScalarizedTrustRegion._has_improved_objective`.

We also added the paper's Figure 2 baseline — the naive multi-objective
extension of TuRBO where each independent trust region optimizes its own random
augmented Chebyshev scalarization. This reuses MORBO's own machinery with the
coordination turned off (`hypervolume=False`, `track_history=False`,
`restart_hv_scalarizations=False`, plus a new `scalarization_type="chebyshev"`
hparam; the released code only implemented linear scalarizations).

To reproduce the Figure 2 comparison (DTLZ2, d=100, 600 evals, batch 50, 3 TRs):

```
python run_comparison.py fig2_dtlz2_100d morbo 0
python run_comparison.py fig2_dtlz2_100d turbo_scalarized 0
python plot_comparison.py fig2_dtlz2_100d 0
```

The plot script computes hypervolume post-hoc from `metric_history` with the
same reference point for both methods (scalarized runs don't track it online)
and writes `experiments/fig2_dtlz2_100d/comparison_seed0.png`. `plot_comparison.py`
auto-discovers every label with a saved result for the given experiment/seed
(pass `--labels a b c` to override) — running a new method never requires
rerunning or touching previously-saved results from other labels.

**Known pre-existing bug (not fixed here, upstream archived repo)**:
`experiments/dtlz5_m2/config.json` actually sets `"evalfn": "DTLZ7"`, and
`experiments/dtlz7_m2/config.json` sets `"evalfn": "DTLZ5"` — a swap between
the two (same for the `_m4` variants). Directory names don't match what they
run.

### Composite MORBO

`composite_morbo` models DTLZ2's raw intermediate response (`[g, cos(x_0*pi/2),
sin(x_0*pi/2)]`, 3-dim) instead of the 2 final objectives directly, applying
the known reduction `L` before anything Pareto/HV-shaped sees it
(`morbo/problems/composite_dtlz2.py`). The composition happens once, at
`TRBOState.__init__` (`morbo/utils.py::compose`) — every trust region is
constructed with `objective=self.objective`, so it propagates automatically
with no changes needed in `trust_region.py`. `CompositeDTLZ2` is mathematically
identical to plain `DTLZ2` (verified numerically), so `morbo` vs.
`composite_morbo` is a controlled A/B on the composite-modeling question
specifically, and can reuse an existing `morbo` result from the same
experiment directory without rerunning it:

```
python run_comparison.py fig2_dtlz2_100d composite_morbo 0
python plot_comparison.py fig2_dtlz2_100d 0
```

Saved outputs now include an `objective_history` field (always in objective
space, i.e. post-reduction) alongside the raw `metric_history` — plotting
prefers it, falling back to `metric_history` for older saved runs where the
two are identical anyway.

### LLM-assisted MORBO

`llm_morbo` adds an LLM as an extra candidate source: once per trust region
per BO iteration (not per batch slot), an LLM proposes a handful of *sparse*
perturbations (which of the ~20 typically-perturbed dimensions to adjust, and
by how much — mirroring MORBO's own `sample_tr_discrete_points_subset_d`
subset-perturbation trick, since verbalizing all `d` raw dimensions gives an
LLM little to reason about at high dimension). Candidates are concatenated
into the existing Thompson-sampling pool and screened by the exact same
HVI/scalarization scoring as every other candidate (`morbo/llm_candidates.py`,
wired into `TS_select_batch_MORBO` in `morbo/gen.py`). Uses the official
`anthropic` SDK with structured output (`client.messages.parse`); requires
`ANTHROPIC_API_KEY` in the environment.

```
python run_comparison.py llm_morbo_vehicle_safety morbo 0
python run_comparison.py llm_morbo_vehicle_safety llm_morbo 0
python plot_comparison.py llm_morbo_vehicle_safety 0
```

Recommended on a low-dimensional problem (VehicleSafety, d=5) rather than the
d=100 DTLZ2 setup — verbalizing which of 100 abstract dimensions to touch
gives an LLM little signal, and this hasn't been tested at higher dimension.

Building this exposed two real bugs in the base MORBO candidate-generation
code, both fixed in `morbo/gen.py`: (1) the final candidate-selection line
indexed the un-concatenated normalized `X_cand` while the scoring array
(`value_score`) was sized to the concatenated `X_cand_unnormalized` — any
concatenation (pending points, and now LLM candidates) could select an
out-of-range index; (2) the feasibility masking that excludes
already-selected pending points from reselection assumes they're the *last*
`len(inds_next_in_tr)` rows of the candidate pool — appending anything after
that block (as an early version of the LLM-candidate injection did) silently
broke the exclusion, letting the same point be re-selected and duplicated in
`X_history`.

The number of LLM candidates requested per trust region **decays** as that
trust region accumulates its own local data since its last restart, reusing
the existing `decay_function` (already used elsewhere in `morbo/gen.py` for
`prob_perturb`) rather than inventing a new schedule — a freshly-restarted
TR gets close to the full `llm_candidates_per_tr`, decaying toward a floor
of 1 as its local data approaches the initial-design size. This is
deliberately *not* a Trust-Aware-BO-style Bernoulli accept/reject schedule
(`morbo/gen.py`'s docstring comment on the precompute block explains why:
MORBO's batch-pooled joint scoring has no single "the optimizer's candidate"
to arbitrate a coin flip against, unlike the single-candidate-per-step
sequential loops that pattern comes from) — every candidate is still scored
identically regardless of iteration; only the proposal *count* is scheduled.

Composite modeling and LLM-assisted candidates compose with **zero new
code** — the LLM always proposes in design space, composite structure only
changes downstream scoring, so `evalfn=CompositeDTLZ2` (or any composite
evalfn) plus `use_llm_candidates=True` together just works.

### LLM-automated BoTier (tiered composite utility)

`botier_llm/` is a small **standalone** module (not built on MORBO's
trust-region/HVI machinery — BoTier deliberately skips hypervolume and
multi-region coordination entirely). It implements BoTier's hierarchical
scalarization (`Ξ = Σ(min(ψ_i, t_i) · Π H(ψ_j − t_j))`, `botier_llm/solver.py`,
reusing `morbo/utils.py::get_fitted_model` as-is) with ordinary
single-objective `qExpectedImprovement` BO, and automates the one input
BoTier's own paper leaves to a domain expert — the tier ordering and
per-objective thresholds — via a single one-shot LLM call
(`botier_llm/llm_tiers.py`). Given a fixed tier structure, the BO converges to
*one* compromise point on the Pareto front (the region implied by the stated
priorities), not the whole front — that's the intended behavior, not a
limitation.

```
python run_botier_comparison.py 10 60 0   # dim=10, max_evals=60, seed=0
```

Compares the LLM-proposed tiers against a hand-specified baseline (thresholds
at the median of a Sobol warm-start) on DTLZ5 (M=2), saving both runs' final
recovered point, composite utility value, and threshold-clearing status to
`experiments/botier_dtlz5/`. Requires `ANTHROPIC_API_KEY`.

### Prior work note: composite modeling + MORBO is not new

Combining composite modeling with MORBO specifically was already done in
Maddox, Feng & Balandat, "Optimizing High-Dimensional Physics Simulations
via Composite Bayesian Optimization" (NeurIPS 2021 ML4PS workshop) — Max
Balandat is a MORBO co-author. They pair MORBO with a High-Order Gaussian
Process (HOGP) that models a Kronecker-structured tensor/image raw response
with explicit cross-dimension correlation, on a 177-dim optical design
problem; their composite variant wins early but converges to similar final
hypervolume to non-composite MORBO. The two experiments below are scoped
around what that leaves open, not around "composite MORBO" itself.

### Correlation ablation: does modeling cross-dimension correlation matter?

`morbo/problems/composite_dtlz2_curve.py` builds a *genuinely correlated*
raw response — a discretized curve `h(θ;x) = (1+g(x))·cos(θ - x₀π/2)`
sampled at `K` points, verified numerically equivalent to plain DTLZ2 (the
curve's two endpoints, at θ=0 and θ=π/2, are exactly the two DTLZ2
objectives via `cos(-a)=cos(a)` and `cos(π/2-a)=sin(a)`). Unlike
`composite_dtlz2.py`'s `(g, cos, sin)` raw response, adjacent points on this
curve are strongly correlated by construction — the same kind of structure
Maddox et al.'s tensor/image outputs have.

Three-way comparison, same trust-region machinery throughout: `morbo`
(direct modeling) vs. `independent_gp_composite` (`K` decoupled single-task
GPs, same as `composite_morbo`) vs. `kronecker_gp_composite`
(`botorch.models.KroneckerMultiTaskGP`, jointly modeling correlation across
all `K` raw dimensions — `morbo/utils.py::get_fitted_kronecker_model`,
wired into `TrustRegion.update_model` via a new `use_kronecker_gp` hparam).
The Kronecker model needs `task_covar_prior=None` and a torch/Adam-based fit
(`fit_gpytorch_mll_torch`) — the default scipy-based fit fails on this
model's default task-covariance prior (`sample_all_priors` raises "Must
provide inverse transform to be able to sample from prior").

```
python run_comparison.py correlation_ablation_dtlz2curve morbo 0
python run_comparison.py correlation_ablation_dtlz2curve independent_gp_composite 0
python run_comparison.py correlation_ablation_dtlz2curve kronecker_gp_composite 0
python plot_comparison.py correlation_ablation_dtlz2curve 0
```

### Composite MORBO on a real simulator: Penicillin

`morbo/problems/composite_penicillin.py` forks
`botorch.test_functions.multi_objective.Penicillin`'s ~2500-step Euler
integrator (5 coupled state variables: penicillin concentration, culture
volume, biomass concentration, glucose concentration, CO2) to checkpoint
the state at `K` fixed absolute step indices, giving a `5K+1`-dim raw
response (`+1` for the stopping time). The public implementation only
returns the final state; once a design's trajectory goes inactive (culture
volume exceeds a max, glucose runs out, or the rate flattens), its state
variables simply stop updating, so a checkpoint at any step is exactly the
state at `min(step, stopping_time)` — meaning the last checkpoint (step
2500) is *guaranteed* to equal the true final state regardless of when a
design actually stopped, which is what makes the reduction reproduce
upstream's own output bit-for-bit (verified numerically, `max abs diff =
0.0`). No composite Penicillin benchmark exists anywhere in the papers
reviewed for this project (Maddox et al. never touch Penicillin; it only
ever appears elsewhere as a direct-modeling benchmark).

```
python run_comparison.py penicillin_composite morbo 0
python run_comparison.py penicillin_composite composite_penicillin 0
python plot_comparison.py penicillin_composite 0
```

## Trust-Region Shape Adaptation

**This is the project's main, most-developed line of work.** MORBO's trust
regions were purely isotropic axis-aligned hypercubes — no per-dimension
lengthscale rescaling at all, a step below even the original TuRBO paper.
This work asks whether adapting trust-region *shape* (not just size) to
local data improves search efficiency in high dimension, tests why, and
where it breaks.

**Headline finding**, confirmed across 5 seeds and multiple problem types:
shape adaptation produces large, unanimous, seed-robust wins (+64-72%
hypervolume) on problems with low-dimensional *effective* structure, and a
noise-level effect (no systematic direction) on problems where all
dimensions genuinely matter. A follow-up synthetic problem (`SparseDTLZ2`)
pinned this down precisely: **the governing variable is effective
dimension relative to the eval budget — not nominal dimension, and not
"the gap" between them.**

Six `tr_shape` modes are implemented in `morbo/trust_region.py`
(`TurboHParams.tr_shape` docstring has the full per-mode math):
`isotropic` (baseline, unchanged behavior), `ard_box` (axis-aligned,
ARD-lengthscale-rescaled — the original TuRBO technique; **fails badly**,
worse than the isotropic baseline, a diagnosed curse-of-dimensionality
effect), `pca_ellipsoid` / `ard_pca_ellipsoid` (PCA-rotated ellipsoids —
the main winners), `cma_ellipsoid` (CMA-ES-style persistent covariance
adaptation — the only method that breaks through at the highest dimension
tested within a tight budget), and `mab_shape` (a per-trust-region
multi-armed bandit that learns which of the above to use, recovering the
best of all of them given adequate budget). A `use_linear_kernel` /
`use_dim_scaled_ls_prior` pair of orthogonal kernel-level variants and a
`label="sobol"` pure-random-search baseline are also available.

```
python run_comparison.py tr_shape_dtlz2_100d pca_ellipsoid 0
python plot_comparison.py tr_shape_dtlz2_100d 0
```

**Full results, reproduction instructions, and session-resumption notes
live in `writeup/`, not here** (this README section is a pointer, not the
source of truth — the numbers change as new cluster runs land):
- [`experiments/tr_shape_dtlz2_100d/RESULTS.md`](experiments/tr_shape_dtlz2_100d/RESULTS.md) —
  the primary results document, all dimensions/problems/methods.
- [`writeup/methods.tex`](writeup/methods.tex) (`sec:tr-shape`) — same
  results in paper form.
- [`writeup/FURTHER_DIRECTIONS.md`](writeup/FURTHER_DIRECTIONS.md) — the
  motivating papers, what's been tried, and ranked ideas not yet tried.
- [`writeup/PROJECT_HANDOFF.md`](writeup/PROJECT_HANDOFF.md) — **start
  here** to resume this work in a new session: architecture, headline
  findings, debugged gotchas, and how to run everything.
- [`LITERATURE_REVIEW.md`](LITERATURE_REVIEW.md) — broader related-work
  notes for the project, including the most closely related prior art
  (LABCAT, CMA-BO) and candidate next benchmarks.
- [`cluster/README.md`](cluster/README.md) — how to run any of this on
  Cornell's Unicorn SLURM cluster.

## License
This repository is MIT licensed, as found in the [LICENSE](LICENSE) file.
