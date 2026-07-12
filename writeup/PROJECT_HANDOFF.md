# Project Handoff — MOBO-HighDim (trust-region shape adaptation study)

Written 2026-07-12 to let a fresh session pick this project up cold. Read
this first, then dig into the specific files it points to as needed.

## What this project is

A research fork of MORBO (Multi-Objective Bayesian Optimization over
High-Dimensional search spaces, Daulton et al. 2022 UAI), ported from an
archived upstream (`facebookresearch/morbo`, targeting botorch ~0.6) to
current `botorch==0.9.5`/`gpytorch==1.11`. Repo: `D:/SURP/MOBO-HighDim`,
GitHub `MuhammadAli-CS/MOBO-HighDim`, pushed directly to `main` (no PR
workflow — established convention in this project, confirmed by explicit
prior authorization).

Three extensions were built, in roughly chronological order:
1. **Composite modeling** (`sec:composite`/`sec:kronecker-ablation`/`sec:penicillin`
   in `writeup/methods.tex`) — GP models a raw response, objectives are a
   known reduction of it. Tested independent-per-dim GPs vs. a joint
   Kronecker/HOGP model; independent wins at low data scale (~1/100th the
   fit cost, same or better accuracy).
2. **LLM-assisted MORBO + LLM-automated BoTier** (`botier_llm/`,
   `sec:llm-composite`, `sec:botier`) — LLM-proposed candidates composed
   with GP candidates under the same objective; LLM-automated tiered
   scalarization. **These need `ANTHROPIC_API_KEY` to actually run** —
   currently only mock-tested, not run for real results.
3. **Trust-region shape adaptation** (`sec:tr-shape`) — **this is the
   active, most-developed line of work** and the subject of the rest of
   this document.

## The core idea and finding (trust-region shape adaptation)

MORBO's `TrustRegion` was a pure isotropic hypercube (no per-dimension
lengthscale rescaling at all — a step below even original TuRBO). This
project asks whether adapting trust-region *shape* (not just size) to
local data helps high-dimensional multi-objective BO, and if so, why.

**Headline, now confirmed across 5 seeds and multiple problem types:**
Trust-region shape adaptation (PCA-rotated ellipsoids) produces large,
unanimous, seed-robust wins (+64–72% hypervolume) on problems with
low-dimensional *effective* structure (DTLZ2: 1 informative variable of
nominal `d`), growing with nominal `d`, and produces a noise-level effect
(near coin-flip win rates, no systematic direction) on problems where all
dimensions genuinely matter (Rover: 60 real spline-waypoint dims). The
naive axis-aligned ARD-rescaled box (original TuRBO's own technique)
*actively hurts*, worse than doing nothing — diagnosed as a curse-of-
dimensionality effect (see below). A newer CMA-ES-based covariance
adaptation variant is the only method that breaks through at d=200 within
the standard budget, validating a temporal-smoothing argument from a
paper read partway through the project.

**Full results are written up in two places, kept in sync:**
- `experiments/tr_shape_dtlz2_100d/RESULTS.md` — narrative + tables, the
  primary results document (despite the directory name, covers the *entire*
  tr_shape study across all dimensions/problems/methods).
- `writeup/methods.tex`, section `\label{sec:tr-shape}` — same content in
  paper form, for eventual write-up/publication.

**Do not re-derive these numbers from scratch** — read the two files above
first. What follows here is architecture and how-to-resume info, not a
restatement of results.

## Code architecture

- `morbo/trust_region.py` — `TurboHParams` dataclass carries the
  `tr_shape` hyperparameter (`"isotropic"` default = old behavior exactly,
  `"ard_box"`, `"pca_ellipsoid"`, `"ard_pca_ellipsoid"`, `"cma_ellipsoid"`),
  plus `cma_c_mu`/`cma_c1`/`cma_c_p` (CMA learning rates),
  `use_linear_kernel`, `use_dim_scaled_ls_prior`. `TrustRegion` holds
  buffers `R` (rotation), `axis_lengths`, and (CMA-only) `cma_C`/`cma_path`/
  `cma_prev_center`. `_update_tr_shape()` dispatches by mode, **wrapped in
  `torch.no_grad()`** (critical — see Gotchas below). Read the `tr_shape`
  docstring on `TurboHParams` for the full per-mode math and citations.
- `morbo/utils.py` — `compute_cma_ellipsoid_shape(...)` (CMA update math),
  `HypersphereProjection` (input transform for the linear-kernel variants),
  `get_fitted_model(..., use_linear_kernel=..., use_dim_scaled_ls_prior=...)`.
- `morbo/state.py`, `morbo/run_one_replication.py` — thread the new kwargs
  through to `TurboHParams` construction; `run_one_replication.py`'s
  `supported_labels` list is the authoritative list of runnable experiment
  labels.
- `run_comparison.py` — `LABEL_OVERRIDES` dict maps label name →
  `TurboHParams` overrides. **This is where you add a new named
  variant/combination** without writing new code — e.g.
  `"linear_gp_pca": {"use_linear_kernel": True, "tr_shape": "pca_ellipsoid"}`.
- `morbo/problems/rover.py` — has a try/except penalty fallback for
  spline-fit crashes on near-degenerate candidates (see Gotchas).
- `smoke_test_tr_shape.py` — fast (no full BO loop) validation of every
  new code path: shape math correctness, GP fit with linear kernel / dim
  prior, CMA math unit test. Run this after any change to
  `trust_region.py`/`utils.py` before trusting a cluster submission.
- `aggregate_seeds.py <experiment_name> [--labels a b c]` — loads all
  `<seed>_<label>.pt` result files for an experiment, prints mean/std/
  per-seed hypervolume. Use this for any multi-seed analysis.

## How to run things

**Locally** (per explicit user instruction: keep local runs minimal, the
device is needed for other things — don't run big sweeps locally):
```
python smoke_test_tr_shape.py        # fast, no cluster needed
python run_comparison.py <experiment_name> <label> [seed]
python plot_comparison.py <experiment_name> [seed]
python aggregate_seeds.py <experiment_name>
```

**On the cluster** (Cornell Unicorn) — see `cluster/README.md` for full
detail, summarized here:
```bash
ssh <netid>@unicorn-login-01.coecis.cornell.edu
cd MOBO-HighDim && git pull
bash cluster/submit_smoke.sh                    # validate on cluster first
bash cluster/submit_tr_shape_new_methods.sh     # cma/linear/dimprior @ d=100/150/200
bash cluster/submit_penicillin_2x2.sh           # composite x shape
bash cluster/submit_tr_shape_multiseed.sh       # seeds 1-4, 64 jobs, core methods
squeue -u $USER                                 # check status
git commit && git push                          # or scp results back
```

### Cluster gotchas (all previously debugged, don't rediscover)
- `kilian` is a SLURM **account**, not a partition. Use
  `--partition=aimi --account=kilian`.
- Conda: `source /share/apps/software/anaconda3/etc/profile.d/conda.sh`
  (NOT `~/.bashrc` — fails in non-interactive `--wrap` shells).
- `torch` must be unpinned / installed against `cu128`+ for the B200 GPUs
  (Blackwell, compute capability 10.0); `botorch==0.9.5`/`gpytorch==1.11`
  stay pinned (this port's compatibility fixes were validated against
  those exact versions).
- `aimi` partition (SURP's own, 8xB200/node) is strongly preferred over
  the general `gpu` partition (tops out at A6000). Check `sinfo -p aimi`
  before large submissions.
- CPU (laptop) vs GPU (cluster) runs are **not bit-comparable** even at
  the same seed/config — floating-point non-determinism across
  hardware/BLAS backends. Don't mix absolute values across a CPU-run
  experiment and a GPU-run experiment; use within-run relative comparisons
  when you need to compare across the divide (done explicitly for the
  Penicillin 2×2, flagged in both RESULTS.md and methods.tex).

## Debugged issues, not to re-litigate

1. **Grad-tracking contamination**: GP lengthscale params are
   `requires_grad=True` `nn.Parameter`s. Any code deriving `R`/`axis_lengths`
   from them (or from data touched by them) must be wrapped in
   `torch.no_grad()` and `.detach()`ed, or grad-tracking silently
   propagates into evaluated candidates and eventually crashes a later
   `.cpu().numpy()` call. Already fixed in `_update_tr_shape()`.
2. **Shape recompute was dead code** originally (nested inside a
   `use_noisy_trbo`-only branch that no test config enables). Fixed —
   shape recompute now fires independently whenever `tr_shape != "isotropic"`.
3. **Rover spline crashes**: `scipy.splprep(..., k=3, s=0)` crashes on
   near-duplicate consecutive control points, which a collapsed trust-region
   axis can produce. Fixed with a large finite penalty (`1e6`) instead of
   letting it crash the whole run.
4. **`ard_box` fails at high d — diagnosed, not just observed**: combining
   `d` independent per-axis lengthscale-derived constraints (even at exactly
   preserved total volume) causes combinatorial region collapse — verified
   directly (1/200 points remain inside the region vs 41/200 for PCA's
   shape on identical d=100 data). A dimension-scaled lengthscale prior
   (removing the hard interval ceiling) does **not** fix this — confirms
   the mechanism is structural (treating d estimates as d separate hard
   constraints), not about where the lengthscale values numerically sit.
5. **Windows `np.random.randint` overflow** (local repro only, doesn't
   affect the Linux cluster) — fixed with explicit `dtype=np.int64`.

## Reading order for a fresh session

1. This file.
2. `experiments/tr_shape_dtlz2_100d/RESULTS.md` — all results, narrative.
3. `writeup/FURTHER_DIRECTIONS.md` — the two papers' insights, the
   implementation-status table, and ranked ideas not yet tried (MAB-guided
   per-region shape selection, crossover-point characterization, learned
   objective-aware rotation, etc.) — **start here for "what's next"**.
4. `writeup/methods.tex` `sec:tr-shape` — same results, paper form.
5. `morbo/trust_region.py`'s `TurboHParams.tr_shape` docstring — the
   authoritative technical spec of every mode.
6. `cluster/README.md` — cluster mechanics.

## Open threads / natural next steps (see FURTHER_DIRECTIONS.md §3 for full list)

- MAB-guided per-trust-region shape selection (AS-SMEA's own answer to
  "no single shape wins everywhere" — directly turns this project's
  conditional-benefit finding into an adaptive strength). Not yet coded.
- Crossover-point characterization: finer dimension grid (60,70,80,90)
  between DTLZ2's "no effect" (d=50) and "dramatic" (d=100) regimes.
- A problem with partial effective-dimension structure (e.g. 5 informative
  of 100 dims) to test whether benefit scales with the *gap* between
  nominal and effective dimension, as the mechanism predicts.
- Learned/objective-aware rotation (current PCA/CMA are both
  variance-driven, not objective-gradient-driven).
- Multi-seed the new-methods sweep (`cma_ellipsoid`, `linear_gp_pca`, dim-
  prior variants) — currently single-seed only, unlike the core 4 methods.
