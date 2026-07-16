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
nominal `d`), and produces a noise-level effect (near coin-flip win rates,
no systematic direction) on problems where all dimensions genuinely matter
(Rover: 60 real spline-waypoint dims). The naive axis-aligned ARD-rescaled
box (original TuRBO's own technique) *actively hurts*, worse than doing
nothing — diagnosed as a curse-of-dimensionality effect (see below). A
CMA-ES-based covariance adaptation variant is the only method that breaks
through at d=200 within the standard budget, validating a
temporal-smoothing argument from a paper read partway through the project.

**Correction, established by the `SparseDTLZ2` follow-up:** the effect is
governed by *effective* dimension relative to the eval budget, not nominal
dimension, and not "the gap" between them (an earlier, now-superseded
framing). Holding effective dimension fixed at 6 while scaling nominal `d`
from 60→200 produces a flat null (<0.3% everywhere); holding nominal `d`
fixed at 100 while scaling effective dimension from 3→51 produces a clean
monotonic dose-response (0.1%→10%). Plain DTLZ2's dimension sweep was
measuring effective dimension all along, since nominal `d` there is
inseparable from it by construction.

**A second follow-up, `mab_shape`** (a per-trust-region bandit that learns
which fixed shape to use, rather than committing to one globally): at the
standard 600-eval budget it's the single best method of all 8 tested at
d=100 (+69.2%), but ties the failing isotropic baseline at d=150/200 —
its own exploration cost eats the narrow late-arriving breakthrough margin
the fixed shapes needed there. A 2000-eval extended-budget rerun resolved
this as a **pure budget artifact**: `mab_shape` fully recovers and becomes
the single best method of every shape tested at d=200 (33.61 vs.
`ard_pca_ellipsoid`'s 33.15), and a strong second at d=150.

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
  `"ard_box"`, `"pca_ellipsoid"`, `"ard_pca_ellipsoid"`, `"cma_ellipsoid"`,
  `"mab_shape"`), plus `cma_c_mu`/`cma_c1`/`cma_c_p` (CMA learning rates),
  `use_linear_kernel`, `use_dim_scaled_ls_prior`, and
  `mab_epsilon`/`mab_reward_ema_alpha`/`mab_arms` (bandit config — see
  below). `TrustRegion` holds buffers `R` (rotation), `axis_lengths`, (CMA
  only) `cma_C`/`cma_path`/`cma_prev_center`, and (mab_shape only)
  `mab_arm_values`/`mab_arm_pulls`/`mab_last_arm`. `_update_tr_shape()`
  dispatches by mode via `_compute_shape_for_mode(shape)`, **wrapped in
  `torch.no_grad()`** (critical — see Gotchas below). Read the `tr_shape`
  docstring on `TurboHParams` for the full per-mode math and citations.
  - **`tr_shape="mab_shape"`**: a per-trust-region epsilon-greedy bandit
    over `{isotropic, ard_box, pca_ellipsoid, ard_pca_ellipsoid, cma_ellipsoid}`.
    Reward is binary — 1.0 if this TR's existing success-streak counter
    (`n_successes`) was just incremented, else 0.0 — folded into a per-arm
    EMA (`_select_mab_arm`). Motivated by this project's own finding that no
    single fixed shape wins on every problem (PCA wins on DTLZ2, no shape
    robustly wins on Rover); mirrors AS-SMEA's own answer to that exact
    problem (Wang et al. 2026, Sec. 3.3, their LS-IMA/MASS). Label:
    `mab_shape`. **Results are in** (`RESULTS.md` §6): best of 8 methods at
    d=100/600ev (+69.2%); ties the failing baseline at d=150/200/600ev but
    a 2000-eval rerun confirmed this was a pure budget artifact — it fully
    recovers and becomes the single best method of every shape tested at
    d=200/2000ev.
- `morbo/utils.py` — `compute_cma_ellipsoid_shape(...)` (CMA update math),
  `HypersphereProjection` (input transform for the linear-kernel variants),
  `get_fitted_model(..., use_linear_kernel=..., use_dim_scaled_ls_prior=...)`.
- `morbo/problems/sparse_dtlz2.py` (new, `evalfn="SparseDTLZ2"`) — DTLZ2
  variant that masks all but `k_eff` of the `k = dim - M + 1` distance
  dimensions out of `g(x)` entirely (the masked ones are literal no-ops on
  every objective), so nominal and effective dimension can be varied
  independently. **Results are in** (`RESULTS.md` §7): nominal `d` alone
  (60→200, effective dim pinned at 6) is a flat null; effective dim alone
  (fixed d=100, k_eff 2→50) gives a clean 0.1%→10% dose-response. This
  **corrected an earlier "gap between nominal and effective dimension"
  framing** — the governing variable is effective dimension relative to
  budget, full stop, not the gap and not nominal dimension.
- `label="sobol"` (`morbo/run_one_replication.py`, a self-contained branch
  near the top of the function, before `TurboHParams`/`TRBOState` are ever
  constructed) — pure random search: one continuous `SobolEngine` sequence
  over the full space, no trust regions or GP fitting. Answers whether
  TuRBO/MORBO's local-modeling machinery is earning its keep at all, a more
  basic question than any `tr_shape` comparison (which all implicitly
  assume it is). **Results are in** (`RESULTS.md` §8): decisive — MORBO
  beats sobol by +38.5% (d=50) up to +3800% (d=100, sobol barely clears the
  reference point) and +74.6% on Rover. Shape adaptation's own wins sit on
  top of this larger base advantage, not instead of it.
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
3. `writeup/FURTHER_DIRECTIONS.md` — motivating papers, implementation
   status (everything originally proposed is now done), and a
   literature-informed "Where to steer this next" section (§ at the
   bottom) — **start here for "what's next"**.
4. `LITERATURE_REVIEW.md`'s "Follow-up review: trust-region shape
   adaptation" section — the fuller literature pass behind that "what's
   next" list (closest prior art to differentiate from: LABCAT, CMA-BO;
   better bandit designs for `mab_shape`; real benchmark candidates to
   validate the effective-dimension finding beyond `SparseDTLZ2`).
5. `writeup/methods.tex` `sec:tr-shape` — same results, paper form, now
   with a related-work paragraph differentiating from LABCAT/CMA-BO.
6. `morbo/trust_region.py`'s `TurboHParams.tr_shape` docstring — the
   authoritative technical spec of every mode.
7. `cluster/README.md` — cluster mechanics.

## Open threads / natural next steps

**QUEUED: BBOB-style landscape taxonomy, `RESULTS.md` §12**
(`cluster/submit_bbob_style.sh`, 160 jobs) — tests the `ard_box`
landscape-dependence finding (rugged-`g` DTLZ3/7 wins, smooth-`g` DTLZ2/5
fails) against BBOB's own 5-category taxonomy, and gives a within-study
comparison point to LABCAT (closest prior art, evaluated on COCO/BBOB).
**Read `morbo/problems/bbob_style.py`'s docstring before citing numbers**:
faithful-in-spirit reimplementations, not the official `cocoex` package —
internally comparable only, not directly comparable to published
COCO/LABCAT tables. Coded, unit-tested (shapes, finiteness, `k_eff`
no-op-masking, optimum-location sanity), end-to-end BO-loop smoke-tested.

**20-seed confirmation program: DONE, results in `RESULTS.md` §11** —
core results confirmed at 20/20-or-0/20 unanimity; three revisions (Rover
conclusively null; PCA's keff50 edge was seed noise, cma is the robust
rotation-invariant winner; mab_shape's d=100 "best" was a lucky seed —
+33.5% mean with std 7.2); one inverted prediction (under mid-run
dimension switching, cma WINS and the epsilon-greedy bandit is the memory
casualty — reward-space memory is more fragile than covariance-space
memory); composite×shape at d=100 shows they STACK (geometry not
redundant; composite adds a 5× variance reduction on top of shape).

**Bandit line: CLOSED, results in `RESULTS.md` §11g–h.** Two iterations
on `mab_shape` fixed it: `mab_shape_ducb` (discounted-UCB arm selection,
Garivier & Moulines 2011) cut epsilon-greedy's d=100 variance from std 7.2
to std 3.5 (+69.8%, 20/20); `mab_shape_ducb_shared` (CMA covariance
advances at every update regardless of active arm) closed the gap
entirely — 32.32±1.88, 20/20, statistically tied with the best fixed
Matérn arm. **`mab_shape_ducb_shared` is the final recommended bandit
configuration.** The one regime that stays unsolved, d=200/tight budget,
is now understood precisely rather than just observed: with staleness
eliminated, the remaining blocker is that the reward signal (success
increments) carries zero information there until something breaks
through — not fixable by any selection policy; the honest options are
commitment (run `cma_ellipsoid` outright) or a differently-informative
reward. See `FURTHER_DIRECTIONS.md` "Next phase" for the full ranked
queue (contextual gating by measured input-space effective dim, noise
experiments, remaining benchmarks).

**Benchmark battery: DONE, results in `RESULTS.md` §10 (single-seed) —
superseded/confirmed by the 20-seed program in §11, see above.** At
single-seed §10 vintage: two honest prediction failures sharpened the
headline claim — LassoBench-MO (30 seeds, their paper's protocol) shows
MORBO externally competitive with their dedicated single-objective methods
(DNA 0.304 vs. their TuRBO 0.292) but **no shape-adaptation benefit**
(effective dim there is coefficient sparsity, not input no-ops);
SparseRover's no-op padding also failed to unlock the benefit — both
findings held up at 20 seeds (§11b). §10's two open loose ends were both
closed by §11: RotatedSparseDTLZ2's "pca_ellipsoid demoted, needs a
multi-seed rerun to interpret" was resolved in §11c (seed noise;
`cma_ellipsoid` is the real, robust rotation-invariant winner), and
TimeVaryingSparseDTLZ2's k_eff=5 null-by-design-flaw was rerun at k_eff=49
in §11d (the max value allowing disjoint masks at d=100) — not the k_eff=50
originally guessed, but the same regime. DTLZ3/5/7 generalization
(PCA variants 5/5 everywhere informative, +8-21%; ard_box's failure
revealed as landscape-dependent, winning on rugged-g DTLZ3/7) was
confirmed at 19-20/20 in §11a.

Everything originally proposed in `FURTHER_DIRECTIONS.md` (multi-seed
sweep, composite×shape, CMA/linear-kernel/dim-prior, `mab_shape`,
`SparseDTLZ2`, and the `sobol` random-search baseline) is now done and
written up. Top literature-informed
priorities (full list in `FURTHER_DIRECTIONS.md`'s last section):

- **Validate the effective-dimension finding on a real problem** —
  LassoBench made bi-objective is the best available bridge from our
  synthetic `SparseDTLZ2` result to something reviewers won't dismiss as
  self-constructed.
- **Upgrade `mab_shape` to a contextual bandit** using an online
  effective-dimension estimate (top-k PCA eigenvalue mass ratio, already
  computed by `pca_ellipsoid`) as context — turns the empirical
  "recovers the best of both worlds" result into a mechanistic one.
- **Anneal `mab_epsilon`** (decay exploration as budget is consumed) — a
  budget-*efficiency* improvement now (the d=150/200 failure itself is
  resolved as a budget artifact, not a design flaw — `mab_shape` fully
  recovers and even wins at 2000 evals), not a fix for a standing flaw.
- **Learned/objective-aware rotation** — current PCA/CMA are both
  variance-driven, not objective-gradient-driven. Not yet coded.
- **Multi-seed the new-methods sweep** (`cma_ellipsoid`, `linear_gp_pca`,
  dim-prior variants, `mab_shape`, `SparseDTLZ2`) — everything past the
  original 4-method core sweep is currently single-seed only. The
  `ard_pca_ellipsoid` loss at `SparseDTLZ2` effective-dim 51 (§7,
  RESULTS.md) is a specific single-seed result worth confirming first.
