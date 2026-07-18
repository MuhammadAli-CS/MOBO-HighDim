# Running on Cornell's Unicorn cluster

One-time setup, then submit jobs from the login node — never run heavy
compute directly on the login node, it kills resource-intensive processes
automatically.

## 1. Connect

```
ssh <netid>@unicorn-login-01.coecis.cornell.edu
```

Requires being on-campus or on the Cornell VPN.

## 2. Get the code onto the cluster

```
git clone https://github.com/MuhammadAli-CS/MOBO-HighDim.git
cd MOBO-HighDim
```

## 3. Build the environment (one time only)

Do this from an interactive allocation, not the login node:

```
salloc --mem=8g --cpus-per-task=4 --time=01:00:00 --partition=default_partition-interactive --account=kilian
bash cluster/setup_env.sh
exit   # end the interactive allocation once it finishes
```

This creates a conda env at `~/morbo-env` (Python 3.11). `torch` is left
unpinned (the cluster's B200 GPUs need a CUDA build recent enough to
support Blackwell/compute capability 10.0, which the `cu121` channel used
locally doesn't have), but `botorch==0.9.5`/`gpytorch==1.11` stay pinned to
the exact versions this codebase's compatibility fixes were validated
against (see the main `README.md` "Fork notes" section — the port
specifically patches botorch APIs that changed between the archived
upstream's target, `~0.6`, and `0.9.5`). Installing a newer botorch
unpinned risks it having moved its API again since 0.9.5, the same class of
break the port already had to fix once. Installs this repo into the env
with `pip install -e .`.

## 4. Fig2-scale correlation-ablation follow-up (composite_curve_dtlz2_100d)

`correlation_ablation_dtlz2curve` (d=20) found only a ~0.3% composite-vs-direct
HV margin, much smaller than `fig2_dtlz2_100d`'s g/cos/sin composite's +25%
margin at d=100. `composite_curve_dtlz2_100d` reruns the same genuinely-
correlated 8-point curve construction at fig2's scale (d=100, 600 evals,
batch 50) to test whether that gap was a dimensionality effect. All three
labels (`morbo`, `independent_gp_composite`, `kronecker_gp_composite`) run
here — the Kronecker one specifically was moved off the laptop for this
config, since it was already ~100x the independent-GP fit cost at the
smaller d=20 scale and is expected to be substantially worse at d=100:

```
bash cluster/submit_composite_curve_100d.sh
```

Check status with:
```
squeue -u $USER
```

Each job writes its own log to `cluster/logs/<label>-100d_<jobid>.out`. Once
all three finish, a dependent plot job runs automatically
(`--dependency=afterok`) and generates
`experiments/composite_curve_dtlz2_100d/comparison_seed0.png` via
`plot_comparison.py`'s auto-discovery — no manual replotting needed.

## 4b. Trust-region shape adaptation (tr_shape_dtlz2_100d)

MORBO's trust regions were purely isotropic hypercubes — no per-dimension
lengthscale rescaling at all (a step below even the original TuRBO paper's
own technique). `tr_shape_dtlz2_100d` compares three alternative geometries
against the isotropic baseline at fig2 scale (d=100, 600 evals, batch 50):
`ard_box` (axis-aligned, rescaled by the TR's fitted GP ARD lengthscales),
`pca_ellipsoid` (rotated into the PCA frame of the TR's local data), and
`ard_pca_ellipsoid` (PCA rotation + lengthscale-reweighted axis widths).
See `morbo/trust_region.py`'s `TurboHParams.tr_shape` docstring for the
full design rationale (why a rotated *box* rather than a true ellipsoid,
why the isotropic baseline is provably unaffected, etc.):

```
bash cluster/submit_tr_shape_100d.sh
```

Same job-log/auto-plot pattern as `composite_curve_dtlz2_100d` above. The
isotropic baseline (`morbo`) is reused from a file already committed to the
repo (`experiments/tr_shape_dtlz2_100d/morbo/0000_morbo.pt`, identical to
`fig2_dtlz2_100d`'s own `morbo` result) rather than resubmitted — `git pull`
already has it, no job needed.

## 4c. New shape methods, robustness, and 2×2 (post-sweep follow-ups)

After the initial sweep, four more submission scripts were added (see
`writeup/FURTHER_DIRECTIONS.md` for the motivation and paper sources).
**Run `submit_smoke.sh` first** — it validates every new code path with a
tiny BO loop on the cluster before the multi-hour sweeps:

```
bash cluster/submit_smoke.sh          # ~5 min; wait for "All smoke tests passed."
bash cluster/submit_tr_shape_new_methods.sh   # cma_ellipsoid, linear-kernel, dim-prior fix @ d=100/150/200
bash cluster/submit_penicillin_2x2.sh          # composite modeling × shape adaptation, Penicillin
bash cluster/submit_tr_shape_multiseed.sh      # seeds 1–4 for the core methods (64 jobs — check sinfo first)
```

New methods (all reuse the existing `run_experiment.sub`, so the 16-CPU /
64 GB / 1-GPU / aimi / kilian config applies):
- `cma_ellipsoid` — CMA-ES covariance adaptation (AS-SMEA).
- `linear_gp` / `linear_gp_pca` / `linear_gp_cma` — spherically-projected
  linear kernel (linear-bo challenge baseline), alone and crossed with shape.
- `ard_box_dimprior` / `ard_pca_dimprior` — dimension-scaled lengthscale
  prior (Hvarfner) as a candidate fix for `ard_box`'s high-d collapse.
- `composite_penicillin_pca` / `_ard_pca` — composite modeling × shape.

After the multi-seed jobs land, aggregate with:
```
python aggregate_seeds.py tr_shape_dtlz2_100d   # mean ± std per method
```

## 4d. Bandit-guided shape selection and the effective-dimension sweep

**Status: complete, results in `experiments/tr_shape_dtlz2_100d/RESULTS.md`
§6-7.** Commands kept here for reproduction (see
`writeup/FURTHER_DIRECTIONS.md` for full motivation):

```
bash cluster/submit_mab_shape.sh       # mab_shape @ d=100/150/200 + Rover
bash cluster/submit_sparse_dtlz2.sh    # SparseDTLZ2, 9 experiments x 4 methods
```

`mab_shape` (new `tr_shape` mode) is a per-trust-region epsilon-greedy
bandit over `{isotropic, ard_box, pca_ellipsoid, ard_pca_ellipsoid,
cma_ellipsoid}`, rewarded by whether the TR's own success streak just
incremented. Motivated by this project's own finding that no single fixed
shape wins everywhere (PCA wins on DTLZ2, no shape robustly won on Rover) --
mirrors AS-SMEA's own answer to that (Wang et al. 2026, Sec. 3.3).

`SparseDTLZ2` (new `evalfn`) masks all but `k_eff` of DTLZ2's distance
dimensions out of `g(x)` entirely, so nominal and effective dimension can be
varied independently -- tests whether shape adaptation's benefit tracks the
*gap* between them (as this project's diagnosis implies) or nominal
dimension alone (which plain DTLZ2 can't distinguish, since its own `k`
grows with nominal `d`). `cluster/submit_sparse_dtlz2.sh` runs two sweeps:
nominal `d` at fixed effective dim, and effective dim at fixed nominal `d`.

## 4e. New benchmark battery: real problems + mechanism-probing synthetics

**Status: complete, results in `experiments/tr_shape_dtlz2_100d/RESULTS.md`
§10.** Commands kept here for reproduction (see the problem files'
docstrings in `morbo/problems/` for full rationale):

```
bash cluster/submit_real_benchmarks.sh              # stage 1: LassoBench synt_medium (30 seeds!) + SparseRover (240 jobs)
bash cluster/submit_real_benchmarks.sh dna          # + LassoBench DNA (180 jobs)
bash cluster/submit_real_benchmarks.sh synt_high    # + LassoBench synt_high (180 LONG 5000-eval jobs)
bash cluster/submit_new_synthetic.sh                # RotatedSparseDTLZ2 + TimeVaryingSparseDTLZ2 (70 jobs)
bash cluster/submit_dtlz_variants.sh                # DTLZ1/3/5/7 landscape variants @ d=100 (80 jobs)
```

**LassoBench must be installed in morbo-env first** (the submit script
checks and refuses otherwise). Either re-run `bash cluster/setup_env.sh`
(now installs it automatically at the end) or manually:

```
conda activate $HOME/morbo-env
git clone https://github.com/ksehic/LassoBench.git ~/LassoBench
pip install -e ~/LassoBench
```

The LassoBench experiments use **30 seeds** deliberately -- that matches
the LassoBench paper's own "30 repetitions per method" protocol (and its
budgets: 1000 evals for synt_medium/DNA, 5000 for synt_high), so our
best-validation-loss curves are directly comparable against their
published TuRBO/CMA-ES/Sparse-HO numbers, not just internally against our
own baseline. Mind the queue: stage 1 alone is 240 jobs.

## 4f. Twenty-seed confirmation program + composite×shape at $d{=}100$

**Status: complete, results in `experiments/tr_shape_dtlz2_100d/RESULTS.md`
§11.** Every core comparison rerun at 20 seeds (matching the MORBO paper's
own replication count), plus the composite×shape factorial rerun at
$d{=}100$ (where shape adaptation actually has an effect for it to
interact with -- the original Penicillin 2×2 ran at $d{=}7$, a shape-null
regime):

```
bash cluster/submit_followups.sh    # 48 jobs: resolves specific open questions from the main sweep
bash cluster/submit_20seed.sh       # stage A: conclusion-changing reruns (~435 jobs)
bash cluster/submit_20seed.sh paper # stage B: protocol-matching reruns (~450 jobs)
bash cluster/submit_composite_shape_100d.sh  # 15 jobs: composite x shape at d=100
```

Three single-seed headline claims did **not** survive 20 seeds and were
corrected in the writeup (not silently overwritten -- see RESULTS.md §11
for what changed and why): the Rover-family "+4.9%" shrank to a
conclusively null "+0.4%"; `pca_ellipsoid`'s apparent edge under a rotated
effective subspace was mostly seed noise (`cma_ellipsoid` is the real,
robust winner there); and `mab_shape`'s single-seed "best method at
$d{=}100$" showing was a lucky draw (20-seed mean +33.5% with std 7.2,
not the best method) -- which directly motivated §4g below.

## 4g. Fixing `mab_shape`'s variance and non-stationarity failures

**Status: complete, results in `experiments/tr_shape_dtlz2_100d/RESULTS.md`
§11g-h.** Two iterations, each fixing a measured failure of the last:

```
bash cluster/submit_mab_ducb.sh     # 45 jobs: discounted-UCB arm selection
bash cluster/submit_mab_shared.sh   # 45 jobs: + shared CMA covariance state
```

`mab_shape_ducb` (`mab_policy="ducb"`) replaces epsilon-greedy with
discounted UCB (Garivier & Moulines 2011): decayed per-arm reward
sums/counts whose exploration bonus regrows for stale arms and anneals as
counts grow, targeting the two failure modes §4f's 20-seed program
measured (stale reward estimates under non-stationarity; a fixed
exploration tax at tight budgets). `mab_shape_ducb_shared`
(`mab_shared_cma=True`) additionally advances the CMA covariance at every
shape update regardless of which arm is played, so arm-switching no longer
starves `cma_ellipsoid`'s state of updates. Net result: the final
configuration (`mab_shape_ducb_shared`) is statistically tied with the
best fixed shape at $d{=}100$ (closing the variance problem entirely,
+75.7\% at 20/20), and the one regime that still fails (tight-budget
$d{=}200$) is now understood as information-theoretic rather than
design-fixable -- every arm's reward is zero until a breakthrough that
never comes, so no selection policy has anything to act on. See
RESULTS.md §11h for the full diagnosis.

## 4h. BBOB-style landscape taxonomy

**Status: DONE, results in -- all 8 experiments x 4 methods x 5 seeds landed.**

```
bash cluster/submit_bbob_style.sh   # 160 jobs: 8 experiments x 5 seeds x 4 methods (already run)
```

Tested whether the `ard_box` landscape-dependence finding (wins on rugged-$g$
DTLZ3/7, fails on smooth-$g$ DTLZ2/5 -- RESULTS.md §10e/11a, from only 4
hand-picked functions) generalizes to BBOB's own 5-category landscape
taxonomy, and gave a within-study comparison point to LABCAT (our closest
prior art, evaluated on COCO/BBOB). **Read
`morbo/problems/bbob_style.py`'s docstring before citing any numbers from
this batch**: these are faithful-in-spirit reimplementations of
representative BBOB functions, not the official `cocoex` package -- results
are internally comparable across our own methods, not directly comparable
to published COCO/LABCAT hypervolume tables. **Headline**: the effect is
real and mostly significant but an order of magnitude smaller than DTLZ2's
(1-9% vs. +66.6%); `cma_ellipsoid` is the most robust geometry; two of four
stated-in-advance predictions were refuted informatively (`peaks_peaks` was
not near-null; the `rastrigin_rastrigin` `k_eff` dose-response did not
reproduce `SparseDTLZ2`'s clean curve); `sphere_sphere` (predicted trivial)
showed the largest gain, pointing to a second "trajectory-alignment"
contributor to shape adaptation's benefit. Full numbers, per-experiment
table, and discussion in RESULTS.md §12 and `writeup/methods.tex` §7.6.

## 4i. `labcat_style`: LABCAT's own construction, implemented directly

**Status: DONE, results in -- all 42/42 experiments x 5 seeds landed.**

```
bash cluster/submit_labcat_style.sh   # 210 jobs: 42 experiments x 5 seeds (already run)
```

Implements LABCAT's actual mechanism (fitness-weighted PCA computed
genuinely in lengthscale-whitened coordinates, rotation kept directly) as
`tr_shape="labcat_style"` -- the opposite order from `ard_pca_ellipsoid`,
which computes an unweighted PCA rotation first and only reweights axis
widths by lengthscale afterward. Closes the completeness gap: §12 tests
LABCAT's own *benchmark family* against our shape variants; this tests
LABCAT's own *construction* directly, across every experiment in the
tr_shape study that already has other shape-variant baselines to compare
against. **Result: our own ordering wins the direct comparison.**
`labcat_style` loses consistently to `pca_ellipsoid`/`ard_pca_ellipsoid`
wherever a real signal exists, including on `bbob_rosenbrock_rosenbrock`
(LABCAT's paper reports its construction winning specifically there --
our reimplementation instead loses tightly/significantly to all three of
our own shape variants, and isn't even distinguishable from plain
isotropic `morbo`). At `tr_shape_dtlz2_100d` it beats `morbo` (+47.2%) but
loses to `pca_ellipsoid` (-11.4%, CI excluding zero). Sharpest result: a
collapse to exactly 0 hypervolume in 3/5 seeds at d=150 -- `morbo`'s own
failure signature -- while both PCA variants stay healthy throughout,
suggesting whiten-then-PCA is more fragile at high d than
PCA-first-then-reweight. Two disclosed confounds (the multi-objective
weighting substitution, and LABCAT being tuned/evaluated in a
lower-dimensional single-objective regime) temper how far this
generalizes back to LABCAT's own setting -- see RESULTS.md §13 and
`writeup/methods.tex`'s `labcat_style` subsection for full numbers, the
exact mechanism, and honest scoping.

## 4j. Five targeted budget extensions (600 -> 2000 evals)

**Status: coded, QUEUED -- not yet run.**

```
bash cluster/submit_bbob_rastrigin_keff20_2000ev.sh      # 25 jobs: 5 methods x 5 seeds
bash cluster/submit_150d_2000ev_paired.sh                 # 12 jobs: 3 methods x 4 new seeds
bash cluster/submit_200d_2000ev_paired.sh                 # 16 jobs: 4 methods x 4 new seeds
bash cluster/submit_tr_shape_methods_200d_2000ev.sh       # 45 jobs: 9 methods x 5 seeds
bash cluster/submit_dtlz1_100d_2000ev.sh                  # 25 jobs: 5 methods x 5 seeds
```

**Deliberately not a blanket budget increase across the study.** The
original MORBO paper (Daulton et al., UAI 2022) ran DTLZ2/3/5/7 and Rover
at 2000 evals, batch 50 -- confirmed directly from this repo's own initial
commit (`be9c062`, before any of our changes), which still has Meta's own
reference `experiments/*/config.json` files. Our whole study instead
standardized on 600 evals. Rather than mechanically re-running everything
at 2000 to "match the paper" (~1,400+ jobs across every experiment, most
of which already show clean, statistically significant, non-budget-limited
separation at 600), these five extensions target only the specific results
where 600 evals left a genuine open question -- is the current number a
real effect/null, or an artifact of insufficient budget:

- **`bbob_rastrigin_rastrigin_keff20_2000ev`**: the k_eff=20 Group B point
  (RESULTS.md §12) was a flat null at 600 evals for every shape variant.
  Does a real signal appear with more budget, or does it stay null?
- **`tr_shape_dtlz2_150d_2000ev` (seeds 1-4 for `morbo`/`pca_ellipsoid`/
  `ard_pca_ellipsoid`)**: this experiment already has `labcat_style` at 5
  seeds (from `submit_labcat_style.sh`), and the existing seed-0-only data
  is striking -- `labcat_style`'s well-documented d=150/600-eval collapse
  (RESULTS.md §13) fully recovers at 2000 evals, landing close to
  `ard_pca_ellipsoid`. But the other three labels only have seed 0 here,
  so there's no paired multi-seed comparison yet.
- **`tr_shape_dtlz2_200d_2000ev` (seeds 1-4 for `morbo`/`pca_ellipsoid`/
  `ard_pca_ellipsoid`/`ard_box`)**: same gap as d=150 -- `labcat_style`
  already has 5 seeds here, the other four labels only have seed 0.
- **`tr_shape_methods_dtlz2_200d_2000ev`** (new experiment, mirrors the
  existing `_2000ev` naming convention): the mab-bandit line's one
  unsolved regime (RESULTS.md §11h) is diagnosed as information-theoretic
  at d=200/600 evals -- `mab_shape_ducb_shared` scores exactly 0.00 on
  4/5 seeds because "the reward signal itself carries zero information
  until something breaks through." The base d=200 dimension-sweep point
  already showed a similar collapse partially resolve at 2000 evals; this
  tests whether the same is true for the bandit/cma methods specifically,
  or whether the diagnosis holds even with 3.3x the budget.
- **`tr_shape_dtlz1_100d_2000ev`** (new experiment): DTLZ1 is a flat null
  for every method at 600 evals ("uninformative... in 600 evals" --
  methods.tex §7.1). Unlike DTLZ2/3/5/7, DTLZ1 isn't one of the original
  MORBO paper's own benchmarks, so this isn't a budget-matching
  correction -- just direct verification of an assumption currently stated
  as fact rather than tested at a longer horizon.

## 5. LLM-dependent parts (Parts 2 and 3)

```
export ANTHROPIC_API_KEY=sk-ant-...
bash cluster/submit_llm.sh
```

Confirm compute nodes (not just the login node) have outbound HTTPS access
before relying on this — some clusters firewall compute nodes off from the
public internet. If they don't, these two parts need to run from an
interactive login-node-adjacent session instead of a batch job.

## Partitions and accounts

`kilian` is a SLURM **account**, not a partition — `scontrol show partition
kilian` returns "not found". The real partitions on Unicorn are
`default_partition`, `gpu`, `spark`, and `aimi` (each with a `-interactive`
variant with a 2-day time limit, for `salloc`). Your priority (per Cornell's
onboarding email) comes from the `kilian` **account**, not a partition name —
confirmed via `sacctmgr -p show assoc user=<netid>`, which lists `kilian` as
your account.

All scripts here therefore use `--partition=aimi --account=kilian` for actual
BO runs, and `--partition=default_partition --account=kilian` for the
lightweight plot jobs that don't need a GPU. `aimi` is the SURP program's
own partition (`aimi-compute-[01-03]`: 224 CPUs / ~2TB RAM / 8x NVIDIA B200
per node) — access is gated by the `en-cc-unicorn-aimi-users` group
(`scontrol show partition aimi`'s `AllowGroups`), confirmed present via `id`,
with `AllowAccounts=ALL` so the existing `kilian` account works there too.
This is meaningfully more powerful than the general `gpu` partition (whose
best nodes top out at RTX A6000), so it's used in preference to `gpu` for
every job here. Check current node load with `sinfo` before submitting if
you want to avoid queueing behind fully-`alloc`'d nodes — GPU
type isn't pinned by default, so jobs land on whatever's free rather than
waiting for a specific card.

## Pulling results back down

```
# from your laptop
scp -r <netid>@unicorn-login-01.coecis.cornell.edu:~/MOBO-HighDim/experiments/composite_curve_dtlz2_100d ./experiments/
```

Or just `git commit && git push` from the cluster checkout directly (mind
`.gitignore` — `experiments/**/*.pt` may be excluded; check before assuming
a push carries the result files).

## Adjusting resources

`cluster/run_experiment.sub` requests `--mem=64g --gres=gpu:1
--cpus-per-task=16 --time=08:00:00` for every job. The Kronecker-GP job here
is the one to watch — at d=20/200 evals it was already ~100x the
independent-GP fit cost (see `experiments/correlation_ablation_dtlz2curve/RESULTS.md`),
and this run is at 5x the input dimension and 3x the eval budget, so it's
the most likely one to need more memory or time. If it gets OOM-killed or
preempted-and-requeued repeatedly, bump `--mem` first (try 32g), and check
`sacct -j <jobid> --format=MaxRSS,Elapsed,State` after a run to see which
limit was actually hit.
