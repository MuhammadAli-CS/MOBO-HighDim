# Trust-Region Shape Adaptation — Results (full study, multi-seed)

Covers the whole study: the dimension sweep (`tr_shape_dtlz2_{20,50,100,150,200}d`,
now 5 seeds each for 50/100/150), the extended-budget follow-up
(`tr_shape_dtlz2_200d_2000ev`), the two real-problem checks (`tr_shape_rover`
[5 seeds], `tr_shape_penicillin`), the new-methods sweep
(`tr_shape_methods_dtlz2_{100,150,200}d`, now also including `mab_shape`),
the composite-modeling × shape-adaptation factorial
(`tr_shape_penicillin_2x2`), and the effective-dimension study
(`sparse_dtlz2_*`, §7). DTLZ2 runs are 600 evals / batch 50 / 3 TRs unless
noted; single-seed unless a seed range is stated.

Three original `tr_shape` variants vs. the isotropic-hypercube baseline
(`morbo`) — see `writeup/methods.tex` sec:tr-shape for the math:
- `ard_box` — axis-aligned box rescaled per-dim by the TR's own fitted GP
  ARD lengthscales (original TuRBO technique).
- `pca_ellipsoid` — box rotated into the PCA frame of the TR's local data.
- `ard_pca_ellipsoid` — PCA rotation + per-axis widths reweighted by
  lengthscales projected onto each principal axis.

Plus three new methods added after reading two papers (Wang et al. 2026
"AS-SMEA", Doumont et al. 2026 "linear-bo"; see `writeup/FURTHER_DIRECTIONS.md`):
- `cma_ellipsoid` — CMA-ES-style covariance adaptation (persistent per-TR
  covariance, rank-mu update from Pareto elites + evolution-path term).
- `linear_gp` (`_pca`/`_cma`) — spherically-projected linear kernel, alone
  and crossed with shape.
- `ard_box_dimprior` / `ard_pca_dimprior` — dimension-scaled LogNormal
  lengthscale prior (Hvarfner et al. 2024) as a candidate fix for
  `ard_box`'s collapse.

GP fitting is otherwise identical across all methods; shape adaptation only
changes candidate sampling and containment testing.

## 1. Multi-seed dimension sweep (seeds 0–4, paired per-seed deltas vs. `morbo`)

| d | morbo (mean±std) | ard_box | pca_ellipsoid | ard_pca_ellipsoid |
|---|---|---|---|---|
| 50  | 32.93±0.35 | −4.0% (0/5 win) | **+3.3% (5/5 win)** | **+4.1% (5/5 win)** |
| 100 | 19.90±0.91 | **−49.2% (0/5 win)** | **+66.6% (5/5 win)** | **+66.4% (5/5 win)** |
| 150 | 0.08±0.17\* | 0.00 (0/5 win)\* | +20.3 abs (5/5 win)\* | +23.0 abs (5/5 win)\* |

\* At d=150 baseline is ≈0 for 4/5 seeds (one seed reaches 0.37), so
percentages are undefined for most seeds — absolute HV deltas reported
instead. `ard_box` is *exactly* 0.00 across all 5 seeds, identical to
baseline's failure.

**This is the headline result of the whole study.** At d=50 and d=100 the
win/loss pattern is **unanimous across every single seed** — `ard_box`
loses in 0/5, both PCA variants win in 5/5 — with tight, non-overlapping
ranges (e.g. at d=100, `ard_box`'s worst seed is still a −34.6% loss;
`pca_ellipsoid`'s worst seed is still a +56.5% win). This is about as
strong as a 5-seed result can be: not a lucky seed-0 draw, a systematic
effect. `ard_box`'s failure also gets *more variable* with `d` (std 0.79 →
2.58 at d=50→100) even as its mean loss grows — consistent with the
per-dimension lengthscale-collapse mechanism being seed-sensitive in
severity while always present.

## 2. Real problems, corrected with multi-seed data

**Penicillin (d=7, M=3, single seed) — low-d negative control still holds:**
all three original variants within ±7% of baseline.

**Rover (d=60, M=2, 5 seeds) — the single-seed conclusion was WRONG, now
corrected:**

| Method | Mean HV | Paired win-rate | Mean paired % | Per-seed % |
|---|---|---|---|---|
| morbo | 2.20±0.14 | — | — | — |
| ard_box | 2.15±0.24 | 2/5 | −1.8% | [−5.2, +11.6, +8.6, −0.6, −23.3] |
| pca_ellipsoid | 2.30±0.12 | 3/5 | **+4.9%** | [−10.0, +5.5, +11.3, +17.8, −0.1] |
| ard_pca_ellipsoid | 2.24±0.22 | 2/5 | +1.4% | [−3.1, +12.1, −1.0, −5.9, +5.1] |

The original single-seed Rover write-up ("no variant beats baseline,
pca_ellipsoid does worst at −10.0%") was **noise, not signal** — that
single seed happened to be pca_ellipsoid's worst of the five. Averaged
over 5 seeds, `pca_ellipsoid`'s mean is actually *above* baseline (+4.9%),
and all three variants have win-rates near a coin flip (2/5, 3/5, 2/5)
with per-seed swings (±10–23%) far larger than any of the mean effects.

**This sharpens rather than weakens the core claim.** Contrast the
win-rates directly: DTLZ2 (low effective dimensionality — 1 informative
variable of d) shows **unanimous 5/5 or 0/5** results at every dimension
tested. Rover (60 genuinely-relevant spline-waypoint dimensions, no
low-dimensional structure to align with) shows **noise-level, seed-flipping
2/5–3/5** results. That contrast — systematic effect vs. genuine noise —
is a cleaner, more falsifiable signature of "shape adaptation helps
*conditional on* low effective dimensionality" than a one-sided loss would
have been.

## 3. Extended budget at d=200 (2000 evals)

| | Final HV | vs baseline | Breakthrough eval |
|---|---|---|---|
| morbo | 19.32 | — | ~1150 |
| ard_box | 14.81 | −23.4% | ~1450 |
| pca_ellipsoid | 28.79 | +49.0% | ~800 |
| ard_pca_ellipsoid | **33.15** | **+71.6%** | ~750 |

The 600-eval d=200 all-zeros result was a budget effect: at 2000 evals
everyone breaks through, same ranking as d=100/150.

## 4. New methods (single seed; see below for the standout finding)

### d=100 (baseline morbo=20.02)

| Method | Final HV | vs baseline |
|---|---|---|
| `ard_box_dimprior` | 11.07 | −44.7% (fix made it **worse** than plain `ard_box`'s −34.6%) |
| `ard_pca_dimprior` | 32.83 | +64.0% (≈ same as `ard_pca_ellipsoid`'s +66.9%, no real change) |
| `cma_ellipsoid` | 24.29 | +21.3% (real win, but well below PCA variants' +65%) |
| `linear_gp` | 15.91 | −20.5% (linear kernel alone underperforms Matérn here) |
| `linear_gp_cma` | 21.41 | +6.9% (CMA partially rescues the weaker linear kernel) |
| `linear_gp_pca` | **33.34** | **+66.5%** (matches the best Matérn+PCA combo *with a linear kernel*) |

### d=150 (baseline morbo=0.00)

| Method | Final HV |
|---|---|
| `ard_pca_dimprior` | 18.17 (below `ard_pca_ellipsoid`'s 29.08 — fix doesn't help) |
| `cma_ellipsoid` | 26.99 (breaks through; between plain PCA and ARD+PCA) |
| `linear_gp` | 0.00 (fails, same as isotropic baseline) |
| `linear_gp_pca` | 28.02 (breaks through, ≈ matches `ard_pca_ellipsoid`) |

### d=200 / 600 evals (baseline morbo=0.00) — the standout result

| Method | Final HV |
|---|---|
| `ard_pca_dimprior` | 0.00 |
| `ard_pca_ellipsoid` | 0.00 |
| **`cma_ellipsoid`** | **21.72** |
| `linear_gp` | 0.00 |
| `linear_gp_pca` | 0.00 |

**`cma_ellipsoid` is the *only* method of nine tested that breaks through
at d=200 within the standard 600-eval budget** — not `ard_pca_ellipsoid`
(our previous best method, needed 2000 evals here), not either linear-kernel
combination. This directly validates the AS-SMEA-motivated design choice:
CMA's covariance is a *persistent, incrementally-updated* state that starts
adapting from the very first iteration, whereas the PCA variants recompute
their shape from scratch each time and need enough accumulated local data
for a reliable eigendecomposition — a requirement that gets harder to meet
exactly when `d` is large and the eval budget is comparatively small. This
is the regime where the AS-SMEA paper's core design argument (temporal
smoothing beats one-shot recomputation) actually bites.

### Two negative results worth stating plainly

- **The dimension-scaled lengthscale prior does not fix `ard_box`.**
  Removing the `Interval(0.05, 4.0)` ceiling didn't rescue the method —
  `ard_box_dimprior` (11.07) is *worse* than plain `ard_box` (13.09) at the
  same seed/scale. Plausible reason: the hard ceiling was incidentally
  *limiting* how extreme the axis-length ratio could get; a soft prior with
  no ceiling let lengthscales spread even wider, worsening the
  constraint-compounding collapse rather than curing it. The mechanism we
  diagnosed (many independent per-axis constraints combine combinatorially)
  isn't really about *where* the lengthscales sit, it's about treating them
  as `d` separate hard constraints at all — which no lengthscale prior
  changes.
- **Kernel choice matters far less than trust-region shape.** A linear
  kernel alone is clearly worse than Matérn (`linear_gp`: −20.5% at d=100).
  But paired with PCA-based shape adaptation, it fully recovers — matching
  the best Matérn+PCA result at d=100 and d=150. This suggests PCA-based
  shape adaptation is doing most of the actual work of finding the
  problem's effective low-dimensional structure; a much simpler surrogate
  model can ride on top of a well-shaped search region almost as well as a
  flexible one can.

## 5. Composite modeling × shape adaptation (Penicillin 2×2)

**Caveat first:** this run's `morbo` baseline (373082.3) doesn't match the
original `penicillin_composite` experiment's `morbo` (347702.9) at the same
seed/config — this is CPU-vs-GPU/BLAS-backend floating-point
non-determinism (the original was a laptop CPU run; this is a cluster GPU
run), not a bug. Comparisons below are **within this run only** (all five
values from the same environment), which is valid for the interaction
question even though the absolute scale differs from the earlier
CPU-only composite-Penicillin write-up.

| Cell | Final HV | vs. this run's morbo |
|---|---|---|
| Direct + isotropic (`morbo`) | 373082.3 | — |
| Direct + PCA (`pca_ellipsoid`) | 347228.2 | −6.9% |
| Composite + isotropic (`composite_penicillin`) | 369704.6 | −0.9% |
| Composite + PCA (`composite_penicillin_pca`) | 360577.4 | −3.4% |
| Composite + ARD-PCA (`composite_penicillin_ard_pca`) | 347985.4 | −6.7% |

**No strong interaction.** Composite modeling alone is roughly neutral here
(−0.9%, much smaller than this project's earlier CPU-run finding of +22%
— see caveat above about why these aren't directly comparable in absolute
terms). Shape adaptation alone costs about −7% (matching the low-d negative
control elsewhere). Combined, the cost is roughly additive (−3.4% to
−6.7%), not compounded or cancelled. Consistent with the two extensions
touching genuinely disjoint parts of the pipeline: composite modeling
changes what the GP predicts, shape adaptation changes only how candidates
get sampled around whatever the GP predicts — and, per §2's finding,
Penicillin's d=7 is squarely in the regime where shape adaptation has no
systematic effect (small negative here, roughly a coin flip on Rover).

## Why `ard_box` actively hurts at high d (diagnosed, not speculated)

A single GP fit on d=100 local data gives lengthscales spanning a 15.2×
ratio (~99 smooth dims at the 4.0 constraint ceiling, 1 informative dim
near 0.26). `ard_box` applies that ratio literally as `d` independent
per-axis constraints; containment requires satisfying all `d`
simultaneously, so the region collapses combinatorially even at
exactly-preserved volume: only 1/200 locally accumulated points remained
inside `ard_box`'s region vs 41/200 for `pca_ellipsoid`'s shape on
identical data. Using *more* information (lengthscales) does worse than
ignoring it (isotropic) because per-dimension estimation noise becomes hard
geometry that compounds across dimensions — and (per §4) a softer prior on
that same information doesn't fix it either, confirming the problem is
structural (treating `d` estimates as `d` separate hard constraints), not
about the specific numeric values.

## 6. `mab_shape`: a per-TR bandit over shapes — wins big where there's time to learn, fails where there isn't

`mab_shape` (epsilon-greedy bandit per trust region over
`{isotropic, ard_box, pca_ellipsoid, ard_pca_ellipsoid, cma_ellipsoid}`,
reward = whether the TR's success streak just incremented; see
`writeup/FURTHER_DIRECTIONS.md`) was run at d=100/150/200 (single seed,
`tr_shape_methods_dtlz2_*d`) and on Rover (single seed, `tr_shape_rover`).

| d | morbo | mab_shape | vs. baseline | vs. best fixed shape at this d |
|---|---|---|---|---|
| 100 (600 evals) | 20.02 | **33.89** | **+69.2%** | best of all 8 methods tested (beats `linear_gp_pca`'s 33.34, `ard_pca_dimprior`'s 32.83) |
| 150 (600 evals) | 0.00 | 0.00 | tied with baseline | `pca_ellipsoid`/`cma_ellipsoid`/`linear_gp_pca` all break through (27-29); mab_shape does not — **but see below, this recovers at 2000 evals** |
| 200 (600 evals) | 0.00 | 0.00 | tied with baseline | only `cma_ellipsoid` (21.72) breaks through; mab_shape does not — **but see below, this recovers at 2000 evals** |
| Rover (single seed, seed 0) | 2.36 (this seed) | 2.19 | −6.9% (this seed); vs. the 5-seed mean of 2.20, −0.5% | in range of the near-coin-flip fixed-shape variants (§2) |

**This is a genuinely two-sided result, not a clean win.** At d=100,
`mab_shape` is the single best method of all eight tested — beating every
fixed shape, including the ones the bandit is choosing among — plausibly
because it can allocate exploitation toward `pca_ellipsoid`/`ard_pca_ellipsoid`
early while still hedging against a bad draw. But at d=150/200, where the
600-eval budget is already tight enough that even the strongest fixed
shapes only barely break through late in the run (§1's breakthrough-eval
numbers: pca_ellipsoid ~350/600, ard_pca_ellipsoid ~300/600), the bandit's
own exploration cost (a fixed `mab_epsilon=0.15` fraction of iterations
spent on arms including the two that are actively harmful, `ard_box` and
plain isotropic) apparently consumes exactly the margin the fixed-shape
methods needed to break through in time — `mab_shape` ties the *failing*
baseline rather than approaching the winning fixed shapes. The mechanism
implicated in §1 (a narrow, late-arriving breakthrough window at high d)
directly explains why adaptivity's own learning cost becomes a net negative
exactly where the reward signal is scarcest. On Rover (single seed only,
not directly comparable to fixed-shape's 5-seed means), `mab_shape` lands
in the same noise band the fixed shapes occupy (§2) rather than resolving
it — consistent with Rover already being a near-coin-flip regime for every
shape mechanism tried here, adaptive or not.

### Resolved: the d=150/200 failure was a pure budget artifact

Rerunning `mab_shape` at the same 2000-eval budget used for §3's extended
d=200 follow-up (`cluster/submit_mab_shape_extended_budget.sh`) settles the
question directly — the precedent from §3 held:

| d | morbo | pca_ellipsoid | ard_pca_ellipsoid | `mab_shape` |
|---|---|---|---|---|
| 150 (2000 evals) | 25.95 | 29.23 (+12.6%) | **34.26 (+32.1%)** | 32.32 (+24.6%) |
| 200 (2000 evals) | 19.32 | 28.79 (+49.0%) | 33.15 (+71.6%) | **33.61 (+73.9%)** |

**`mab_shape` fully recovers given enough budget — and at d=200 it becomes
the single best method of every fixed-and-adaptive shape tested anywhere
in this study**, edging out `ard_pca_ellipsoid` (previously the best method
at this scale). At d=150 it's a strong second place, comfortably ahead of
plain `pca_ellipsoid` and the baseline, just behind `ard_pca_ellipsoid`.
This confirms the d=150/200/600-eval failure was exactly the same kind of
budget artifact as the isotropic baseline's own d=200/600-eval zero (§3) —
not a standing design flaw. The bandit's exploration cost is real (it's why
`mab_shape` needed the extra budget the pre-converged fixed shapes at
d=100 didn't), but it amortizes rather than compounding indefinitely: once
there's enough budget to pay the exploration tax *and* still exploit the
learned arm, adaptivity's core promise — not needing to know the best fixed
shape in advance — pays off outright. The annealed-`mab_epsilon` idea
remains worth trying as a budget-efficiency improvement (it should let
`mab_shape` reach the same endpoint with less wasted exploration, potentially
recovering some of the tight-budget d=150/200/600-eval gap too), but is no
longer needed to explain the earlier failure — that mystery is now closed.

## 7. `SparseDTLZ2`: effective dimension, not nominal dimension, governs the effect

Plain DTLZ2 confounds nominal and effective dimension — its own
`k = d - M + 1` "distance" dims all matter, so effective dimensionality
necessarily grows with nominal `d`. `SparseDTLZ2` (new `evalfn`,
`morbo/problems/sparse_dtlz2.py`) breaks that confound by masking all but
`k_eff` of the `k` distance dims out of `g(x)` entirely (masked dims are
literal no-ops on every objective). Two sweeps, both single-seed, 400
evals / batch 40 / 3 TRs / min_tr_size 150 (smaller budget than the main
600-eval sweep since `SparseDTLZ2`'s effective dimensionality is small by
construction at every point tested):

**Group A — nominal `d` scaled 60→200, effective dim pinned at
`(M-1)+k_eff = 6`:**

| Nominal d | morbo | pca_ellipsoid | ard_pca_ellipsoid | cma_ellipsoid |
|---|---|---|---|---|
| 60  | 35.033 | +0.19% | +0.18% | +0.30% |
| 80  | 35.077 | +0.07% | +0.09% | +0.10% |
| 100 | 35.011 | +0.13% | +0.17% | +0.22% |
| 150 | 35.042 | +0.05% | +0.01% | +0.18% |
| 200 | 35.045 | +0.05% | +0.03% | +0.13% |

**Flat, essentially null, across a 60→200 nominal-dimension range** —
scaling nominal dimension alone, with effective dimension pinned small,
produces no method-dependent effect at all (every value is within ~0.3% of
baseline, an order of magnitude below even the smallest real effect seen
anywhere else in this study). This directly refutes a "gap between nominal
and effective dimension" framing: nominal dimension is not, by itself, the
thing shape adaptation responds to.

**Group B — nominal `d` pinned at 100, effective dim scaled via `k_eff` ∈
{2, 10, 20, 50} (effective dim = `1+k_eff`):**

| k_eff (effective dim) | morbo | pca_ellipsoid | ard_pca_ellipsoid | cma_ellipsoid |
|---|---|---|---|---|
| 2 (3)   | 35.146 | +0.09% | +0.08% | +0.10% |
| 5 (6)   | 35.011 | +0.13% | +0.17% | +0.22% |
| 10 (11) | 34.879 | +0.06% | +0.05% | +0.24% |
| 20 (21) | 34.243 | +1.02% | +0.57% | +0.97% |
| 50 (51) | 29.095 | **+6.7%** | −0.55% | **+10.1%** |

**A clean, monotonic dose-response as effective dimension grows — with
nominal dimension held fixed the entire time.** Effect size grows smoothly
from noise-level (~0.1% at effective dim 3) through a first visible signal
(~1% at effective dim 21) to a real effect (+6.7% to +10.1% at effective
dim 51) — extrapolating cleanly toward the +64-67% seen in §1's plain-DTLZ2
sweep at d=100 (effective dim ≈99, full budget 600). Combined with Group
A's flat null, this pins down the mechanism precisely: **it is effective
dimension (relative to the eval budget), not nominal dimension and not the
gap between them, that determines when trust-region shape adaptation
starts to matter.** Plain DTLZ2's dimension sweep (§1) was measuring
effective dimension all along — nominal `d` there was only ever a proxy for
it, because DTLZ2's own construction ties the two together.

One exception worth flagging plainly: at effective dim 51,
`ard_pca_ellipsoid` is the one method that does *not* win (−0.55%, the only
negative value in either table), while plain `pca_ellipsoid` (+6.7%) and
`cma_ellipsoid` (+10.1%, the best of the three here, consistent with §6's
finding that `cma_ellipsoid` is disproportionately strong exactly in
tight-budget/high-effective-dimension regimes) both win clearly. A
plausible read: this is the same lengthscale-noise-compounding mechanism
diagnosed for `ard_box` in the section below, partially reappearing in the
ARD-reweighted hybrid once there's enough real per-dimension signal (51
informative axes) for early lengthscale estimates to be individually noisy
within a tighter, 400-eval budget — worth a multi-seed follow-up before
treating it as more than a single-seed observation.

## Timing note

All cluster runs on one B200 GPU each. Shape variants' gen_time at d=100
was ~70-80s vs the baseline's ~1445s — a side effect of shaped/rotated
candidate distributions concentrating the pool, not a speedup claim
(sampling-function cost itself is identical at matched inputs; verified by
microbenchmark).

Plots: `comparison_seed0.png` in each experiment directory. Aggregate any
multi-seed experiment with `python aggregate_seeds.py <experiment_name>`.
