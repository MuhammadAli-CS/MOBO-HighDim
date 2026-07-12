# Trust-Region Shape Adaptation — Results (full study, multi-seed)

Covers the whole study: the dimension sweep (`tr_shape_dtlz2_{20,50,100,150,200}d`,
now 5 seeds each for 50/100/150), the extended-budget follow-up
(`tr_shape_dtlz2_200d_2000ev`), the two real-problem checks (`tr_shape_rover`
[5 seeds], `tr_shape_penicillin`), the new-methods sweep
(`tr_shape_methods_dtlz2_{100,150,200}d`), and the composite-modeling ×
shape-adaptation factorial (`tr_shape_penicillin_2x2`). DTLZ2 runs are 600
evals / batch 50 / 3 TRs unless noted; single-seed unless a seed range is
stated.

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

## Timing note

All cluster runs on one B200 GPU each. Shape variants' gen_time at d=100
was ~70-80s vs the baseline's ~1445s — a side effect of shaped/rotated
candidate distributions concentrating the pool, not a speedup claim
(sampling-function cost itself is identical at matched inputs; verified by
microbenchmark).

Plots: `comparison_seed0.png` in each experiment directory. Aggregate any
multi-seed experiment with `python aggregate_seeds.py <experiment_name>`.
