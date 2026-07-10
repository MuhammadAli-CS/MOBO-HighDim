# Trust-Region Shape Adaptation — Results (full sweep)

Covers the whole study: this directory (d=100, the headline), the dimension
sweep (`tr_shape_dtlz2_{20,50,150,200}d`), the extended-budget follow-up
(`tr_shape_dtlz2_200d_2000ev`), and the two real-problem checks
(`tr_shape_rover`, `tr_shape_penicillin`). All single-seed (seed 0), all
DTLZ2 runs at 600 evals / batch 50 / 3 TRs unless noted.

Three `tr_shape` variants vs. the isotropic-hypercube baseline (`morbo`):
- `ard_box` — axis-aligned box rescaled per-dim by the TR's own fitted GP
  ARD lengthscales (the original TuRBO paper's technique, absent from this
  codebase until now).
- `pca_ellipsoid` — box rotated into the PCA frame of the TR's local data
  (L∞ in the rotated frame — a true L2 ellipsoid's rejection sampling
  underflows at d=100).
- `ard_pca_ellipsoid` — PCA rotation + per-axis widths reweighted by
  lengthscales projected onto each principal axis.

GP fitting is identical across all four; only candidate sampling and
containment testing change. See `writeup/methods.tex` sec:tr-shape for
the math.

## DTLZ2 dimension sweep (fixed 600-eval budget)

| d | morbo | ard_box | pca_ellipsoid | ard_pca_ellipsoid |
|---|---|---|---|---|
| 20 | 34.91 | +0.1% | +0.0% | +0.0% |
| 50 | 32.65 | −2.9% | +3.4% | +4.6% |
| 100 | 20.02 | **−34.6%** | **+64.5%** | **+66.9%** |
| 150 | 0.00* | 0.00* | 20.87† | 29.08† |
| 200 | 0.00* | 0.00* | 0.00* | 0.00* |

\* Genuine zeros: 600 real evals, real non-trivial Pareto fronts (4–23
points), none dominating the reference point within budget.
† Absolute HV (no ratio defined when baseline is 0). Both PCA variants
start at 0 and break through late (~evals 300–400), then climb.

## Extended budget at d=200 (2000 evals)

| | Final HV | vs baseline | Breakthrough eval |
|---|---|---|---|
| morbo | 19.32 | — | ~1150 |
| ard_box | 14.81 | −23.4% | ~1450 |
| pca_ellipsoid | 28.79 | +49.0% | ~800 |
| ard_pca_ellipsoid | **33.15** | **+71.6%** | ~750 |

The 600-eval d=200 all-zeros row was purely a budget effect: at 2000 evals
everyone breaks through eventually, in exactly the same method ranking as
d=100/150. `ard_pca_ellipsoid` at d=200 nearly matches what the baseline
achieves at d=20 (33.15 vs 34.91).

## Real problems

**Penicillin (d=7, M=3, 200 evals) — low-d negative control:** all three
variants within ±7% of baseline (ard_box +6.3%, pca −6.9%, ard_pca +6.5%)
— noise-level, as predicted for low d.

**Rover (d=60, M=2, 2000 evals) — the honest wrinkle:** *no* variant beats
baseline (morbo 2.356; ard_box −5.2%, pca_ellipsoid −10.0%,
ard_pca_ellipsoid −3.1%). pca_ellipsoid — the DTLZ2 star — does worst,
and finds only 17 Pareto points vs baseline's 44.

## Interpretation: it's about effective dimensionality, not nominal d

DTLZ2's structure is extremely low-dimensional in effect: one informative
position variable, d−1 "distance" variables entering only through a smooth
sum. That is exactly the structure a PCA rotation can find and exploit, and
the advantage grows with nominal d because the isotropic baseline wastes
proportionally more of its search volume. Rover's 60 spline-waypoint
dimensions all matter roughly equally — there is no low-dimensional
subspace to find, so the rotation just adds covariance-estimation noise
(60×60 covariance from limited local data) and narrows front coverage.

Refined claim: **shape adaptation helps when (and because) the problem has
low-dimensional effective structure for the geometry to align with, with
the benefit growing in nominal d; on problems where all dimensions matter
it is mildly harmful at any scale.**

## Why ard_box actively hurts at high d (diagnosed, not speculated)

A single GP fit on d=100 local data gives lengthscales spanning a 15.2x
ratio (~99 smooth dims at the 4.0 constraint ceiling, 1 informative dim
near 0.26). ard_box applies that ratio literally as d independent per-axis
constraints; containment requires satisfying all d simultaneously, so the
region collapses combinatorially even at exactly-preserved volume: only
1/200 locally accumulated points remained inside ard_box's region vs
41/200 for pca_ellipsoid's shape on identical data. Using *more*
information (lengthscales) does worse than ignoring it (isotropic)
because per-dimension estimation noise becomes hard geometry that
compounds across dimensions. The same collapse crashed Rover's
exact-interpolation spline evaluator outright (one axis at 0.019 vs ~0.85
elsewhere → near-duplicate spline control points → scipy "Invalid
inputs.") — fixed by penalizing rather than crashing
(`morbo/problems/rover.py`).

## Timing note

All cluster runs on one B200 GPU each. The three shape variants' gen_time
at d=100 was ~70-80s vs the baseline's ~1445s — a side effect of the
shaped/rotated candidate distributions concentrating the candidate pool,
not a speedup claim (sampling-function cost itself is identical at matched
inputs; verified by microbenchmark).

Plots: `comparison_seed0.png` in each experiment directory.
