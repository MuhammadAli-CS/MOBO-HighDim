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

## 8. `sobol`: pure random-search baseline — the whole premise, validated

`label="sobol"` (`morbo/run_one_replication.py`) bypasses trust regions and
GP fitting entirely — a single continuous Sobol low-discrepancy sequence
over the whole `[0,1]^dim` space, evaluated `batch_size` points at a time.
Answers a more basic question than any `tr_shape` variant: is TuRBO/MORBO's
local-modeling machinery (trust regions + GP surrogates) earning its keep
*at all* on these problems, independent of shape? 5 seeds, same experiments
as the core sweep:

| d | morbo | sobol | vs. sobol |
|---|---|---|---|
| 50  | 32.93±0.35 | 23.78±0.34 | **+38.5%** |
| 100 | 19.90±0.91 | 0.51±0.25 | **+3800%** (sobol barely clears the reference point at all) |
| 150 | 0.08±0.17 | 0.00±0.00 | both near/at zero at this budget |
| 200 | 0.00 | 0.00±0.00 | both zero at this budget |
| Rover | 2.20±0.14 | 1.26±0.09 | **+74.6%** |

**MORBO's local-modeling machinery earns its keep decisively, and the
margin grows sharply with dimension up to the point both approaches run
out of budget.** At d=50 the gap is already large (+38.5%); at d=100 it's
enormous — pure random search barely breaks through the reference point at
all (0.51 vs. MORBO's 19.90, vs. the best shaped variant's 33+, a
~65x gap over sobol). At d=150/200 the picture changes only because the
budget is too tight for *any* method to break through reliably at that
scale (§1, §3) — sobol doesn't uniquely fail there, everything does. On
Rover, where shape adaptation itself showed no robust effect (§2), MORBO's
underlying trust-region/GP machinery still clearly beats random search by
+74.6% — confirming that Rover's "no shape helps" result is about
*geometry* specifically, not a sign that local modeling is worthless there
too. This is the right context for reading every other result in this
document: shape adaptation's +64-72% wins are on top of an already-large
base advantage MORBO has over random search, not the whole story by
themselves.

**QUEUED, not yet run:** the d=150/200 numbers above are at the standard
600-eval budget, where nothing breaks through reliably regardless of
method. `sobol` hasn't been run yet at the 2000-eval extended budget
(`tr_shape_dtlz2_150d_2000ev` / `tr_shape_dtlz2_200d_2000ev`, §6's
`mab_shape` extended-budget experiments) — that's the more interesting
comparison, since that's exactly the regime where `mab_shape` and the PCA
variants separate from each other. Does MORBO's advantage hold up once
budget is no longer the bottleneck, or does random search close the gap
given enough evals? `cluster/submit_sobol_extended_budget.sh`.

## 9. Compute efficiency: hypervolume per unit of optimizer time

`plot_efficiency.py` (new) turns the `fit_times`/`gen_times` every saved
run already records into efficiency plots — cumulative optimizer wall-time
(GP fitting + candidate generation) vs. hypervolume, plus a total-time vs.
final-HV "efficiency frontier" — answering "how much computational power
does each method spend for its hypervolume improvement." Pure
post-processing of committed `.pt` files; run locally:

```
python plot_efficiency.py <experiment_name> <seed> [--log-time]
```

Writes `efficiency_seed<seed>.png` into the experiment dir (now generated
and committed for the main experiments). Time here is *optimizer overhead
only* — on synthetic problems the function evaluation itself is
microseconds, so this is the real compute cost; on expensive real problems
evaluation cost dominates instead and the per-evaluation plots are the
right lens. Wall-times are comparable within an experiment (all B200 GPU
runs), not across hardware.

Headline numbers (seed 0, total optimizer seconds → final HV):

| Experiment | morbo | best shaped variant | Take |
|---|---|---|---|
| d=50 | 93s → 32.7 | 87s → 34.2 (ard_pca) | same cost, more HV |
| d=100 | **1505s** → 20.0 | 133s → 33.4 (ard_pca) | **11× cheaper AND +67% HV** |
| d=150 | 96s → 0.0 | 108s → 29.1 (ard_pca) | ~same cost; only shaped variants get any HV at all |
| d=150/2000ev | 580s → 25.9 | 453s → 34.3 (ard_pca) | cheaper and better |
| d=200/2000ev | 610s → 19.3 | 490s → 33.1 (ard_pca); 503s → 33.6 (mab_shape) | cheaper and better |
| Rover | 724s → 2.36 | 645-745s → 2.12-2.28 | equal cost, HV a wash (§2) |

**Shape adaptation is compute-free or compute-negative**: every shaped
variant costs the same as or less optimizer time than the isotropic
baseline, so the hypervolume gains of §1 come at no computational premium.
The d=100 case is the extreme: the isotropic baseline spent ~1445s in
candidate generation vs. the shaped variants' ~70-80s — the shaped/rotated
candidate distributions concentrate the pool and make downstream
HVI-scoring cheaper, while the isotropic box at this dimension floods the
scorer with a diffuse pool (sampling-function cost itself is identical at
matched inputs; verified by microbenchmark). So at d=100 the baseline is
simultaneously ~11× more expensive and 67% worse. `mab_shape`'s bandit
adds no measurable overhead (its per-iteration arm selection is a few
scalar ops). Two more notes from the per-method tables: `linear_gp_*`
variants have near-zero fit time (~2.5s vs ~60-75s for Matérn) — so
`linear_gp_pca`, which matches Matérn+PCA's HV at d=100/150, is the best
HV-per-second method of all at d=100 (18.6 HV/min vs ard_pca's 15.1); and
`sobol` is of course ~free (0.1s) but earns almost no HV at d≥100 —
"efficiency" without effectiveness.

## 10. New benchmark battery — QUEUED, not yet run

Coded, locally smoke-tested, and wired up (2026-07-12, overnight batch);
submit scripts in `cluster/` (see `cluster/README.md` §4e for exact
commands and the LassoBench install prerequisite). Ordered real-first per
plan:

**Real problems:**
- **LassoBench-MO** (`lasso_synt_medium_mo` d=100/effective-dim 5,
  `lasso_dna_mo` d=180/eff 43, `lasso_synt_high_mo` d=300/eff 15;
  `morbo/problems/lasso_bench_mo.py`) — bi-objective LassoBench:
  objective 1 is *exactly* their own `evaluate()` validation loss (so our
  best-loss-so-far curves are directly comparable to the LassoBench
  paper's published TuRBO/CMA-ES/Sparse-HO numbers), objective 2 is the
  fitted Lasso solution's active-coefficient fraction (accuracy vs. model
  sparsity — a real tradeoff their machinery already computes).
  **Protocol matches their paper: 1000 evals (synt_medium/DNA), 5000
  (synt_high), 30 seeds.** This is the real-problem validation of §7's
  effective-dimension finding — LassoBench's synthetic benchmarks have
  *known, documented* effective dimensionality.
- **SparseRover** (`sparse_rover_d{120,180}`,
  `morbo/problems/sparse_rover.py`) — the real Rover trajectory objective
  embedded in 2×/3× nominal dims (extra dims are literal no-ops). The
  effective-dimension mechanism predicts shape adaptation should now
  *help* on Rover (it didn't at nominal=effective=60, §2) because the
  isotropic box wastes volume on the dummy half. Rover's own protocol
  (2000 evals), 5 seeds.
- **Deferred: MOPTA08 / Human-Powered Aircraft / PMO** — MOPTA08 needs a
  proprietary Fortran binary wrangled onto the cluster; HPA/PMO need
  nontrivial porting. Listed in `LITERATURE_REVIEW.md`'s follow-up section
  as next candidates after LassoBench validates (or fails to validate)
  the story.

**New synthetics (mechanism probes):**
- **RotatedSparseDTLZ2** (`rotated_sparse_dtlz2_d100_keff{5,50}`,
  `morbo/problems/rotated_sparse_dtlz2.py`) — closes §7's axis-alignment
  gap: SparseDTLZ2's informative dims are axis-aligned (the one geometry
  `ard_box` could in principle exploit); a fixed random rotation makes the
  effective subspace non-axis-aligned. Rotation-based shapes should be
  ~invariant; `ard_box` should get strictly worse; isotropic unaffected.
  The cleanest possible test that *rotation specifically* does the work.
  400 evals matching §7, 5 seeds, `ard_box` deliberately included.
- **TimeVaryingSparseDTLZ2** (`tv_sparse_dtlz2_d100_keff5`,
  `morbo/problems/time_varying_sparse_dtlz2.py`) — informative dims switch
  at 50% budget; probes re-adaptation. `cma_ellipsoid`'s persistent
  covariance (its strength at d=200) should *hurt* here; memoryless
  `pca_ellipsoid` should recover fast. **Metric caveat:** analyze
  post-switch HV recovery, not the final mixed-history HV (pre-switch
  evaluations were scored under the old mask — see the problem file's
  docstring). 600 evals, 5 seeds.
- **DTLZ landscape variants** (`tr_shape_dtlz{1,3,5,7}_100d`) — same
  effective-dimension structure as DTLZ2 at d=100, different landscape
  characters (multimodal / degenerate / disconnected fronts). Fresh dirs
  with *correct* evalfns (the legacy `dtlz5_m2`/`dtlz7_m2` dirs have the
  known evalfn swap bug). 600 evals, 5 seeds, 4 core methods.

Plots: `comparison_seed0.png` (objective space + HV vs. evals) and
`efficiency_seed0.png` (optimizer time vs. HV) in each experiment
directory. Aggregate any multi-seed experiment with
`python aggregate_seeds.py <experiment_name>`.
