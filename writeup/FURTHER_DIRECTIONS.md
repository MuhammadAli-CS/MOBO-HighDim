# Trust-Region Shape Adaptation — Further Directions

Written after the dimension sweep + extended-budget + Rover results, and
after reading two papers you flagged:
- **Wang et al. 2026, "Surrogate-assisted evolutionary algorithm with
  adaptive local region search" (AS-SMEA)**, Swarm & Evol. Comput.
- **Doumont et al. 2026, "We Still Don't Understand High-Dimensional
  Bayesian Optimization" (linear-bo)**, AISTATS.

Both are directly relevant and, usefully, point in *different* directions —
one validates and extends what we're doing, the other challenges its
premise. Taken together they sharpen what the strongest version of this
project's contribution would be.

---

## Implementation status — results are now in (see `experiments/tr_shape_dtlz2_100d/RESULTS.md` for full numbers)

| Idea (from below) | Label(s) | Status | Headline result |
|---|---|---|---|
| CMA-ES covariance adaptation (§1) | `cma_ellipsoid` | **done** | Only method of 9 that breaks through at d=200/600 evals (HV=21.72 vs 0.00 for everything else) |
| Composite × shape (§1) | `composite_penicillin_pca`, `composite_penicillin_ard_pca` | **done** | Roughly additive, not interacting: composite alone ~-0.9%, shape alone ~-7%, combined ~-3 to -7% |
| Linear-kernel baseline (§2) | `linear_gp`, `linear_gp_pca`, `linear_gp_cma` | **done** | `linear_gp` alone underperforms (-20.5% @ d=100); `linear_gp_pca` matches best Matérn+PCA @ d=100/150, but fails @ d=200 like other PCA methods |
| Dim-scaled lengthscale prior as `ard_box` fix (§3, new) | `ard_box_dimprior`, `ard_pca_dimprior` | **done — negative result** | Does not fix `ard_box` (11.07, worse than plain ard_box's 13.09); `ard_pca_dimprior` ≈ same as `ard_pca_ellipsoid`, no real change |
| Multi-seed robustness | (all core labels, seeds 0–4) | **done** | d=50/100 unanimous 5/5 or 0/5 win-rates; Rover's earlier single-seed "no variant wins" conclusion was noise — corrected to near-50/50 win-rates, contrasting sharply with DTLZ2's unanimity |
| MAB-guided shape selection (§3) | `mab_shape` | **done** | Best of all 8 methods at d=100/600ev (+69.2%); ties the failing baseline at d=150/200/600ev but **fully recovers at 2000 evals, becoming the single best method of every shape tested at d=200** (33.61 vs. ard_pca_ellipsoid's 33.15) — confirmed a pure budget artifact, not a design flaw |
| Effective- vs. nominal-dimension test (§3) | `SparseDTLZ2` (new `evalfn`) | **done** | Nominal `d` alone (60→200, effective dim pinned at 6) produces a flat null; effective dim alone (fixed d=100, k_eff 2→50) produces a clean monotonic dose-response 0.1%→10% — confirms effective dimension, not nominal dimension or "the gap," is the governing variable |

All of the above are written up in full in `experiments/tr_shape_dtlz2_100d/RESULTS.md`
and `writeup/methods.tex` (`sec:tr-shape`). See `writeup/PROJECT_HANDOFF.md`
for a full session-resumption summary of the whole project.

One idea below I added that isn't in either paper: the **dim-scaled
lengthscale prior as a targeted fix for `ard_box`'s collapse** (§3). Our own
diagnosis was that ~99/100 fitted lengthscales pin against the
`Interval(0.05, 4.0)` ceiling at d=100; Hvarfner et al.'s prior both removes
that ceiling and *expects* lengthscales to grow like √d, so it directly
attacks the mechanism we identified. `ard_box_dimprior` tests whether that
rescues the method that otherwise fails worst.

---

## 1. Answering your direct questions

### "Does the ellipsoid have to use PCA?"

**No — and AS-SMEA shows the more principled alternative we should try:
CMA-ES-style covariance adaptation.** Our `pca_ellipsoid` recomputes the
shape *from scratch* every iteration as a one-shot eigendecomposition of
the current local data's second-moment matrix about the TR center. That's
memoryless and reactive. AS-SMEA (their Eq. 2, Sec. 2.1/3.2) instead
maintains an *evolving* covariance `C` updated multiplicatively each
iteration:

```
C^(t) = (1 - c1 - cμ) C^(t-1)               # decay the old shape
        + cμ Σ w_i (x_i - m)(x_i - m)^T      # rank-μ: this iter's spread
        + c1 p p^T                            # rank-1: evolution path (momentum)
```

Three things our PCA shape lacks and CMA has, each a concrete experiment:

- **Temporal smoothing** (the `(1-c1-cμ)C^(t-1)` decay term). Our shape can
  swing wildly between iterations when the local point cloud is small and
  noisy — plausibly part of why `ard_box`/PCA are unstable at low sample
  counts. A decayed running covariance would be far steadier. **New variant:
  `cma_ellipsoid`** — same rotated-box machinery we already built, but `R`,
  `axis_lengths` come from a per-TR persistent `C` buffer updated by the
  formula above instead of a fresh eigendecomposition. Minimal new code:
  add `C` as a registered buffer alongside `R`/`axis_lengths`, replace the
  body of `compute_pca_ellipsoid_shape` with the CMA update, eigendecompose
  `C` for the rotation.
- **The evolution path `p` (rank-1 / momentum term).** Encodes *where the
  TR center has been moving*, not just where its data currently sits — so
  the region elongates along the direction of progress. This is exactly the
  "align with the informative direction" behavior that made `pca_ellipsoid`
  win on DTLZ2, but with directionality PCA can't see. Strong candidate for
  the single highest-value follow-up.
- **Learning rates `c1, cμ` decoupled from TR size.** Our shape is fully
  slaved to the isotropic `length` (geometric-mean-normalized to it). CMA
  separates step size `σ` from shape `C`, which may let shape adapt faster
  than the conservative success/failure `length` schedule allows.

### "Can we do composite modeling with this?"

**Yes, and it composes with zero new algorithmic code** — verified
end-to-end (`composite_penicillin_pca` / `composite_penicillin_ard_pca`
labels added to `run_comparison.py`, smoke-tested). Why it just works:
`tr_shape` only changes candidate *sampling and containment* in design
space, while composite modeling only changes *what the GP models* (a raw
response `Y_raw`) and *how objectives are reconstructed* (the reduction
`L`). They touch disjoint parts of the pipeline. The one seam —
`extract_ard_lengthscale` for the ARD-based variants — already
geometric-means across however many outputs the local `ModelListGP` has,
so it transparently handles composite's `K` raw dimensions the same way it
handles direct modeling's `M` objectives.

**The genuinely interesting experiment this enables:** composite modeling
changes the *effective dimensionality of what the surrogate sees*, and our
whole tr_shape finding is that shape adaptation's benefit is governed by
effective (not nominal) dimensionality. So: does composite modeling —
which we showed wins big on Penicillin precisely because the raw trajectory
has exploitable structure — *change how much shape adaptation helps*? Two
plausible, opposite hypotheses, both publishable:
  1. Composite modeling *already* captures the low-dim structure, leaving
     less for shape adaptation to exploit → they're redundant, ~additive at
     best.
  2. Composite modeling produces *better-conditioned local models* whose
     lengthscales/covariances are more reliable → shape adaptation gets a
     cleaner signal and helps *more*.
  Running `morbo` / `pca_ellipsoid` / `composite` / `composite+pca` as a
  2×2 on Penicillin (and a composite-DTLZ2 at high d) directly answers this.

---

## 2. The challenge from the linear-bo paper (read this one carefully)

Doumont et al. is a "sit up straight" result for this whole line of work.
Their claim: on 60–6000 dim problems, **Bayesian linear regression** (a GP
with a *linear* kernel), after one geometric fix (bijectively mapping the
`[0,1]^d` cube onto a `d+1`-sphere to kill boundary-seeking), **matches or
beats** TuRBO, SAASBO, and the other structure-exploiting HDBO methods —
including the locality-based methods our trust-region work descends from.

Why this matters for us, honestly:
- Their headline mechanism (Hvarfner et al. 2024, which they build on) is
  that in the `N ≈ D` regime you *can't even fit a first-order Taylor
  expansion*, so the winning move is maximal smoothness / simplicity, not
  cleverer structure. Our `d=200, N=200` runs are squarely in this
  regime — and it's exactly where we saw the isotropic baseline find
  *nothing* in 600 evals.
- This reframes our own positive result. `pca_ellipsoid` winning at high d
  might be a *different route to the same underlying fix*: by concentrating
  search along a few data-supported directions, it's implicitly imposing a
  low-effective-dimension prior, much as the linear kernel imposes maximal
  smoothness. The Rover result (no benefit when all dims matter) is
  consistent with this reading — when there's no low-dim structure to lean
  on, neither the rotation *nor* (per their paper) heavy smoothness helps
  beyond a well-tuned simple baseline.

**Two concrete things to do about it:**
  1. **Add their baseline.** A spherically-projected linear-kernel GP is a
     cheap, strong, and currently-missing comparison point. If our
     shape-adapted MORBO can't beat a linear kernel at d=200, that's a
     crucial (and honest) thing to know before claiming shape adaptation is
     the answer to high-d MOBO. Their code is public
     (github.com/colmont/linear-bo) — mostly a kernel swap + input warp.
  2. **Test the composition, not just the competition.** They study linear
     *global* models; we do *local* trust regions. The interesting question
     is whether a linear (or lengthscale-`√D`-scaled) surrogate *inside*
     each trust region, combined with our shape adaptation, beats either
     alone. Their `N ≈ D` argument is about global modeling; locality
     changes the effective `N/D` ratio per region.

---

## 3. Other directions, ranked by value/effort

**High value, low effort:**
- **Multi-seed the whole sweep.** Everything so far is seed 0. The
  headline numbers (+64–72% at high d) are large enough to likely survive,
  but the small-d and Rover effects (±3–10%) are within plausible
  single-seed noise. 5 seeds turns "suggestive" into "publishable."
- **The `cma_ellipsoid` variant** (Sec. 1 above) — highest-value new method,
  and reuses all the rotated-box plumbing we already built.

**High value, medium effort — both DONE, results in:**
- **MAB-guided variant selection (AS-SMEA's LS-IMA/MASS, Sec. 3.3) — DONE.**
  `tr_shape="mab_shape"` (`morbo/trust_region.py`): a per-trust-region
  epsilon-greedy bandit over `{isotropic, ard_box, pca_ellipsoid,
  ard_pca_ellipsoid, cma_ellipsoid}`, rewarded 1.0 whenever the TR's own
  success-streak counter (`n_successes`) was just incremented, folded into
  a per-arm EMA. **Result:** best of all 8 methods at d=100/600ev (+69.2%
  vs. baseline, beating every fixed shape it chooses among); at
  d=150/200/600ev it initially tied the failing isotropic baseline
  (exploration cost eating the narrow late-breakthrough margin), but a
  2000-eval extended-budget rerun confirmed this was a pure budget
  artifact, not a design flaw: `mab_shape` fully recovers and becomes the
  single **best** method of every shape tested at d=200/2000ev (33.61 vs.
  `ard_pca_ellipsoid`'s 33.15), and a strong second at d=150/2000ev (32.32,
  behind `ard_pca_ellipsoid`'s 34.26). Did **not** recover Rover's lost
  ground (landed in the same near-coin-flip band the fixed shapes occupy,
  single seed, standard budget only). See
  `experiments/tr_shape_dtlz2_100d/RESULTS.md` §6 / `writeup/methods.tex`
  for full numbers. An annealed `mab_epsilon` remains worth trying as a
  budget-*efficiency* improvement (reach the same endpoint with less
  wasted exploration, possibly narrowing the tight-budget gap too), but is
  no longer needed to explain the original failure.
- **The linear-kernel baseline + local composition** (Sec. 2) — done, see
  results in `experiments/tr_shape_dtlz2_100d/RESULTS.md` §4.

**Medium value — DONE, and the result corrects our own earlier framing:**
- **Crossover-point characterization + effective-vs-nominal dimension test —
  DONE (combined into one problem).** New `evalfn="SparseDTLZ2"`
  (`morbo/problems/sparse_dtlz2.py`): masks all but `k_eff` of DTLZ2's
  `k = dim - M + 1` distance dimensions out of `g(x)` entirely, so nominal
  and true effective dimension vary independently (plain DTLZ2 confounds
  them, since its `k` necessarily grows with nominal `d`). Two sweeps
  (`cluster/submit_sparse_dtlz2.sh`, 4 core methods each, seed 0):
    - Group A (`sparse_dtlz2_d{60,80,100,150,200}_keff5`): effective dim
      pinned at 6 while nominal `d` scales 60→200. **Result: flat null**
      (every value within 0.3% of baseline) — nominal dimension alone does
      *not* reproduce the effect, however large it gets.
    - Group B (`sparse_dtlz2_d100_keff{2,10,20,50}`): nominal `d` pinned at
      100 while effective dim scales 3→51. **Result: clean monotonic
      dose-response**, 0.1%→10%, extrapolating toward the +64-67% seen at
      full effective dimension.
  **This corrects the "gap between nominal and effective dimension" framing
  used when this idea was first proposed below**: the governing variable is
  effective dimension relative to the eval budget, full stop — not the gap,
  and not nominal dimension. Plain DTLZ2's dimension sweep was measuring
  effective dimension all along; nominal `d` was only ever a proxy for it.
  One single-seed exception worth a multi-seed follow-up:
  `ard_pca_ellipsoid` is the one method that loses (-0.55%) at the highest
  effective dimension tested (51), while `pca_ellipsoid`/`cma_ellipsoid`
  both win clearly there. See `experiments/tr_shape_dtlz2_100d/RESULTS.md`
  §7 for full tables.

**Speculative / bigger:**
- **Learned rotation instead of PCA/CMA.** Both PCA and CMA derive `R` from
  second moments. A supervised alternative: rotate toward directions of
  high *objective* gradient/sensitivity (which the GP already estimates via
  lengthscales), not just high input variance. `ard_pca_ellipsoid` gestures
  at this by reweighting axes with lengthscales, but the *rotation* is still
  variance-driven. A genuinely objective-aware rotation is unexplored.

---

## Status: everything proposed so far is now implemented and has results

Every item originally suggested in this document — the multi-seed sweep,
composite×shape 2×2, `cma_ellipsoid`/linear-kernel/dim-prior fix,
`mab_shape` (including its extended-budget resolution), the
effective-dimension study (`SparseDTLZ2`), and the `sobol` pure
random-search baseline — is coded, smoke-tested, run on the cluster, and
written up in `experiments/tr_shape_dtlz2_100d/RESULTS.md` (§1–8) and
`writeup/methods.tex` (`sec:tr-shape`). `sobol`'s result is decisive:
MORBO beats random search by +38.5% (d=50) up to +3800% (d=100, where
sobol barely clears the reference point) and +74.6% on Rover — the whole
local-modeling premise this line of work builds on is solidly validated,
and shape adaptation's own wins sit on top of that larger base advantage.

**Further ideas not yet coded** (ranked):
- Learned/objective-aware rotation (see Speculative section below) — the
  remaining ranked idea with no code yet.
- Multi-seed the new-methods sweep (cma/linear-kernel/dimprior/mab_shape/
  SparseDTLZ2) — everything past the original 4-method sweep is currently
  single-seed only.

## Where to steer this next (literature-informed, 2026-07-12)

Full findings in `LITERATURE_REVIEW.md`'s "Follow-up review: trust-region
shape adaptation" section — this is the short version. Ranked:

1. **Validate the effective-dimension finding on a real problem.**
   `SparseDTLZ2` (§7 of RESULTS.md) is our strongest result and rests
   entirely on a synthetic construction we built ourselves. LassoBench
   (made bi-objective — its true coefficient sparsity already controls
   effective dimension, and its internal active-coefficient count is a
   free second objective) is the best available bridge to a real problem.
2. **Upgrade `mab_shape` to a contextual bandit.** Use an online estimate
   of effective dimension (top-k PCA eigenvalue mass ratio in the TR's
   local data — `pca_ellipsoid` already computes the eigendecomposition
   this would reuse) as context, replacing epsilon-greedy trial-and-error.
   This turns "the bandit recovers the best of both worlds empirically"
   into "the bandit learns the exact rule we discovered mechanistically" —
   a much stronger result, and cheap since the underlying computation
   already exists.
3. **Add missing baselines/citations.** LABCAT (Visser et al. 2023/24) and
   CMA-BO (Ngo et al., TMLR 2024) are the closest prior art and now have a
   differentiation paragraph in `methods.tex`; still missing a true
   no-trust-region global-BO baseline (Hvarfner et al. 2024's
   dimension-scaled-prior vanilla BO) and an AdaScale-TuRBO (2026) rerun
   of the `ard_box` fix question (their joint lengthscale/TR-size scaling,
   unlike the static prior we tried, might actually rescue it).

Lower priority: Thompson-sampling or Hedge/EXP3 variants of `mab_shape`
(remove the `mab_epsilon` hyperparameter — see GP-Hedge and self-tuning
portfolio-BO in the literature review), and the Human-Powered
Aircraft/PMO/MOPTA08-MO benchmark candidates as further real-world
validation once LassoBench is done.
