# Correlation Ablation: Does Cross-Dimension Correlation Modeling Matter? — Results

DTLZ2 (M=2), reparametrized as `CompositeDTLZ2Curve`: an 8-point discretized
curve `h(theta;x) = (1+g(x))*cos(theta - x_0*pi/2)` sampled at `K=8` points
(`morbo/problems/composite_dtlz2_curve.py`), verified numerically equivalent
to plain DTLZ2 — the curve's two endpoints (`theta=0`, `theta=pi/2`) are
exactly DTLZ2's two objectives. Unlike `composite_dtlz2.py`'s `(g, cos, sin)`
raw response, adjacent points on this curve are genuinely correlated by
construction, the kind of structure a joint (Kronecker) model could
plausibly exploit and an independent-per-dimension model cannot.

d=20, 200 evals (25 shared Sobol init + 175 BO evals — `n_evals` checkpoints
land at 205 due to batch size 15), batch size 15, 3 trust regions, seed 0.
Config: `config.json`.

Three-way comparison, same trust-region machinery throughout:
- `morbo` — direct modeling, plain `DTLZ2` evalfn (not the curve reparametrization).
- `independent_gp_composite` — `K=8` decoupled single-task GPs on the raw curve.
- `kronecker_gp_composite` — one `KroneckerMultiTaskGP` jointly modeling
  correlation across all `K=8` raw dimensions.

## Result

| | Final true HV | Total fit time | Total candidate-gen time | Wall clock |
|---|---|---|---|---|
| `morbo` (direct) | 34.216 | 7.4s | 444.0s | 7.5 min |
| `independent_gp_composite` | **34.307** | 22.7s | 1818.9s | 30.7 min |
| `kronecker_gp_composite` | 34.271 | 2255.4s | 424.7s | 45.5 min |

All three land within **0.3% HV of each other** — composite modeling (either
variant) edges out direct modeling by a small margin, consistent with
`fig2_dtlz2_100d`'s finding that composite and direct modeling are close on
DTLZ2-family problems. But **the joint-correlation Kronecker model does not
beat the independent-per-dimension model** — `independent_gp_composite`
actually finishes marginally ahead of `kronecker_gp_composite` (34.307 vs.
34.271), despite the curve's raw response being genuinely correlated by
construction.

## The real story is compute cost, not accuracy

`kronecker_gp_composite`'s **total model-fit time is 2255 seconds — ~100x
`independent_gp_composite`'s 22.7 seconds** — for a slightly *worse* final
HV. This is the direct, data-backed explanation for why the Kronecker run
was the slowest and most resource-intensive of the five experiments last
night (the one that originally got killed running in parallel with two
other jobs). `KroneckerMultiTaskGP`'s joint covariance over `K=8` raw
dimensions scales far worse than 8 independent single-task GP fits, and
here that cost bought nothing: whatever cross-dimension correlation exists
in this curve either isn't being exploited effectively by the joint model,
or isn't worth exploiting at this data scale (205 evals, 3 trust regions
each fitting their own local GP on a handful of points at a time — likely
too little per-TR data for the extra correlation parameters to pay off).

Note the *inverted* time profile between the two composite variants:
`independent_gp_composite` spends almost all its extra time in
candidate-generation/scoring (1818.9s) with cheap fitting (22.7s), while
`kronecker_gp_composite` is the reverse (2255.4s fit, 424.7s gen) — the
Kronecker model's cost is concentrated in the `fit_gpytorch_mll_torch`
Adam-based fit (`morbo/utils.py::get_fitted_kronecker_model`, required
because the default scipy-based fit fails on this model's task-covariance
prior — see README "Correlation ablation" section), not in scoring
candidates once fit.

## No restarts fired

Consistent with `fig2_dtlz2_100d`'s finding at a similar eval budget: `zero`
trust-region restarts across all three runs (`tr_restarts` all `[[], [],
[]]`), so none of the HV differences here are attributable to
restart-timing luck between runs.

## Caveats

- Single seed, one problem instance — this is a controlled A/B on *this*
  correlated raw response at `K=8`, `d=20`, not a general claim that
  correlation-aware GPs never help composite MORBO.
- Maddox, Feng & Balandat (NeurIPS 2021 ML4PS workshop) found their HOGP
  composite variant *wins early but converges to similar final HV* to
  non-composite MORBO on a 177-dim optical design problem — a different
  regime (much higher `d`, image-structured raw response, different
  correlation-modeling architecture) from this synthetic curve ablation.
  This result doesn't contradict theirs; it's a different question (does a
  *Kronecker* multi-task GP specifically help *this* curve-structured raw
  response, at this data scale) with its own answer: no, and it costs ~100x
  more compute to find that out.
- Thread count differs from the other four experiments run this session:
  this run predates the `penicillin_composite`-only thread cap added later
  (see `run_comparison.py`), so its wall-clock numbers reflect full
  available-thread usage — the fit/gen time *ratios* between labels are
  still directly comparable since all three ran under identical conditions.

Plot: `comparison_seed0.png`.
