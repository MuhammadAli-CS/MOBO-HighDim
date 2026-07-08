# Composite MORBO on a Real Simulator: Penicillin — Results

`CompositePenicillin` forks `botorch.test_functions.multi_objective.Penicillin`'s
~2500-step Euler integrator (5 coupled state variables: penicillin
concentration, culture volume, biomass concentration, glucose concentration,
CO2) to checkpoint the state at `K=5` fixed absolute step indices, giving a
`5K+1 = 26`-dim raw response (`+1` for stopping time) reduced to the same 3
objectives (yield, CO2, time) plain `Penicillin` exposes
(`morbo/problems/composite_penicillin.py`). Verified bit-for-bit against
upstream's own direct-final-state output (`max abs diff = 0.0`).

d=7, 200 evals (20 shared Sobol init + 180 BO evals), batch size 10, 3 trust
regions, seed 0. Config: `config.json`. Reference point `[-1.85, -86.93,
-514.70]` (all three objectives maximized after negation, per botorch
convention).

## Result

| | Final true HV | Total fit time | Total candidate-gen time | Wall clock |
|---|---|---|---|---|
| `morbo` (direct) | 347,702.9 | 14.5s | 810.6s | 13.9 min |
| `composite_penicillin` | **424,234.9** | 134.3s | 13,531.4s | 3h 48min |

Composite modeling wins by **+22.0% final hypervolume** — the largest margin
of any composite-vs-direct comparison run this session (vs. ~+0.3% on
synthetic DTLZ2-family problems in `fig2_dtlz2_100d` and
`correlation_ablation_dtlz2curve`). This is the first result in this repo
where composite modeling produces a *qualitatively* different outcome, not
just a marginal edge — consistent with Penicillin's raw trajectory carrying
real structure (5 correlated state variables evolving over time) that the
final 3-objective reduction throws away, unlike DTLZ2's synthetic `(g, cos,
sin)` decomposition which has no comparably rich intermediate structure to
exploit.

## Where the win comes from: one large, late jump

HV trajectory (`true_hv` at each `n_evals` checkpoint):

| n_evals | `morbo` | `composite_penicillin` |
|---|---|---|
| 30 | 291,931 | 281,476 |
| 90 | 322,022 | 330,654 |
| 150 | 345,798 | 345,534 |
| 160 | 346,757 | 346,113 |
| 170 | 347,453 | 346,140 |
| **180** | 347,453 | **419,788** |
| 190 | 347,453 | 419,980 |
| 200 | 347,703 | 424,235 |

Through `n=170`, `composite_penicillin` is essentially tied with (slightly
behind) `morbo`. Between `n=170` and `n=180` — one batch of 10 evaluations —
its true HV jumps **+21.3%** in a single step, then holds and creeps further.
Checked `tr_restarts`: **zero restarts fired in either run** (`[[], [],
[]]` for both), so this isn't a restart discovering a fresh region — it's
one batch's candidates landing on a genuinely much better point in raw
trajectory space that the reduction maps to a large true-objective
improvement. `tr_sizes` at that point show TR2 and TR3 mid-shrink (0.4, both
had just halved from 0.8 a few checkpoints earlier) rather than freshly
initialized — consistent with an exploitative rather than exploratory step.
Not chasing this further with additional runs here (single seed) — flagging
it as the concrete mechanism behind the headline number rather than a smooth
aggregate improvement.

## Compute cost

Composite modeling costs **~16x the wall-clock time** of direct modeling
here (3h48m vs 14min) — `run_comparison.py` caps this experiment's runs to
8 CPU threads (see thermal note in `run_comparison.py`), so the absolute
wall-clock numbers reflect that cap rather than full-machine throughput, but
this is still the most expensive experiment in the repo by a wide margin:
`fit_time` grows ~9x (26-dim raw response is a heavier `SingleTaskGP` fit
per trust-region iteration than the direct 3-objective model) and
`gen_time` grows ~17x (posterior sampling and reduction `L(Y_raw)` over the
candidate pool, applied every iteration, is doing meaningfully more
arithmetic per candidate at 26 raw dims than 3 final objectives). Unlike the
correlation-ablation Kronecker result, this compute cost bought a real
result — but it's a genuine tradeoff, not a free win.

## Caveats

- Single seed — the +22% figure and the specific jump location are concrete
  facts about this run, not an averaged claim.
- No composite Penicillin benchmark exists elsewhere in the papers reviewed
  for this project (Maddox et al. never touch Penicillin; it only appears
  elsewhere as a direct-modeling benchmark) — there's no published number to
  cross-check this against.
- Thread-capped run (8 of 20 available threads) for thermal reasons unrelated
  to the science — reruns at full thread count would be faster in wall time
  but should reach the same `n_evals`-indexed HV trajectory (thread count
  affects speed, not the sequence of points the optimizer visits, since
  MORBO's candidate generation is CPU-parallelized numerics, not a source of
  run-to-run randomness by itself at fixed seed).

Plot: `comparison_seed0.png`.
