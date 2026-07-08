# MORBO vs. Scalarized-TuRBO on DTLZ2 (d=100) — Results

Reproduction of Daulton et al. (UAI 2022), Figure 2: DTLZ2, d=100, M=2, 600 evals
(200 shared Sobol init + 400 BO evals), batch size 50, 3 trust regions, seed 0.
Config: `config.json`. Commands and code: see repo README "Fork notes" section.

## Result

| | Final hypervolume | Pareto set size |
|---|---|---|
| MORBO | 20.02 | 6 |
| TuRBO + Chebyshev scalarizations | 16.92 | 4 |

MORBO wins by ~18% HV, matching the paper's qualitative claim (their Fig. 2 has
no error bars / HV numbers to compare directly — it's an illustrative
comparison, not part of their main benchmark suite).

## What `turbo_scalarized` actually disables

Three MORBO mechanisms at once, matching the paper's own "naive multi-objective
TuRBO extension" (not an isolated single-variable ablation):
- `hypervolume=False` — each TR optimizes one random Chebyshev scalarization,
  fixed for the TR's lifetime.
- `track_history=False` — no shared GP training data across TRs; each TR only
  sees points it personally collected.
- `restart_hv_scalarizations=False` — no coordinated global restart search.

## Why MORBO wins here — and why it's *not* what I first assumed

My first guess was "data sharing + smarter restarts." Checking the saved
`tr_restarts` field: **zero trust regions restarted in either run**, over the
full 600-eval budget. With `failure_streak = max(dim//3, 10) = 33` and length
needing ~6 halvings to hit `length_min`, a TR needs on the order of 200
consecutive non-improving batches to terminate — more than this short run's
budget provides. So the "resample λ on restart" mechanism the paper credits
for spreading TRs across tradeoffs (§2.3) is inactive here. That rules out
restart coordination as the explanation for *this particular result*.

What's actually different, and it happens every single iteration rather than
only at restart:

1. **TR center reselection is tied to the shared global frontier in MORBO,
   to a fixed private objective in TuRBO.** Every iteration, a MORBO trust
   region's center jumps to whichever pareto point *currently* has the
   highest hypervolume contribution — computed over the pooled Pareto front
   from all three TRs' shared data. If another TR has recently covered the
   region this TR was sitting in, this TR's center gets pulled toward a
   different, currently-underrepresented part of the frontier. A scalarized
   TuRBO TR's center just moves to whichever point (from its own private
   history) scores highest on its one fixed weight vector — there's no pull
   toward "cover what's missing," so it just keeps climbing the same ridge.
   This is visible directly in the scatter plot: each TuRBO trust region's
   points form exactly one continuous diagonal streak (one weight vector,
   one direction, for the whole run), while MORBO's three colors each show up
   in multiple separate clusters across the frontier — the same TR visited
   different tradeoff regions at different iterations.

2. **Batch selection is collaborative in MORBO, independent in TuRBO.**
   MORBO picks each of the 50 points in a batch sequentially to maximize
   hypervolume improvement *jointly with the other 49 already chosen* in that
   batch (candidates pooled across all 3 TRs). This actively diversifies a
   single batch. TuRBO's batch loop instead picks a random TR per slot and
   takes that TR's best-scalarized candidate independently — no mechanism
   ties one slot's choice to another's.

3. **Data sharing (`track_history`) is a supporting/enabling factor, not the
   primary driver.** It's what lets a TR's center-selection "see" the global
   Pareto front and other TRs' data at all — without it, mechanism (1)
   couldn't function even if HV-based selection were otherwise turned on. The
   paper's own separate ablation (their Fig. 4, not reproduced here) shows
   disabling data-sharing alone degrades MORBO but not as sharply as
   disabling the HVI acquisition function itself — consistent with it being
   the infrastructure that mechanisms (1) and (2) run on top of, rather than
   the thing directly responsible for the streak-vs-spread pattern seen here.

## Caveats

- Single seed — the streak pattern and restart count are concrete facts from
  this run, not averaged claims.
- This run cannot cleanly separate "HVI batch selection," "shared-data center
  reselection," and "restart coordination" from each other, since
  `turbo_scalarized` turns off all three simultaneously (as the paper's own
  Figure 2 strawman does). The paper's Figure 4 ablation is the source for
  isolating each one individually — reproducing that here would need
  additional label variants (e.g. `hypervolume=True, track_history=False`) to
  turn off exactly one mechanism at a time.
