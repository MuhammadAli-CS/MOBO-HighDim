# Plug-and-play: trust-region shape methods x benchmarks

A minimal, modular interface to this project's trust-region
shape-adaptation methods and benchmarks -- a REGISTRY over the real
engine (this repo's own validated, multi-region MORBO implementation),
not a reimplementation of it. "Minimal" means minimal *wrapper* code:
every file here imports the real logic from `morbo/` directly rather than
copying it, so there's nothing that could silently drift from the
original (see the "Why imports, not copies" note below -- that's not a
hypothetical concern, it already caught a real bug once). Results should
match `experiments/tr_shape_dtlz2_100d`'s recorded numbers up to ordinary
floating-point/hardware nondeterminism (see `verify_reproduction.py`).

| File | What it is |
|---|---|
| [`methods.py`](methods.py) | Every trust-region shape-adaptation method (`isotropic`, `ard_box`, `pca_ellipsoid`, `ard_pca_ellipsoid`, `cma_ellipsoid`, `labcat_style`) as a `SHAPE_METHODS` registry, plus `MABShapeBandit` for the `mab_shape` meta-strategy. Every `compute_*_shape` function is imported directly from `morbo/utils.py` -- the exact code `morbo/trust_region.py`'s `TurboHParams(tr_shape=...)` dispatches to internally. |
| [`benchmarks.py`](benchmarks.py) | Every benchmark used in this study (DTLZ1/2/3/5/7, `composite_dtlz2`, `sparse_dtlz2`, `rotated_sparse_dtlz2`, `time_varying_sparse_dtlz2`, `rover`, `sparse_rover`, `bbob_biobj`, `lasso_bench_mo`, `penicillin`, `vehicle_safety`, `welded_beam`) behind one `get_benchmark(name, dim=..., **kwargs)` call, normalized to a uniform `[0,1]^d` input / maximize-every-objective convention. Every project-specific benchmark imports directly from `morbo/problems/*.py`. |
| [`run.py`](run.py) | Thin driver: maps a `(benchmark, method)` pair onto `morbo/run_one_replication.py` and runs one full BO replication through the real engine. |
| [`run_and_save.py`](run_and_save.py) / [`plot_study.py`](plot_study.py) | CLI runner + multi-seed aggregate plotting, for reproducing a full study (e.g. the `tr_shape_dtlz2_100d` comparison) via `plug_and_play`'s naming instead of `run_comparison.py`'s. See `cluster/submit_plug_and_play_dtlz2_100d_study.sh`. |
| [`verify_reproduction.py`](verify_reproduction.py) | Runs every method on DTLZ2 (d=100, 600 evals, seed 0 -- the exact `tr_shape_dtlz2_100d` config) and checks the result against the recorded `.pt` value for that label, within tolerance. The actual "does this reproduce the real results" check. |
| [`smoke_test.py`](smoke_test.py) | Fast local sanity checks at toy scale -- every benchmark evaluates, every method runs to completion through the real engine. Correctness-of-wiring only, not numerical reproduction (see `verify_reproduction.py` for that). |

Every function has a full docstring explaining its mechanism -- read
`methods.py`'s module docstring first for the shared representation all
the shape methods return, then each function's own docstring (in
`morbo/utils.py`, where the real code lives) for its specific math.

## Why imports, not copies

Every file in this folder was, at one point, a self-contained copy of the
relevant `morbo/` code instead of an import of it. That version had a
real, non-cosmetic bug: `methods.py`'s copy of `compute_cma_ellipsoid_shape`
omitted CMA-ES's sigma/trust-region-length normalization on both the
evolution-path and elite-covariance updates, and used the wrong formula
in one branch. It never affected any recorded result (nothing in
`run.py` ever called it -- `run.py` always called the real engine), but
it did mean the copy quietly diverged from what it claimed to reproduce.
Importing instead of copying makes that entire class of bug structurally
impossible: there is no second copy of the math left to drift.

## Quickstart

```bash
cd plug_and_play
python run.py --benchmark dtlz2 --method pca_ellipsoid --dim 100 --seed 0 --max-evals 600
```

```python
from run import run

result = run(benchmark="dtlz2", method="pca_ellipsoid", dim=100, seed=0, max_evals=600)
print(result["true_hv"][-1])  # final hypervolume
```

## Using `methods.py` / `benchmarks.py` standalone

Both are usable without touching `run.py` -- useful if you want the shape
methods or benchmark implementations plugged into your own optimization
loop instead of this repo's `morbo` engine (they still import from
`morbo/utils.py`/`morbo/problems/*.py`, since that's where the real code
lives, but neither needs the rest of `morbo`'s stateful machinery --
`TrustRegion`, `TRBOState`, GP fitting, etc. -- to be usable):

```python
import torch
from methods import SHAPE_METHODS
from benchmarks import get_benchmark

bench = get_benchmark("dtlz2", dim=100, num_objectives=2)
X = torch.rand(50, bench.dim, dtype=torch.double)
Y = bench.eval_fn(X)

R, axis_lengths = SHAPE_METHODS["pca_ellipsoid"](
    X=X, X_center=X.mean(dim=0, keepdim=True), length=torch.tensor(0.2), dim=bench.dim,
)
```

### Composite benchmarks (`composite_dtlz2`)

One benchmark, `composite_dtlz2` (defaults: `dim=6`, `num_objectives=5`),
is COMPOSITE: it exposes a `Benchmark.raw_eval_fn` (the raw intermediate
response DTLZ2's formula is built from -- richer than the final
objectives, with a known deterministic reduction to them) alongside the
usual `eval_fn`, which is just `composite_reduction(raw_eval_fn(X))`:

```python
bench = get_benchmark("composite_dtlz2")  # dim=6, num_objectives=5
Y = bench.eval_fn(X)              # n x 5, final objectives -- use like any other benchmark
Y_raw = bench.raw_eval_fn(X)      # n x 9, the raw response a composite GP would model
```

Verified numerically identical to direct `dtlz2` at the same `dim`/`num_objectives`
(same Pareto front, same optimum) -- see `morbo/problems/composite_dtlz2_general.py`,
a genuinely new file (not extracted from an existing one) generalizing
`morbo/problems/composite_dtlz2.py`'s hardcoded `M=2` case to any `M`.
**Only runs through `run.py` at `num_objectives=2`**: `morbo/problems/composite_dtlz2.py`
(what `run.py` actually calls for `evalfn="CompositeDTLZ2"`) hardcodes
`M=2` -- its reduction formula isn't implemented for other `M`.
Requesting a different `M` through `run.py` surfaces `morbo`'s own
`ValueError` rather than silently working around it; use
`raw_eval_fn`/`composite_reduction` directly at other `M` with your own
composite-aware modeling code instead.

## Adding a new benchmark or method

- **Benchmark**: add the real implementation to `morbo/problems/<name>.py`
  (or use a BoTorch built-in), then write a `_make_<name>(dim, **kwargs)
  -> Benchmark` factory in `benchmarks.py` that imports and wraps it (see
  any existing one for the pattern), and add it to the `BENCHMARKS` dict
  at the bottom. If it needs to run through `run.py` too, add a `name ->
  evalfn` entry to `run.py`'s `_EVALFN_MAP` (only needed if
  `morbo/run_one_replication.py` doesn't already know that `evalfn`).
- **Method**: add the real implementation to `morbo/utils.py` as a
  `compute_<name>_shape(...) -> (R, axis_lengths)` function (same
  convention as the others), then import and register it in `methods.py`'s
  `SHAPE_METHODS`. To run it end-to-end through `run.py`, it also needs a
  `tr_shape` mode registered in `morbo/trust_region.py`'s
  `TurboHParams`/`TrustRegion._compute_shape_for_mode` -- see that file
  for the existing methods' wiring as a template. This step is
  unavoidable, not a gap in this folder: `run.py` deliberately reuses the
  real engine rather than a second, parallel dispatch mechanism.

## Running the checks

```bash
cd plug_and_play
python smoke_test.py            # fast, toy scale, wiring only
python verify_reproduction.py   # full scale (d=100, 600 evals x 7 methods), checks against recorded results
```
