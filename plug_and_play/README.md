# Plug-and-play: trust-region shape methods x benchmarks

A minimal, modular interface to this project's trust-region
shape-adaptation methods and benchmarks -- exposing the REAL engine
(this repo's own validated, multi-region MORBO implementation), not a
simplified reimplementation of it. "Minimal" here means minimal *wrapper*
code around the real thing, not a smaller/different algorithm: results
should match `experiments/tr_shape_dtlz2_100d`'s recorded numbers up to
ordinary floating-point/hardware nondeterminism (see
`verify_reproduction.py`).

| File | What it is |
|---|---|
| [`methods.py`](methods.py) | Every trust-region shape-adaptation method (`isotropic`, `ard_box`, `pca_ellipsoid`, `ard_pca_ellipsoid`, `cma_ellipsoid`, `labcat_style`) as pure functions, plus `MABShapeBandit` (a small stateful class) for the `mab_shape` meta-strategy. Zero dependency on the rest of this repo -- import it into any project. These are the SAME functions `morbo/trust_region.py`'s `TurboHParams(tr_shape=...)` dispatches to internally, not a copy. |
| [`benchmarks.py`](benchmarks.py) + [`problems/`](problems) | Every benchmark used in this study (DTLZ1/2/3/5/7, `composite_dtlz2`, `sparse_dtlz2`, `rotated_sparse_dtlz2`, `time_varying_sparse_dtlz2`, `rover`, `sparse_rover`, `bbob_biobj`, `lasso_bench_mo`, `penicillin`, `vehicle_safety`, `welded_beam`) behind one `get_benchmark(name, dim=..., **kwargs)` call, all normalized to the same `[0,1]^d` input / maximize-every-objective convention. Project-specific problem implementations live in `problems/` (self-contained copies, not imports across the repo boundary) -- these two files have zero dependency on the rest of this repo. |
| [`run.py`](run.py) | Thin driver: maps a `(benchmark, method)` pair onto this repo's actual, validated MORBO engine (`morbo/run_one_replication.py`) and runs one full BO replication. This is the one file that depends on the top-level `morbo` package -- necessary, not a shortcut avoided, since it's what makes results match the rest of this repo. |
| [`run_and_save.py`](run_and_save.py) / [`plot_study.py`](plot_study.py) | CLI runner + multi-seed aggregate plotting, for reproducing a full study (e.g. the `tr_shape_dtlz2_100d` comparison) via `plug_and_play`'s naming instead of `run_comparison.py`'s. See `cluster/submit_plug_and_play_dtlz2_100d_study.sh`. |
| [`verify_reproduction.py`](verify_reproduction.py) | Runs every method on DTLZ2 (d=100, 600 evals, seed 0 -- the exact `tr_shape_dtlz2_100d` config) and checks the result against the recorded `.pt` value for that label, within tolerance. The actual "does this reproduce the real results" check. |
| [`smoke_test.py`](smoke_test.py) | Fast local sanity checks at toy scale -- every benchmark evaluates, every method runs to completion through the real engine. Correctness-of-wiring only, not numerical reproduction (see `verify_reproduction.py` for that). |

Every function has a full docstring explaining its mechanism -- read
`methods.py`'s module docstring first for the shared representation all
the shape methods return, then each function's own docstring for its
specific math.

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

Both are usable with zero knowledge of the rest of this repo (including
`run.py` and the top-level `morbo` package) -- useful if you want the
shape methods or benchmark implementations in a different BO codebase
entirely:

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
(same Pareto front, same optimum) -- see `problems/composite_dtlz2_general.py`.
**Only runs through `run.py` at `num_objectives=2`**: this repo's own
`morbo/problems/composite_dtlz2.py` (what `run.py` actually calls for
`evalfn="CompositeDTLZ2"`) hardcodes `M=2` -- its reduction formula isn't
implemented for other `M`. Requesting a different `M` through `run.py`
surfaces `morbo`'s own `ValueError` rather than silently working around
it; use `raw_eval_fn`/`composite_reduction` directly at other `M` with
your own composite-aware modeling code instead.

## Adding a new benchmark or method

- **Benchmark**: write a `_make_<name>(dim, **kwargs) -> Benchmark` factory
  in `benchmarks.py` (see any existing one for the pattern), add it to the
  `BENCHMARKS` dict at the bottom. If it needs project-specific objective
  math, put that in a new file under `problems/` (self-contained --
  only `torch`/`numpy`, no imports from outside this folder). If it needs
  to run through `run.py` too, add a `name -> evalfn` entry to `run.py`'s
  `_EVALFN_MAP` (only needed if `morbo/run_one_replication.py` doesn't
  already know that `evalfn`).
- **Method**: write a `compute_<name>_shape(...) -> (R, axis_lengths)`
  function in `methods.py` returning the same `(R, axis_lengths)`
  convention as the others (see `methods.py`'s module docstring), add it
  to `SHAPE_METHODS`. To run it end-to-end through `run.py`, it also needs
  a `tr_shape` mode registered in `morbo/trust_region.py`'s
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
