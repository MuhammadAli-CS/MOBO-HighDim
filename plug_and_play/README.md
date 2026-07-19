# Plug-and-play: trust-region shape methods x benchmarks

A minimal, self-contained, modular interface to this project's
trust-region shape-adaptation methods and benchmarks. Everything needed
to run an experiment lives inside this folder -- nothing here imports
from this repo's top-level `morbo` package. Only `torch`, `botorch`, and
`gpytorch` are required (plus, only if you use it, the third-party
`LassoBench` package for the `lasso_bench_mo` benchmark).

| File | What it is |
|---|---|
| [`methods.py`](methods.py) | Every trust-region shape-adaptation method (`isotropic`, `ard_box`, `pca_ellipsoid`, `ard_pca_ellipsoid`, `cma_ellipsoid`, `labcat_style`) as pure functions, plus `MABShapeBandit` (a small stateful class) for the `mab_shape` meta-strategy. |
| [`benchmarks.py`](benchmarks.py) + [`problems/`](problems) | Every benchmark used in this study (DTLZ1/2/3/5/7, `composite_dtlz2`, `sparse_dtlz2`, `rotated_sparse_dtlz2`, `time_varying_sparse_dtlz2`, `rover`, `sparse_rover`, `bbob_biobj`, `lasso_bench_mo`, `penicillin`, `vehicle_safety`, `welded_beam`) behind one `get_benchmark(name, dim=..., **kwargs)` call, all normalized to the same `[0,1]^d` input / maximize-every-objective convention. Project-specific problem implementations live in `problems/`, copied into this folder rather than imported across the repo boundary. |
| [`optimizer.py`](optimizer.py) | A minimal, self-contained single-trust-region Bayesian optimization loop, built directly on BoTorch primitives (independent `SingleTaskGP`s, `qLogExpectedHypervolumeImprovement`). Candidates are drawn from a trust region whose shape is recomputed every iteration by one of `methods.py`'s functions -- the one thing this whole folder is actually about. |
| [`run.py`](run.py) | Thin driver tying `benchmarks.py` + `methods.py` + `optimizer.py` together: `run(benchmark=..., method=...)`. |
| [`smoke_test.py`](smoke_test.py) | Fast local sanity checks -- every benchmark evaluates, every method optimizes end-to-end. |

Every function has a full docstring explaining its mechanism -- read
`methods.py`'s module docstring first for the shared representation all
the shape methods return, then each function's own docstring for its
specific math.

## Quickstart

```bash
cd plug_and_play
python run.py --benchmark dtlz2 --method pca_ellipsoid --dim 20 --seed 0 --n-iter 20
```

```python
from run import run

result = run(benchmark="dtlz2", method="pca_ellipsoid", dim=20, seed=0, n_iter=20)
print(result["final_hypervolume"])
```

## Design note: this is NOT the same engine as the rest of the repo

`optimizer.py` is a single trust region (not this project's full
coordinated multi-region MORBO), so its numbers won't match this repo's
recorded experiment results -- that's expected, not a bug. The tradeoff
is deliberate: this folder is meant to be a small, from-scratch reference
implementation anyone can read start to finish, not a second copy of the
production system. What it keeps faithfully is the actual subject of this
project -- every shape method is the same function, producing the same
`(R, axis_lengths)` rotated-box representation, that the full system uses.

## Using `methods.py` / `benchmarks.py` standalone

Both are usable with zero knowledge of the rest of this folder -- useful
if you want the shape methods or benchmark implementations in a different
BO codebase entirely:

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
`run.py`/`optimizer.py` don't model composite structure (they only ever
call `eval_fn`) -- use `raw_eval_fn`/`composite_reduction` directly with
your own composite-aware modeling code if you want that.

## Adding a new benchmark or method

- **Benchmark**: write a `_make_<name>(dim, **kwargs) -> Benchmark` factory
  in `benchmarks.py` (see any existing one for the pattern), add it to the
  `BENCHMARKS` dict at the bottom. If it needs project-specific objective
  math, put that in a new file under `problems/` (self-contained --
  only `torch`/`numpy`, no imports from outside this folder).
- **Method**: write a `compute_<name>_shape(...) -> (R, axis_lengths)`
  function in `methods.py` returning the same `(R, axis_lengths)`
  convention as the others (see `methods.py`'s module docstring), add it
  to `SHAPE_METHODS`. It becomes runnable through `run.py` immediately --
  `optimizer.py` dispatches off `SHAPE_METHODS` directly, no separate
  wiring step needed (unlike this repo's full `morbo` engine, where a new
  method also needs registering in `morbo/trust_region.py`).

## Running the smoke tests

```bash
cd plug_and_play
python smoke_test.py
```
