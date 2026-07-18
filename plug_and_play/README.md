# Plug-and-play: trust-region shape methods x benchmarks

A minimal, modular interface to this project's trust-region
shape-adaptation methods and benchmarks, split into two standalone files
plus a thin runner:

| File | What it is |
|---|---|
| [`methods.py`](methods.py) | Every trust-region shape-adaptation method (`isotropic`, `ard_box`, `pca_ellipsoid`, `ard_pca_ellipsoid`, `cma_ellipsoid`, `labcat_style`, `mab_shape`) as pure functions (or, for `mab_shape`, a small stateful bandit class). Zero dependency on the rest of this repo -- import it into any project. |
| [`benchmarks.py`](benchmarks.py) | Every benchmark used in this study (DTLZ1/2/3/5/7, `sparse_dtlz2`, `rotated_sparse_dtlz2`, `time_varying_sparse_dtlz2`, `rover`, `sparse_rover`, `bbob_biobj`, `lasso_bench_mo`, `penicillin`, `vehicle_safety`, `welded_beam`) behind one `get_benchmark(name, dim=..., **kwargs)` call, all normalized to the same `[0,1]^d` input / maximize-every-objective convention. Also zero dependency on the rest of this repo. |
| [`run.py`](run.py) | Thin glue: maps a `(benchmark, method)` pair onto this project's actual, validated MORBO engine (`morbo/run_one_replication.py`) and runs one full BO replication. This is the only file that touches the rest of the repo. |

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

Both files work with zero knowledge of MORBO or this repo -- useful if
you want the shape methods or benchmark implementations in a different
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

## Adding a new benchmark or method

- **Benchmark**: write a `_make_<name>(dim, **kwargs) -> Benchmark` factory
  in `benchmarks.py` (see any existing one for the pattern), add it to the
  `BENCHMARKS` dict at the bottom. If it needs to plug into `run.py` too,
  add a `name -> evalfn` entry to `run.py`'s `_EVALFN_MAP` (only needed if
  the benchmark isn't already known to `morbo/run_one_replication.py`).
- **Method**: write a `compute_<name>_shape(...) -> (R, axis_lengths)`
  function in `methods.py` returning the same `(R, axis_lengths)`
  convention as the others (see `methods.py`'s module docstring), add it
  to `SHAPE_METHODS`. To actually run it end-to-end via `run.py`, it also
  needs a `tr_shape` mode wired into `morbo/trust_region.py`'s
  `TurboHParams`/`TrustRegion._compute_shape_for_mode` (see that file for
  the existing methods' wiring as a template).

## Why `run.py` doesn't reimplement BO from scratch

`methods.py`'s functions are already exactly what powers this project's
full MORBO implementation via `TurboHParams(tr_shape=...)`. Writing a
second, simplified BO loop here would be new, untested code that could
silently diverge from the engine these methods were actually developed
and measured against -- so `run.py` reuses the real one. See its module
docstring for details.
