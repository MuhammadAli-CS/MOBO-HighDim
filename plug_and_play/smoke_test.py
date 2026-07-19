r"""Fast local sanity checks for this folder's self-contained
implementation -- every benchmark evaluates and every method optimizes,
at toy scale, with no dependency outside this folder.

This replaced an earlier version of this file that compared results
against this repo's full multi-region MORBO engine (`morbo/run_one_replication.py`)
byte-for-byte. That comparison stopped being meaningful once `optimizer.py`
became a genuinely different (simpler, single-trust-region) algorithm
rather than a thin wrapper around the full engine -- the point of this
folder is a small, from-scratch reference implementation, not a
numerically-matching reimplementation. What's checked here instead: every
method actually runs to completion, produces finite output, and improves
(or at least never decreases) hypervolume over the run -- i.e. the wiring
is correct, not that the numbers match some other implementation.

Usage: python smoke_test.py
"""
import torch

from benchmarks import BENCHMARKS, get_benchmark
from methods import SHAPE_METHODS
from run import METHODS, run

DIM = 8
N_INIT = 12
N_ITER = 3
BATCH_SIZE = 4


def check_all_benchmarks_evaluate() -> None:
    print("=== benchmarks: shape/finiteness ===")
    for name in BENCHMARKS:
        kwargs = {}
        if name in ("sparse_dtlz2", "rotated_sparse_dtlz2"):
            kwargs = {"num_objectives": 2, "k_eff": 3}
        elif name == "time_varying_sparse_dtlz2":
            kwargs = {"num_objectives": 2, "k_eff": 3, "switch_at_eval": 10}
        elif name in ("dtlz1", "dtlz2", "dtlz3", "dtlz5", "dtlz7"):
            kwargs = {"num_objectives": 2}
        elif name == "rover":
            kwargs = {"dim": 20}
        elif name == "sparse_rover":
            kwargs = {"dim": 26, "base_dim": 20}
        elif name == "bbob_biobj":
            kwargs = {"f1_name": "rastrigin", "f2_name": "peaks"}
        elif name == "lasso_bench_mo":
            try:
                import LassoBench  # noqa: F401
            except ImportError:
                print(f"SKIPPED {name} (LassoBench not installed)")
                continue

        dim = kwargs.pop("dim", DIM)
        b = get_benchmark(name, dim=dim, **kwargs)
        X = torch.rand(5, b.dim, dtype=torch.double)
        Y = b.eval_fn(X)
        assert Y.shape == (5, b.num_objectives), f"{name}: bad shape {Y.shape}"
        assert torch.isfinite(Y).all(), f"{name}: non-finite output"
        print(f"OK {name:28s} dim={b.dim:<4d} M={b.num_objectives}")


def check_all_methods_optimize() -> None:
    print("\n=== methods: end-to-end optimization on dtlz2 ===")
    assert set(SHAPE_METHODS) | {"mab_shape"} == set(METHODS), (
        f"run.METHODS out of sync with methods.SHAPE_METHODS: "
        f"{set(SHAPE_METHODS) | {'mab_shape'} ^ set(METHODS)}"
    )
    for method in METHODS:
        result = run(
            benchmark="dtlz2", method=method, dim=DIM, seed=0,
            n_init=N_INIT, n_iter=N_ITER, batch_size=BATCH_SIZE,
            benchmark_kwargs={"num_objectives": 2},
        )
        hv_history = result["hv_history"]
        assert torch.isfinite(hv_history).all(), f"{method}: non-finite hypervolume trace"
        assert (hv_history.diff() >= -1e-9).all(), (
            f"{method}: hypervolume decreased somewhere -- should be monotone "
            f"non-decreasing by construction (Pareto front only grows)"
        )
        n_expected = N_INIT + N_ITER * BATCH_SIZE
        assert result["X"].shape[0] == n_expected, (
            f"{method}: expected {n_expected} evaluations, got {result['X'].shape[0]}"
        )
        print(f"OK {method:20s} final HV = {result['final_hypervolume']:.4f}")


if __name__ == "__main__":
    check_all_benchmarks_evaluate()
    check_all_methods_optimize()
    print("\nAll smoke tests passed.")
