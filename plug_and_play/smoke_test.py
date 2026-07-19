r"""Fast local sanity checks: every benchmark evaluates, and every method
runs to completion end-to-end through the REAL engine (``run.py`` ->
``morbo.run_one_replication`` -> ``morbo/trust_region.py``'s
``TurboHParams(tr_shape=...)``, the same tested implementation this
project's recorded results come from -- not a simplified reimplementation).
This only checks correctness at toy scale (wiring, no crashes, finite
output, monotone hypervolume); it does NOT check numerical reproduction
of any specific recorded result -- see ``verify_reproduction.py`` for
that (byte-for-byte comparison against this repo's own recorded
experiment results at full scale).

Usage: python smoke_test.py
"""
import torch

from benchmarks import BENCHMARKS, get_benchmark
from methods import SHAPE_METHODS
from run import METHODS, run

DIM = 8
N_INITIAL_POINTS = 15
N_TRUST_REGIONS = 2
MIN_TR_SIZE = 12
BATCH_SIZE = 5
MAX_EVALS = 40


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
    print("\n=== methods: end-to-end optimization on dtlz2 (real morbo engine) ===")
    assert set(SHAPE_METHODS) | {"mab_shape"} == set(METHODS), (
        f"run.METHODS out of sync with methods.SHAPE_METHODS: "
        f"{set(SHAPE_METHODS) | {'mab_shape'} ^ set(METHODS)}"
    )
    for method in METHODS:
        result = run(
            benchmark="dtlz2", method=method, dim=DIM, seed=0,
            max_evals=MAX_EVALS, batch_size=BATCH_SIZE,
            n_initial_points=N_INITIAL_POINTS, n_trust_regions=N_TRUST_REGIONS,
            min_tr_size=MIN_TR_SIZE, benchmark_kwargs={"num_objectives": 2},
        )
        true_hv = torch.as_tensor(result["true_hv"])
        assert torch.isfinite(true_hv).all(), f"{method}: non-finite hypervolume trace"
        assert (true_hv.diff() >= -1e-9).all(), (
            f"{method}: hypervolume decreased somewhere -- should be monotone "
            f"non-decreasing by construction (Pareto front only grows)"
        )
        assert result["n_evals"][-1] == MAX_EVALS, (
            f"{method}: expected {MAX_EVALS} evaluations, got {result['n_evals'][-1]}"
        )
        print(f"OK {method:20s} final HV = {float(true_hv[-1]):.4f}")


if __name__ == "__main__":
    check_all_benchmarks_evaluate()
    check_all_methods_optimize()
    print("\nAll smoke tests passed.")
