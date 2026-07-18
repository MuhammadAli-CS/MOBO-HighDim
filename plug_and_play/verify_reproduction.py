r"""Verify that `plug_and_play/run.py` reproduces this project's own
recorded results -- i.e. that the modularized `methods.py`/`benchmarks.py`
interface is wired correctly, not just "runs without crashing."

Runs every method in `run.METHODS` on DTLZ2 (d=100, 2 objectives, 600
evals, batch 50, 200 initial points, 3 trust regions, min_tr_size=200,
seed=0) -- the exact configuration of `experiments/tr_shape_dtlz2_100d`
and `experiments/tr_shape_methods_dtlz2_100d` -- and compares the final
hypervolume against the already-recorded seed-0 `.pt` result for the
matching label in this repo.

Honesty note on exact reproduction: both paths call the identical
`morbo.run_one_replication.run_one_replication` function with the same
seed (which internally does `torch.manual_seed(BASE_SEED + seed)`), so
the *procedure* is provably the same -- but bit-exact reproduction across
different hardware/thread counts isn't guaranteed for PyTorch CPU code
(BLAS reduction order isn't invariant to thread count). This script
checks results land within a generous tolerance of the recorded value,
not that they're byte-identical; run it on the SAME cluster node/thread
config the original results were produced on for the tightest check.

Usage: python verify_reproduction.py [--tolerance-pct 10]
"""
import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from run import run  # noqa: E402

# (plug_and_play method name, reference experiment dir, reference label)
# -- "mab_shape" here uses the ducb policy (run.METHODS' default), so its
# correct reference is the "mab_shape_ducb" label, not plain "mab_shape"
# (which uses epsilon-greedy in the original study's LABEL_OVERRIDES).
CHECKS = [
    ("isotropic", "tr_shape_dtlz2_100d", "morbo"),
    ("ard_box", "tr_shape_dtlz2_100d", "ard_box"),
    ("pca_ellipsoid", "tr_shape_dtlz2_100d", "pca_ellipsoid"),
    ("ard_pca_ellipsoid", "tr_shape_dtlz2_100d", "ard_pca_ellipsoid"),
    ("labcat_style", "tr_shape_dtlz2_100d", "labcat_style"),
    ("cma_ellipsoid", "tr_shape_methods_dtlz2_100d", "cma_ellipsoid"),
    ("mab_shape", "tr_shape_methods_dtlz2_100d", "mab_shape_ducb"),
]

DIM = 100
MAX_EVALS = 600
BATCH_SIZE = 50
N_INITIAL_POINTS = 200
N_TRUST_REGIONS = 3
MIN_TR_SIZE = 200
SEED = 0


def _reference_hv(exp: str, label: str) -> float:
    path = os.path.join(_REPO_ROOT, "experiments", exp, label, f"{SEED:04d}_{label}.pt")
    d = torch.load(path, map_location="cpu", weights_only=False)
    return float(d["true_hv"][-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance-pct", type=float, default=10.0)
    args = parser.parse_args()

    print(f"Verifying plug_and_play/run.py against recorded results "
          f"(DTLZ2, d={DIM}, {MAX_EVALS} evals, seed={SEED})\n")
    print(f"{'method':22s} {'plug_and_play HV':>18s} {'recorded HV':>14s} {'delta %':>10s}  status")
    print("-" * 80)

    failures = []
    for method, ref_exp, ref_label in CHECKS:
        ref_hv = _reference_hv(ref_exp, ref_label)
        result = run(
            benchmark="dtlz2",
            method=method,
            dim=DIM,
            seed=SEED,
            max_evals=MAX_EVALS,
            batch_size=BATCH_SIZE,
            n_initial_points=N_INITIAL_POINTS,
            n_trust_regions=N_TRUST_REGIONS,
            min_tr_size=MIN_TR_SIZE,
            benchmark_kwargs={"num_objectives": 2},
            verbose=False,
        )
        got_hv = float(result["true_hv"][-1])
        delta_pct = (got_hv - ref_hv) / ref_hv * 100.0 if ref_hv != 0 else float("nan")
        ok = abs(delta_pct) <= args.tolerance_pct
        status = "OK" if ok else "FAIL"
        if not ok:
            failures.append((method, ref_exp, ref_label, ref_hv, got_hv, delta_pct))
        print(f"{method:22s} {got_hv:18.4f} {ref_hv:14.4f} {delta_pct:9.2f}%  {status}")

    print()
    if failures:
        print(f"{len(failures)}/{len(CHECKS)} methods exceeded {args.tolerance_pct}% tolerance:")
        for method, ref_exp, ref_label, ref_hv, got_hv, delta_pct in failures:
            print(f"  {method} (vs. {ref_exp}/{ref_label}): "
                  f"got {got_hv:.4f}, expected ~{ref_hv:.4f} ({delta_pct:+.2f}%)")
        sys.exit(1)
    print(f"All {len(CHECKS)} methods reproduced within {args.tolerance_pct}% of recorded results.")


if __name__ == "__main__":
    main()
