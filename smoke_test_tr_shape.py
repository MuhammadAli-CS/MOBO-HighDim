#!/usr/bin/env python3
r"""Small-scale sanity check for the `tr_shape` trust-region variants
(`ard_box`, `pca_ellipsoid`, `ard_pca_ellipsoid`) before running the full
`tr_shape_dtlz2_100d` comparison.

Runs `run_one_replication` at a tiny scale (dim=10, max_evals=40) once per
label, `verbose=True` so `TrustRegion.update()` prints `axis_lengths`/`R`
each iteration, and checks:
  (a) it runs to completion without exceptions,
  (b) for the isotropic baseline, no axis_lengths/R lines are printed at all
      (the isotropic path never touches them, confirmed structurally in
      `_update_tr_shape`'s caller in `update()`),
  (c) for the three new labels, R is non-identity at least once by the end
      of the run for the two PCA variants (guards against the shape-update
      insertion-point bug silently never firing again in the future) and
      axis_lengths is non-uniform at least once for all three.

Usage: python smoke_test_tr_shape.py
"""
import io
import sys
from contextlib import redirect_stdout

import torch

from morbo.run_one_replication import run_one_replication

DIM = 10
MAX_EVALS = 40
N_INITIAL_POINTS = 15
N_TRUST_REGIONS = 2
MIN_TR_SIZE = 12
BATCH_SIZE = 5


def run_and_capture(label: str, tr_shape: str) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_one_replication(
            seed=0,
            label=label,
            max_evals=MAX_EVALS,
            evalfn="DTLZ2",
            dim=DIM,
            batch_size=BATCH_SIZE,
            n_initial_points=N_INITIAL_POINTS,
            n_trust_regions=N_TRUST_REGIONS,
            min_tr_size=MIN_TR_SIZE,
            max_reference_point=[-6, -6],
            verbose=True,
            tr_shape=tr_shape,
            save_callback=lambda output: None,
        )
    return buf.getvalue()


def check_isotropic() -> None:
    print("=== isotropic (baseline regression guard) ===")
    out = run_and_capture("morbo", "isotropic")
    assert "axis_lengths:" not in out, "isotropic run printed axis_lengths -- shape logic fired when it shouldn't have"
    assert "R is identity:" not in out, "isotropic run printed R -- shape logic fired when it shouldn't have"
    print("OK: no shape-adaptation output for tr_shape='isotropic', as expected.")


def check_variant(label: str, tr_shape: str, expect_rotation: bool) -> None:
    print(f"=== {tr_shape} ===")
    out = run_and_capture(label, tr_shape)
    assert "axis_lengths:" in out, f"{tr_shape}: no axis_lengths printed at all -- shape update never fired (placement bug?)"

    axis_lines = [l for l in out.splitlines() if l.startswith("axis_lengths:")]
    r_lines = [l for l in out.splitlines() if l.startswith("R is identity:")]

    # Non-uniform axis_lengths at least once (parse the tensor's printed values
    # loosely -- just check they're not all equal within a line).
    saw_nonuniform = False
    for line in axis_lines:
        vals_str = line[len("axis_lengths: tensor("):].rstrip(")\n")
        try:
            vals = eval(vals_str.split(",")[0] if "[" not in vals_str else vals_str.split("]")[0] + "]")
        except Exception:
            continue
        if isinstance(vals, list) and len(set(round(v, 6) for v in vals)) > 1:
            saw_nonuniform = True
            break
    assert saw_nonuniform, f"{tr_shape}: axis_lengths stayed uniform the entire run -- shape math may be degenerate"

    if expect_rotation:
        saw_rotation = any("R is identity: False" in line for line in r_lines)
        assert saw_rotation, f"{tr_shape}: R was identity the entire run -- PCA rotation never activated"
        print(f"OK: axis_lengths varied and R became non-identity during the run.")
    else:
        assert all("True" in line for line in r_lines), f"{tr_shape}: R should stay identity (ard_box has no rotation) but wasn't"
        print(f"OK: axis_lengths varied, R stayed identity as expected for ard_box.")


def run_and_capture_kw(label: str, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_one_replication(
            seed=0,
            label=label,
            max_evals=MAX_EVALS,
            evalfn="DTLZ2",
            dim=DIM,
            batch_size=BATCH_SIZE,
            n_initial_points=N_INITIAL_POINTS,
            n_trust_regions=N_TRUST_REGIONS,
            min_tr_size=MIN_TR_SIZE,
            max_reference_point=[-6, -6],
            verbose=True,
            save_callback=lambda output: None,
            **kwargs,
        )
    return buf.getvalue()


def check_linear_kernel() -> None:
    print("=== linear_gp (spherical linear kernel, isotropic shape) ===")
    out = run_and_capture_kw("linear_gp", tr_shape="isotropic", use_linear_kernel=True)
    # Isotropic shape => no shape-adaptation output; this check just confirms
    # the linear-kernel model path runs a full BO loop end-to-end.
    assert "axis_lengths:" not in out
    print("OK: linear-kernel BO loop ran to completion.")


def check_dim_prior() -> None:
    print("=== ard_box_dimprior (dim-scaled lengthscale prior) ===")
    out = run_and_capture_kw(
        "ard_box_dimprior", tr_shape="ard_box", use_dim_scaled_ls_prior=True
    )
    assert "axis_lengths:" in out, "ard_box shape update never fired"
    print("OK: ard_box + dim-scaled prior ran to completion.")


def check_mab_shape() -> None:
    print("=== mab_shape (per-TR bandit over shapes) ===")
    # Force exploration (mab_epsilon=1.0) so a short smoke run reliably
    # visits a non-isotropic arm at least once, without depending on the
    # exploitation path's argmax tie-breaking (which favors the first arm,
    # "isotropic", when reward estimates are still equal/near-zero).
    out = run_and_capture_kw("mab_shape", tr_shape="mab_shape", mab_epsilon=1.0)
    assert "axis_lengths:" in out, "mab_shape: shape update never fired"
    r_lines = [l for l in out.splitlines() if l.startswith("R is identity:")]
    assert any("False" in l for l in r_lines), (
        "mab_shape: R was identity for the entire run even under full "
        "exploration (mab_epsilon=1.0) -- arm selection likely isn't wired up."
    )
    print("OK: bandit selected a rotated arm at least once under full exploration.")


def check_sparse_dtlz2() -> None:
    print("=== SparseDTLZ2 (partial effective-dimension problem) ===")
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_one_replication(
            seed=0,
            label="pca_ellipsoid",
            max_evals=MAX_EVALS,
            evalfn="SparseDTLZ2",
            dim=DIM,
            batch_size=BATCH_SIZE,
            n_initial_points=N_INITIAL_POINTS,
            n_trust_regions=N_TRUST_REGIONS,
            min_tr_size=MIN_TR_SIZE,
            max_reference_point=[-6, -6],
            sparse_dtlz2_k_eff=2,
            tr_shape="pca_ellipsoid",
            verbose=True,
            save_callback=lambda output: None,
        )
    out = buf.getvalue()
    assert "axis_lengths:" in out, "SparseDTLZ2: pca_ellipsoid shape update never fired"
    print("OK: SparseDTLZ2 BO loop with k_eff=2 ran to completion.")


if __name__ == "__main__":
    torch.manual_seed(0)
    check_isotropic()
    check_variant("ard_box", "ard_box", expect_rotation=False)
    check_variant("pca_ellipsoid", "pca_ellipsoid", expect_rotation=True)
    check_variant("ard_pca_ellipsoid", "ard_pca_ellipsoid", expect_rotation=True)
    check_variant("cma_ellipsoid", "cma_ellipsoid", expect_rotation=True)
    check_linear_kernel()
    check_dim_prior()
    check_mab_shape()
    check_sparse_dtlz2()
    print("\nAll smoke tests passed.")
