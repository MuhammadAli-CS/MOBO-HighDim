#!/usr/bin/env python3
r"""Diagnostic: isolate whether sample_tr_discrete_points_subset_d_rotated is
intrinsically cheaper than sample_tr_discrete_points_subset_d for matched
inputs (same d, same n_discrete_points, same best_X size), or whether the
~20x gen_time gap seen in tr_shape_dtlz2_100d comes from somewhere else in
the pipeline (e.g. downstream HVI computation depending on candidate values).
"""
import time
import torch

from morbo.utils import (
    sample_tr_discrete_points_subset_d,
    sample_tr_discrete_points_subset_d_rotated,
)

torch.manual_seed(0)
d = 100
n_discrete_points = 4096
n_best_X = 50  # rough Pareto-front-in-TR size

X_center = torch.rand(1, d, dtype=torch.double)
best_X = torch.rand(n_best_X, d, dtype=torch.double)
length = torch.tensor(0.4, dtype=torch.double)
normalized_tr_bounds = torch.stack(
    [X_center[0] - length / 2, X_center[0] + length / 2], dim=0
).clamp(0.0, 1.0)

R_identity = torch.eye(d, dtype=torch.double)
axis_lengths_uniform = length.expand(d).clone()

# random orthonormal rotation, for the genuinely-rotated case
Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.double))
axis_lengths_varied = (length * torch.exp(torch.randn(d, dtype=torch.double) * 0.3))

N_REPS = 20

def bench(fn, *args, **kwargs):
    # warmup
    fn(*args, **kwargs)
    t0 = time.perf_counter()
    for _ in range(N_REPS):
        fn(*args, **kwargs)
    return (time.perf_counter() - t0) / N_REPS


t_isotropic = bench(
    sample_tr_discrete_points_subset_d,
    best_X=best_X,
    normalized_tr_bounds=normalized_tr_bounds,
    n_discrete_points=n_discrete_points,
    length=length,
    qmc=True,
    trunc_normal_perturb=False,
)
print(f"sample_tr_discrete_points_subset_d (isotropic path):        {t_isotropic*1000:.2f} ms/call")

t_rotated_identity = bench(
    sample_tr_discrete_points_subset_d_rotated,
    best_X=best_X,
    X_center=X_center,
    R=R_identity,
    axis_lengths=axis_lengths_uniform,
    n_discrete_points=n_discrete_points,
    qmc=True,
)
print(f"sample_tr_discrete_points_subset_d_rotated (R=I, matches ard_box): {t_rotated_identity*1000:.2f} ms/call")

t_rotated_real = bench(
    sample_tr_discrete_points_subset_d_rotated,
    best_X=best_X,
    X_center=X_center,
    R=Q,
    axis_lengths=axis_lengths_varied,
    n_discrete_points=n_discrete_points,
    qmc=True,
)
print(f"sample_tr_discrete_points_subset_d_rotated (R=random, matches pca): {t_rotated_real*1000:.2f} ms/call")

print()
print(f"Ratio isotropic / rotated(R=I):   {t_isotropic / t_rotated_identity:.2f}x")
print(f"Ratio isotropic / rotated(R=rnd): {t_isotropic / t_rotated_real:.2f}x")
