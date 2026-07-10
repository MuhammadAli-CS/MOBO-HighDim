#!/usr/bin/env python3
r"""Fast diagnostic: fit ONE real GP (no full BO loop) on synthetic
DTLZ2-shaped d=100 data, extract its ARD lengthscales, and check what
axis_lengths ratio compute_ard_box_shape actually produces -- testing
whether real fitted lengthscales at d=100 hit the [0.05, 4.0] constraint
bounds badly enough to explain ard_box's poor HV and cheap gen_time.
"""
import torch

from morbo.utils import get_fitted_model, compute_ard_box_shape, extract_ard_lengthscale

torch.manual_seed(0)
d = 100
n = 200  # matches n_initial_points

# Synthetic DTLZ2-like local data: cluster of points around a random center,
# roughly matching what a trust region's local X actually looks like.
X_center = torch.rand(1, d, dtype=torch.double)
X = (X_center + 0.1 * torch.randn(n, d, dtype=torch.double)).clamp(0.0, 1.0)

# DTLZ2 M=2 direct objectives
def dtlz2(X):
    x0 = X[:, 0]
    g = ((X[:, 1:] - 0.5) ** 2).sum(dim=-1)
    f1 = (1 + g) * torch.cos(x0 * torch.pi / 2)
    f2 = (1 + g) * torch.sin(x0 * torch.pi / 2)
    return torch.stack([f1, f2], dim=-1)

Y = dtlz2(X)

print(f"Fitting GP on {n} points, d={d}...")
model = get_fitted_model(X=X, Y=Y, use_ard=True, max_cholesky_size=50_000)

lengthscale = extract_ard_lengthscale(model, d)
print(f"lengthscale is None: {lengthscale is None}")
if lengthscale is not None:
    print(f"lengthscale min={lengthscale.min().item():.5f} max={lengthscale.max().item():.5f} "
          f"ratio={(lengthscale.max()/lengthscale.min()).item():.1f}x")
    print(f"lengthscale mean={lengthscale.mean().item():.4f} median={lengthscale.median().item():.4f}")

    length = torch.tensor(0.4, dtype=torch.double)
    R, axis_lengths = compute_ard_box_shape(lengthscale=lengthscale, length=length, dim=d)
    axis_lengths_clamped = axis_lengths.clamp(0.01, 1.6)  # length_min, length_max defaults
    print()
    print(f"axis_lengths (pre-clamp)  min={axis_lengths.min().item():.5f} max={axis_lengths.max().item():.5f} "
          f"ratio={(axis_lengths.max()/axis_lengths.min()).item():.1f}x")
    print(f"axis_lengths (post-clamp) min={axis_lengths_clamped.min().item():.5f} max={axis_lengths_clamped.max().item():.5f} "
          f"ratio={(axis_lengths_clamped.max()/axis_lengths_clamped.min()).item():.1f}x")
    n_clamped_low = (axis_lengths < 0.01).sum().item()
    n_clamped_high = (axis_lengths > 1.6).sum().item()
    print(f"dims clamped to length_min: {n_clamped_low}/{d}, clamped to length_max: {n_clamped_high}/{d}")

    # How many of the original local X points would fall inside this box?
    W = (X - X_center)
    inside = (W.abs() <= axis_lengths_clamped / 2).all(dim=-1)
    print()
    print(f"Of {n} local points, {inside.sum().item()} fall inside the ard_box region "
          f"(vs isotropic cube of same edge length would contain: "
          f"{(W.abs() <= length/2).all(dim=-1).sum().item()})")
