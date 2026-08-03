#!/usr/bin/env python3
r"""A composite-structure, genuinely high-dimensional (1024D) photonic
crystal bandgap-design benchmark: a 2D square-lattice photonic crystal
whose unit cell is a free-form 32x32 pixelated dielectric-density map
(the standard "topology optimization" continuous relaxation used in real
inverse photonic design, not a fixed rod/hole shape), solved via the
plane-wave expansion (PWE) method -- the same physics MPB
(MIT Photonic Bands) implements, self-derived here in pure NumPy/SciPy
rather than depending on MPB/MEEP (compiled C/MPI/HDF5 tools that would
have repeated this project's OC20/fairchem isolated-environment ordeal
for no real benefit, since the underlying physics is a textbook
eigenvalue problem cheap enough to just implement directly).

Physics (Joannopoulos, Johnson, Winn, Meade, "Photonic Crystals: Molding
the Flow of Light", 2nd ed., Ch. 5): for a 2D photonic crystal periodic in
the x-y plane, Maxwell's equations decouple into two independent scalar
polarizations, each a Hermitian (generalized) eigenvalue problem in a
truncated plane-wave (reciprocal-lattice) basis:
    E-pol (E_z out of plane, historically "TM"):
        diag((k+G)^2) e = (omega/c)^2 [eps(G-G')] e      (generalized)
    H-pol (H_z out of plane, historically "TE"):
        [(k+G).(k+G') eta(G-G')] h = (omega/c)^2 h        (standard)
where eps(G) and eta(G) are Fourier coefficients of the real-space
dielectric function eps(r) and its inverse 1/eps(r). This module's PWE
solver was verified directly against the classic textbook validation case
(square lattice of dielectric rods, eps=8.9, r=0.2a in air): it reproduces
the known E-pol band-1/2 gap (~0.32 to ~0.44, in units of 2*pi*c/a)
almost exactly, and correctly shows no H-pol gap for that same structure
-- both textbook-known results, not tuned to match.

Raw composite response g(x): the lowest 4 bands' frequencies at 6 k-points
along the square lattice's irreducible Brillouin zone path
(Gamma - X - M - Gamma), for both polarizations -- 6*4*2 = 48-dim,
the same checkpointed-trajectory-style pattern as this package's other
composite benchmarks (expose the intermediate physical quantity a real
solver computes, not just the final reduced objectives).

Known reduction L(g): the complete photonic bandgap width for each
polarization between bands 1 and 2 -- max_k(band_1(k)) subtracted from
min_k(band_2(k)) -- is the standard, textbook definition of "does this
crystal have a gap, and how wide" (Joannopoulos et al., same reference).
Both are maximized simultaneously: finding a structure with *both* a wide
E-pol gap and a wide H-pol gap (a "complete" bandgap) is a well-known hard
problem in photonics, since the two polarizations' physics couple to the
same dielectric structure through different combinations of Fourier
coefficients -- a genuine, not synthetic, two-objective trade-off.
"""
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

PHOTONIC_GRID = 32  # design pixels per side
PHOTONIC_DIM = PHOTONIC_GRID * PHOTONIC_GRID  # 1024
_A = 1.0  # lattice constant
_GCUT = 4  # reciprocal-lattice truncation: G=(2pi/a)(m,n), m,n in [-GCUT,GCUT]
_N_BANDS = 4
_EPS_BG = 1.0
_EPS_HI = 8.9  # silicon-like, matches the textbook validation case

_G_LIST = [(m, n) for m in range(-_GCUT, _GCUT + 1) for n in range(-_GCUT, _GCUT + 1)]
_NG = len(_G_LIST)


def _kpath(n_points: int = 6):
    """Evenly-spaced points along Gamma-X-M-Gamma, the irreducible
    Brillouin-zone path of a 2D square lattice."""
    pts_sym = [(0.0, 0.0), (np.pi / _A, 0.0), (np.pi / _A, np.pi / _A), (0.0, 0.0)]
    seg_lengths = [np.hypot(pts_sym[i + 1][0] - pts_sym[i][0], pts_sym[i + 1][1] - pts_sym[i][1]) for i in range(3)]
    total = sum(seg_lengths)
    pts = []
    for i in range(n_points):
        s = total * i / (n_points - 1)
        acc = 0.0
        for seg in range(3):
            if s <= acc + seg_lengths[seg] or seg == 2:
                t = 0.0 if seg_lengths[seg] == 0 else (s - acc) / seg_lengths[seg]
                t = min(max(t, 0.0), 1.0)
                k0, k1 = pts_sym[seg], pts_sym[seg + 1]
                pts.append((k0[0] + t * (k1[0] - k0[0]), k0[1] + t * (k1[1] - k0[1])))
                break
            acc += seg_lengths[seg]
    return pts


_KPTS = _kpath(6)


def _fourier_coeffs(eps_grid: np.ndarray) -> np.ndarray:
    n = eps_grid.shape[0]
    F = np.fft.fft2(eps_grid) / (n * n)
    return np.fft.fftshift(F)


def _get_coeff(F: np.ndarray, n: int, dm: int, dn: int) -> complex:
    i, j = dm + n // 2, dn + n // 2
    if 0 <= i < n and 0 <= j < n:
        return F[i, j]
    return 0.0 + 0.0j


def _bands_at_k(kx: float, ky: float, F: np.ndarray, n: int, polarization: str) -> np.ndarray:
    kGx = np.array([kx + 2 * np.pi / _A * m for (m, _n) in _G_LIST])
    kGy = np.array([ky + 2 * np.pi / _A * _n for (m, _n) in _G_LIST])

    coeff_cache = {}
    for a in range(_NG):
        for b in range(_NG):
            dm = _G_LIST[a][0] - _G_LIST[b][0]
            dn = _G_LIST[a][1] - _G_LIST[b][1]
            if (dm, dn) not in coeff_cache:
                coeff_cache[(dm, dn)] = _get_coeff(F, n, dm, dn)

    if polarization == "H":
        M = np.zeros((_NG, _NG), dtype=complex)
        for a in range(_NG):
            for b in range(_NG):
                dm = _G_LIST[a][0] - _G_LIST[b][0]
                dn = _G_LIST[a][1] - _G_LIST[b][1]
                M[a, b] = (kGx[a] * kGx[b] + kGy[a] * kGy[b]) * coeff_cache[(dm, dn)]
        w2 = np.linalg.eigvalsh(M)[:_N_BANDS]
    else:
        from scipy.linalg import eigh as sp_eigh

        Adiag = kGx**2 + kGy**2
        B = np.zeros((_NG, _NG), dtype=complex)
        for a in range(_NG):
            for b in range(_NG):
                dm = _G_LIST[a][0] - _G_LIST[b][0]
                dn = _G_LIST[a][1] - _G_LIST[b][1]
                B[a, b] = coeff_cache[(dm, dn)]
        Amat = np.diag(Adiag).astype(complex)
        w2 = sp_eigh(Amat, B, eigvals_only=True, subset_by_index=[0, _N_BANDS - 1])

    return np.sqrt(np.clip(w2, 0, None)) / (2 * np.pi)


def _bands_for_density(density: np.ndarray) -> np.ndarray:
    r"""density: PHOTONIC_GRID x PHOTONIC_GRID array in [0,1] (fraction of
    high-index material). Returns (6, 4, 2)-shape array: k-points x bands
    x [E-pol, H-pol]."""
    eps_grid = _EPS_BG + density * (_EPS_HI - _EPS_BG)
    eps_inv_grid = 1.0 / eps_grid
    F_eps = _fourier_coeffs(eps_grid)
    F_eta = _fourier_coeffs(eps_inv_grid)

    out = np.zeros((len(_KPTS), _N_BANDS, 2))
    for ki, (kx, ky) in enumerate(_KPTS):
        out[ki, :, 0] = _bands_at_k(kx, ky, F_eps, PHOTONIC_GRID, "E")
        out[ki, :, 1] = _bands_at_k(kx, ky, F_eta, PHOTONIC_GRID, "H")
    return out


def get_composite_photonic_fn(dtype=torch.double, device=None) -> Tuple[callable, Tensor]:
    r"""Construct the raw-response (band-structure) function and bounds
    for composite photonic-crystal bandgap design.

    Returns:
        A tuple `(raw_response, bounds)`:
            raw_response: callable mapping an `n x 1024`-dim tensor (raw,
                *unnormalized* per-pixel dielectric-density fractions,
                each in `[0, 1]`) to an `n x 48`-dim tensor: 6 k-points
                along Gamma-X-M-Gamma x 4 bands x 2 polarizations
                (E-pol, H-pol), flattened in that order.
            bounds: `2 x 1024`-dim tensor, `[0, 1]` per pixel.
    """
    def raw_response(X_input: Tensor) -> Tensor:
        X_flat = X_input.reshape(-1, PHOTONIC_DIM)
        rows = []
        for row in X_flat.detach().cpu().numpy():
            density = np.clip(row, 0.0, 1.0).reshape(PHOTONIC_GRID, PHOTONIC_GRID)
            bands = _bands_for_density(density)
            rows.append(bands.reshape(-1))
        out = torch.tensor(np.stack(rows), dtype=dtype, device=device)
        return out.view(*X_input.shape[:-1], len(_KPTS) * _N_BANDS * 2)

    bounds = torch.tensor([[0.0] * PHOTONIC_DIM, [1.0] * PHOTONIC_DIM], dtype=dtype, device=device)
    return raw_response, bounds


def composite_photonic_reduction(Y_raw: Tensor) -> Tensor:
    r"""Known reduction: the complete bandgap width between bands 1 and 2,
    for each polarization -- `max_k(band_1(k))` subtracted from
    `min_k(band_2(k))` -- the standard textbook definition of a photonic
    bandgap (Joannopoulos et al., see module docstring).

    Args:
        Y_raw: `... x 48`-dim tensor, 6 k-points x 4 bands x 2
            polarizations (E-pol, H-pol), as produced by `raw_response`
            above.

    Returns:
        An `... x 2`-dim tensor `[gap_E, gap_H]` (maximize convention:
        wider gaps are better; a negative value means the bands overlap,
        i.e. no gap at that k-sampling resolution).
    """
    n_k = len(_KPTS)
    Y = Y_raw.view(*Y_raw.shape[:-1], n_k, _N_BANDS, 2)
    band1_all_k = Y[..., :, 0, :]  # (..., n_k, 2)
    band2_all_k = Y[..., :, 1, :]  # (..., n_k, 2)

    max_band1 = band1_all_k.amax(dim=-2)  # (..., 2)
    min_band2 = band2_all_k.amin(dim=-2)  # (..., 2)
    gap = min_band2 - max_band1  # (..., 2): [gap_E, gap_H]
    return gap
