#!/usr/bin/env python3
r"""A composite-structure, genuinely high-dimensional (1152D) structural
topology-optimization benchmark: 2D linear-elastic compliance minimization
of a half-MBB beam via the Solid Isotropic Material with Penalization
(SIMP) method, using the exact element stiffness matrix, DOF numbering,
and boundary conditions from the widely-used, peer-reviewed "88 lines"
MATLAB code \citep{andreassen2011topopt88} (itself an efficiency
rewrite of Sigmund's original 99-line code) -- verified here against a
public mirror of the actual `top88.m` source, not retyped from the
paper's typeset equations, to avoid the same class of transcription
error already caught once in this project's RCM17/EvoXBench/Optiland
surveys (see methods.tex's benchmark-candidate survey section).

Design variables x (1152 = 48x24 element grid): per-element material
density in [0, 1], passed through the same radius-based density filter
`top88.m` itself uses (`rmin`-weighted neighbor averaging, verified
against the mirrored source's `H`/`Hs` construction) before the FE
solve. Unlike the gradient-based SIMP setting the filter was originally
built for (mesh-independency, checkerboard suppression), here it serves
a second, load-bearing purpose: raw iid-random densities (as MORBO's
Sobol initialization and off-manifold exploration will propose) can by
chance isolate a near-zero-stiffness element along the sole load path,
producing a near-singular effective stiffness and a compliance outlier
several orders of magnitude beyond the typical range (verified
empirically: unfiltered random designs' compliance spans 4+ orders of
magnitude at the p99 tail). The filter -- real, standard SIMP practice,
not a benchmark-specific patch -- smooths exactly this pathology by
blending each element's density with its neighbors within `rmin`.

Raw composite response g(x): solving the linear system K(x) U = F (a
real, sparse, ~2500-DOF finite-element solve every evaluation -- the
"simulation" here, exactly analogous to RCM40/46's admittance solve and
composite_photonic's plane-wave eigenvalue solve) yields the per-element
strain energy / compliance contribution
    ce_e = (Emin + x_e^p (E0 - Emin)) * u_e^T KE u_e
Exposing all 1152 elements directly would repeat the GP-fitting crash
risk already found at OC20's 100-dim raw output (see composite_oc20.py);
following the same fix used in composite_photonic.py (6 k-points x 4
bands x 2 pols = 48, rather than the full 1024-pixel field), this module
aggregates ce into a coarse 8x6 = 48-cell block grid (each block summing
a 6x4 patch of elements) -- a genuine physical intermediate (regional
strain-energy distribution), not a synthetic dimensionality reduction:
the total compliance reduction below is an *exact* sum over these
blocks, not an approximation.

Known reduction L(g): both objectives are literal reductions over the
same 48-dim block field --
    f_1 (total compliance)     = sum_blocks(ce_block)
    f_2 (peak block compliance) = max_blocks(ce_block)
f_1 is the standard SIMP objective (minimize compliance = maximize
stiffness for the given material budget). f_2 is a physically motivated
proxy for strain-energy concentration/robustness (avoiding a design that
is globally stiff but relies on a single overloaded region) -- a real
structural-engineering concern, though a coarse proxy for, not
equivalent to, the pointwise von Mises stress constraints used in the
stress-constrained topology optimization literature (that would require
computing per-element stress via the constitutive matrix, not attempted
here). The two objectives are genuinely in tension: the compliance-
optimal SIMP criterion concentrates material precisely where local
strain energy density is highest, which is exactly what increases
f_2 -- so minimizing f_1 and minimizing f_2 are not aligned.

Both objectives are minimized in the underlying physics; this module
returns them negated to maximize, matching this project's convention
elsewhere (`composite_rcm40_reduction`, `composite_photonic_reduction`).

Boundary conditions (half-MBB beam, the classic example the 88-line
code itself uses): left edge horizontally roller-supported (all
left-column nodes' x-DOF fixed, representing the symmetry plane of a
full MBB beam), bottom-right corner vertically roller-supported, unit
downward point load at the top-left corner. Verified directly from the
mirrored `top88.m` source's `KE`, `edofMat`, `fixeddofs`, and `F`
definitions (not re-derived from scratch), the same "extract don't
retype" discipline used for RCM40/46's admittance data.
"""
from typing import Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import Tensor

NELX = 48
NELY = 24
TOPOPT_DIM = NELX * NELY  # 1152
_E0 = 1.0
_EMIN = 1e-9
_NU = 0.3
_PENAL = 3.0
_RMIN = 1.5

_BLOCK_ROWS = 6  # blocks along y
_BLOCK_COLS = 8  # blocks along x
assert NELY % _BLOCK_ROWS == 0 and NELX % _BLOCK_COLS == 0
_BH = NELY // _BLOCK_ROWS  # elements per block, y
_BW = NELX // _BLOCK_COLS  # elements per block, x
TOPOPT_RAW_DIM = _BLOCK_ROWS * _BLOCK_COLS  # 48


def _element_stiffness(nu: float) -> np.ndarray:
    r"""Verbatim from the mirrored `top88.m` (Andreassen et al. 2011):
    the 8x8 bilinear plane-stress quad element stiffness matrix for a
    unit square element, unit Young's modulus."""
    A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]], dtype=float)
    A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]], dtype=float)
    B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]], dtype=float)
    B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]], dtype=float)
    top = np.hstack([A11, A12])
    bot = np.hstack([A12.T, A11])
    KE_A = np.vstack([top, bot])
    top_b = np.hstack([B11, B12])
    bot_b = np.hstack([B12.T, B11])
    KE_B = np.vstack([top_b, bot_b])
    return (KE_A + nu * KE_B) / (1 - nu**2) / 24


def _build_density_filter(nelx: int, nely: int, rmin: float) -> sp.csr_matrix:
    r"""Verbatim translation of `top88.m`'s `H`/`Hs` density-filter
    construction (radius-`rmin` neighbor weighting in element-index
    space), returning the normalized filter matrix `H / Hs` (so that
    `xPhys_flat = filter_matrix @ x_flat`, with `x_flat` in the same
    column-major (e1 = (i1-1)*nely+j1) element order used throughout."""
    iH, jH, sH = [], [], []
    for i1 in range(1, nelx + 1):
        for j1 in range(1, nely + 1):
            e1 = (i1 - 1) * nely + j1
            for i2 in range(max(i1 - (int(np.ceil(rmin)) - 1), 1), min(i1 + (int(np.ceil(rmin)) - 1), nelx) + 1):
                for j2 in range(max(j1 - (int(np.ceil(rmin)) - 1), 1), min(j1 + (int(np.ceil(rmin)) - 1), nely) + 1):
                    e2 = (i2 - 1) * nely + j2
                    w = max(0.0, rmin - np.hypot(i1 - i2, j1 - j2))
                    if w > 0:
                        iH.append(e1 - 1)
                        jH.append(e2 - 1)
                        sH.append(w)
    nele = nelx * nely
    H = sp.coo_matrix((sH, (iH, jH)), shape=(nele, nele)).tocsr()
    Hs = np.asarray(H.sum(axis=1)).flatten()
    return sp.diags(1.0 / Hs) @ H


_FILTER = _build_density_filter(NELX, NELY, _RMIN)


def _build_dof_maps(nelx: int, nely: int):
    r"""Verbatim translation of `top88.m`'s `nodenrs`/`edofVec`/`edofMat`
    (1-indexed in the original MATLAB, converted to 0-indexed here)."""
    nodenrs = np.arange(1, (1 + nelx) * (1 + nely) + 1).reshape((nely + 1, nelx + 1), order="F")
    edofVec = (2 * nodenrs[:-1, :-1] + 1).flatten(order="F")  # (nelx*nely,), 1-indexed
    offsets = np.array([0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1])
    edofMat = edofVec[:, None] + offsets[None, :]  # (nelx*nely, 8), 1-indexed dofs
    return edofMat - 1  # 0-indexed


def _solve_topopt(x_density: np.ndarray) -> np.ndarray:
    r"""x_density: (NELY, NELX) array in [0, 1]. Returns the (NELY, NELX)
    per-element compliance contribution `ce` (SIMP-scaled)."""
    nelx, nely = NELX, NELY
    ndof = 2 * (nelx + 1) * (nely + 1)
    KE = _element_stiffness(_NU)
    edofMat = _build_dof_maps(nelx, nely)  # (nele, 8), 0-indexed

    x_flat = x_density.flatten(order="F")  # (nele,), matches edofMat's element order
    xPhys = _FILTER @ x_flat  # rmin-radius neighbor-averaged density (see module docstring)
    scale = _EMIN + xPhys**_PENAL * (_E0 - _EMIN)

    rows = np.repeat(edofMat, 8, axis=1).reshape(-1)
    cols = np.tile(edofMat, (1, 8)).reshape(-1)
    vals = (scale[:, None] * KE.flatten()[None, :]).reshape(-1)
    K = sp.coo_matrix((vals, (rows, cols)), shape=(ndof, ndof)).tocsr()
    K = (K + K.T) / 2

    F = np.zeros(ndof)
    F[1] = -1.0  # y-dof of top-left node (0-indexed dof 1)

    fixed_x = np.arange(0, 2 * (nely + 1), 2)  # left-edge x-dofs, all rows
    fixed_corner = np.array([2 * (nelx + 1) * (nely + 1) - 1])  # bottom-right y-dof
    fixeddofs = np.union1d(fixed_x, fixed_corner)
    alldofs = np.arange(ndof)
    freedofs = np.setdiff1d(alldofs, fixeddofs)

    U = np.zeros(ndof)
    U[freedofs] = spla.spsolve(K[freedofs, :][:, freedofs].tocsc(), F[freedofs])

    Ue = U[edofMat]  # (nele, 8)
    ce_unscaled = np.einsum("ij,jk,ik->i", Ue, KE, Ue)
    ce = scale * ce_unscaled
    return ce.reshape((nely, nelx), order="F")


_LOG_EPS = 1e-6


def _blockify(ce_grid: np.ndarray) -> np.ndarray:
    r"""(NELY, NELX) -> (TOPOPT_RAW_DIM,) block-summed compliance field,
    returned in log-space (see `get_composite_topopt_fn` docstring for
    why)."""
    blocks = ce_grid.reshape(_BLOCK_ROWS, _BH, _BLOCK_COLS, _BW).sum(axis=(1, 3))
    return np.log(blocks.reshape(-1) + _LOG_EPS)


def get_composite_topopt_fn(dtype=torch.double, device=None) -> Tuple[callable, Tensor]:
    r"""Construct the raw-response (block-compliance field) function and
    bounds for composite SIMP topology optimization.

    The per-block compliance field is strictly positive and spans
    several orders of magnitude across blocks (blocks near the load/
    support dominate the strain-energy field over distant ones by
    100x+) -- verified empirically: the raw covariance matrix's
    condition number is ~2e8 in linear space, which crashed
    `TS_select_batch_MORBO`'s joint multitask posterior sampling with
    the same Windows `linear_operator` Cholesky access violation
    already seen at OC20's 100-dim raw output (see composite_oc20.py),
    but here triggered by numerical conditioning rather than
    dimension (confirmed via traceback into
    `MultitaskMultivariateNormal.rsample` -> `_cholesky`, not a
    dimension-count code path). `raw_response` therefore returns
    `log(block_compliance + 1e-6)` rather than the linear-space value
    -- standard practice for GP-modeling a strictly-positive,
    multiplicative-scale physical quantity (this project already uses
    the analogous log-scale parameterization for GRI-Mech's Arrhenius
    rate constants, see composite_gri_mech.py). `composite_topopt_reduction`
    inverts this exactly before summing/maxing, so the reported
    objectives are still the true linear-space compliance values, not
    an approximation.

    Returns:
        A tuple `(raw_response, bounds)`:
            raw_response: callable mapping an `n x 1152`-dim tensor (raw
                per-element densities in `[0, 1]`) to an `n x 48`-dim
                tensor of `log(block-summed compliance + 1e-6)`.
            bounds: `2 x 1152`-dim tensor, `[0, 1]` per element.
    """
    def raw_response(X_input: Tensor) -> Tensor:
        X_flat = X_input.reshape(-1, TOPOPT_DIM)
        rows = []
        for row in X_flat.detach().cpu().numpy():
            x_density = np.clip(row, 0.0, 1.0).reshape(NELY, NELX, order="F")
            ce_grid = _solve_topopt(x_density)
            rows.append(_blockify(ce_grid))
        out = torch.tensor(np.stack(rows), dtype=dtype, device=device)
        return out.view(*X_input.shape[:-1], TOPOPT_RAW_DIM)

    bounds = torch.tensor([[0.0] * TOPOPT_DIM, [1.0] * TOPOPT_DIM], dtype=dtype, device=device)
    return raw_response, bounds


def composite_topopt_reduction(Y_raw: Tensor) -> Tensor:
    r"""Known reduction: total compliance (sum over blocks) and peak
    block compliance (max over blocks) -- see module docstring. `Y_raw`
    is in log-space (see `get_composite_topopt_fn`); this inverts that
    exactly before reducing, so the returned objectives are the true
    linear-space compliance values.

    Args:
        Y_raw: `... x 48`-dim tensor of `log(block-summed compliance +
            1e-6)`, as produced by `raw_response` above.

    Returns:
        An `... x 2`-dim tensor `[-total_compliance, -peak_compliance]`
        (maximize convention: both minimized in the source, negated
        here to match this project's maximize-everything convention).
    """
    blocks = Y_raw.exp() - _LOG_EPS
    total = blocks.sum(dim=-1)
    peak = blocks.amax(dim=-1)
    return torch.stack([-total, -peak], dim=-1)
