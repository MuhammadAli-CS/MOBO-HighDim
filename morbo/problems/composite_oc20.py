#!/usr/bin/env python3
r"""A composite-structure, 57-dimensional real-world benchmark: ML-surrogate
catalyst-adsorbate relaxation on the Open Catalyst 2020 (OC20) task family
\citep{chanussot2021oc20}, using a real pretrained GemNet-OC potential
(\texttt{fairchem-core}, formerly \texttt{ocp}) as a live, open,
no-authentication-required proxy for the true DFT relaxation this task is
normally defined against (see methods.tex \S\ref{sec:oc20} for why a live
DFT oracle itself -- VASP, Quantum ESPRESSO -- was ruled out as too heavy a
dependency, and why the newer, gated \texttt{facebook/UMA} checkpoint was
also ruled out: it requires a HuggingFace account and accepting Meta's
license agreement, neither of which this process can do on the user's
behalf. The older \texttt{GemNet-OC-S2EF-OC20-2M} checkpoint, downloaded
via \texttt{fairchem.core.models.model_registry.model_name_to_local_file},
requires no authentication at all -- verified directly).

Why an isolated environment, not a normal import: \texttt{fairchem-core}'s
dependency chain (an old \texttt{torch\_scatter}/\texttt{torch\_sparse}/
\texttt{torch\_geometric} stack, itself requiring an older
\texttt{fairchem-core==1.10.0} release to have a loadable
\texttt{OCPCalculator} for this checkpoint format at all -- newer releases
moved to a different, gated checkpoint schema) pins \texttt{torch==2.4.1},
incompatible with this project's own pinned \texttt{torch==2.12.0}.
Installing \texttt{fairchem-core} into the main environment was tried
directly first and silently downgraded the project's own torch to 2.8.0 --
a real, verified risk to every other composite benchmark in this
repository, caught immediately and reverted (see methods.tex's OC20/OC22
"Update, now checked" paragraph). This module instead drives a
\emph{separate} venv (\texttt{.fairchem\_env/}, gitignored, not part of
this project's own dependency set) via a persistent subprocess worker
(\texttt{oc20\_fairchem\_worker.py}, which runs \emph{inside} that venv and
is never imported here) communicating over stdin/stdout JSON lines -- the
model is loaded once per worker process lifetime, not once per evaluation.

The real "IS2RS" task this mirrors (initial structure -> DFT-relaxed
structure, \S\ref{sec:oc20}): a fixed Cu(111) slab (3x3x4 supercell, 36
atoms) plus one adsorbed O atom (37 atoms total), with the bottom two
layers held fixed (standard surface-relaxation practice) and the top two
layers plus the adsorbate (19 atoms, 57 real-valued coordinates) free to
be perturbed and then relaxed by the ML potential via ASE's BFGS
optimizer. Unlike this project's exact-mechanism composite benchmarks
(GRI-Mech, RCM40), the reduction here is over an \emph{approximate}
adsorption-energy proxy: the true adsorption-energy formula needs a
reference isolated-adsorbate energy subtracted alongside the slab
reference, but a single free O atom (and an O2 molecule, both tried) are
degenerate edge cases for this graph-neural-network potential's edge
construction and raised a runtime error rather than returning a value --
so only the bare-slab reference (computed successfully, verified
directly) is subtracted, giving a well-defined \emph{relative} energy for
comparing different perturbations of the same fixed slab+adsorbate
system, not a reference-corrected adsorption energy matching published
DFT tables. This is a real, documented simplification, not silently
assumed to be the textbook quantity.
"""
import atexit
import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import Tensor

OC20_MOBILE_ATOMS = 19
OC20_DIM = 3 * OC20_MOBILE_ATOMS  # 57
OC20_N_CHECKPOINTS = 5
# Per checkpoint: 1 energy + 4 order statistics (max, mean, std, min) of the
# raw per-atom force-magnitude vector the worker returns. Using order
# statistics of the *real* per-atom vector, rather than the worker
# pre-collapsing to a single max (the original design), is the actual fix:
# it's real distributional signal, not fabricated. The full 19-component
# per-atom vector was tried first (100-dim raw response total) and reliably
# crashed GP fitting (a Windows access violation inside
# gpytorch/linear_operator's Cholesky routine during Thompson-sampling
# posterior draws, confirmed via `faulthandler` -- likely the joint
# posterior covariance across 100 independent-output GPs times
# `raw_samples=4096` discrete TS candidates becoming too large for this
# library/platform to handle, well past GRI-Mech's 24 and RCM40's 29 raw
# dims, both of which run fine). 25 raw dims keeps the real fix (no more
# scalar pre-reduction inside the black box) while staying in the range
# this project's other composite benchmarks have already verified works.
OC20_N_FORCE_STATS = 4
OC20_RAW_DIM = OC20_N_CHECKPOINTS * (1 + OC20_N_FORCE_STATS)  # 25

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Linux venvs/conda envs put the interpreter under bin/; Windows under
# Scripts/ (see cluster/setup_fairchem_env.sh, which builds either layout
# depending on platform).
_FAIRCHEM_PYTHON = _REPO_ROOT / ".fairchem_env" / "bin" / "python"
if not _FAIRCHEM_PYTHON.exists():
    _FAIRCHEM_PYTHON = _REPO_ROOT / ".fairchem_env" / "Scripts" / "python.exe"
_WORKER_SCRIPT = Path(__file__).resolve().parent / "oc20_fairchem_worker.py"

_worker_process = None
_e_slab_bare = None


def _ensure_worker():
    global _worker_process, _e_slab_bare
    if _worker_process is not None and _worker_process.poll() is None:
        return _worker_process
    if not _FAIRCHEM_PYTHON.exists():
        raise RuntimeError(
            f"Isolated fairchem environment not found at {_FAIRCHEM_PYTHON}. "
            "This composite benchmark requires a separate venv with "
            "fairchem-core==1.10.0 (see composite_oc20.py's module docstring "
            "for why it can't share this project's own torch pin)."
        )
    _worker_process = subprocess.Popen(
        [str(_FAIRCHEM_PYTHON), str(_WORKER_SCRIPT)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        cwd=str(_REPO_ROOT),
        text=True,
        bufsize=1,
    )
    ready_line = _worker_process.stdout.readline()
    ready = json.loads(ready_line)
    assert ready.get("ready") and ready["dim"] == OC20_DIM, f"unexpected worker handshake: {ready}"
    _e_slab_bare = ready["e_slab_bare"]

    def _cleanup():
        if _worker_process is not None and _worker_process.poll() is None:
            try:
                _worker_process.stdin.write(json.dumps({"cmd": "shutdown"}) + "\n")
                _worker_process.stdin.flush()
            except Exception:  # noqa: BLE001
                pass
            _worker_process.terminate()

    atexit.register(_cleanup)
    return _worker_process


def get_composite_oc20_fn(dtype=torch.double, device=None) -> Tuple[callable, Tensor]:
    r"""Construct the raw-response (checkpointed relaxation trajectory)
    function and bounds for composite OC20 catalyst relaxation.

    Returns:
        A tuple `(raw_response, bounds)`:
            raw_response: callable mapping an `n x 57`-dim tensor (raw,
                *unnormalized* per-mobile-atom-coordinate perturbations,
                each in `[-1, 1]`, scaled internally by the worker to
                `+-0.5` Angstrom) to an `n x 25`-dim tensor
                `[energy_1..energy_5, stats_1[4]..stats_5[4]]` -- 5
                checkpoints along the 15-step BFGS relaxation, each an
                energy scalar plus `[max, mean, std, min]` of that
                checkpoint's raw per-mobile-atom force-magnitude vector
                (the worker itself only ever returns the raw, un-reduced
                per-atom vector -- these order statistics are computed
                here, not inside the worker; see
                `oc20_fairchem_worker.py`'s module docstring and this
                module's `OC20_RAW_DIM` comment for why order statistics
                rather than the full per-atom vector).
            bounds: `2 x 57`-dim tensor, `[-1, 1]` per dimension.
    """
    proc = _ensure_worker()

    def raw_response(X_input: Tensor) -> Tensor:
        X_flat = X_input.reshape(-1, OC20_DIM)
        rows = []
        for row in X_flat.detach().cpu().tolist():
            proc.stdin.write(json.dumps({"x": row}) + "\n")
            proc.stdin.flush()
            resp = json.loads(proc.stdout.readline())
            stats = []
            for checkpoint_forces in resp["atom_forces"]:
                f = torch.tensor(checkpoint_forces, dtype=dtype)
                stats.extend([f.max().item(), f.mean().item(), f.std().item(), f.min().item()])
            rows.append(resp["energies"] + stats)
        out = torch.tensor(rows, dtype=dtype, device=device)
        return out.view(*X_input.shape[:-1], OC20_RAW_DIM)

    bounds = torch.tensor([[-1.0] * OC20_DIM, [1.0] * OC20_DIM], dtype=dtype, device=device)
    return raw_response, bounds


def composite_oc20_reduction(Y_raw: Tensor) -> Tensor:
    r"""Known reduction: final-checkpoint relative adsorption energy
    (bare-slab-referenced, see module docstring) and final-checkpoint
    residual force (the max-force order statistic at that checkpoint --
    the actual convergence criterion ASE/DFT relaxations use, one of the
    4 order statistics `raw_response` computes from the worker's raw
    per-atom force vector).

    Args:
        Y_raw: `... x 25`-dim tensor
            `[energy_1..energy_5, stats_1[4]..stats_5[4]]` (each
            `stats_i = [max, mean, std, min]`), as produced by
            `raw_response` above.

    Returns:
        An `... x 2`-dim tensor `[-E_rel, -F_res]` (maximize convention:
        more negative relative energy is more stable, so `-E_rel` is
        maximized; smaller residual force is better convergence, so
        `-F_res` is maximized).
    """
    _ensure_worker()
    energies = Y_raw[..., 0:OC20_N_CHECKPOINTS]
    stats = Y_raw[..., OC20_N_CHECKPOINTS:].view(
        *Y_raw.shape[:-1], OC20_N_CHECKPOINTS, OC20_N_FORCE_STATS
    )
    max_forces = stats[..., 0]  # [max, mean, std, min] -> index 0 is max

    e_final = energies[..., -1]
    f_final = max_forces[..., -1]
    e_rel = e_final - _e_slab_bare
    return torch.stack([-e_rel, -f_final], dim=-1)
