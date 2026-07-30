#!/usr/bin/env python3
r"""Persistent worker process, run inside the isolated `.fairchem_env`
(a separate venv with its own, older, incompatible torch/fairchem-core
pin -- see `composite_oc20.py`'s module docstring for why this can't
just be imported in-process). Reads one JSON request per line from
stdin, runs one ML-relaxation black-box evaluation with GemNet-OC
(`fairchem-core`, a real pretrained OC20 potential, no gated
authentication required -- see module docstring), writes one JSON
response per line to stdout, and loops -- avoiding the cost of
reloading the ~39M-parameter model for every single design point.

Not imported by anything in the main environment; invoked only via
`subprocess.Popen([".fairchem_env/Scripts/python.exe", <this file>], ...)`
from `composite_oc20.py`.
"""
import json
import sys

import numpy as np
from ase import Atoms
from ase.build import add_adsorbate, fcc111
from ase.optimize import BFGS
from ase.constraints import FixAtoms

from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from fairchem.core.models.model_registry import model_name_to_local_file

SLAB_SIZE = (3, 3, 4)
ADSORBATE = "O"
N_RELAX_STEPS = 15
N_CHECKPOINTS = 5
PERTURBATION_BOUND_ANGSTROM = 0.5


def _build_reference_slab():
    slab = fcc111("Cu", size=SLAB_SIZE, vacuum=10.0)
    add_adsorbate(slab, ADSORBATE, height=1.5, position="fcc")
    mobile_mask = [atom.tag <= 2 for atom in slab]  # top 2 layers + adsorbate
    fixed_mask = [not m for m in mobile_mask]
    mobile_indices = [i for i, m in enumerate(mobile_mask) if m]
    return slab, fixed_mask, mobile_indices


def main() -> None:
    ckpt = model_name_to_local_file("GemNet-OC-S2EF-OC20-2M", local_cache=".fairchem_cache/")
    calc = OCPCalculator(checkpoint_path=ckpt, cpu=True)

    ref_slab, fixed_mask, mobile_indices = _build_reference_slab()
    ref_positions = ref_slab.get_positions().copy()
    n_mobile = len(mobile_indices)

    bare_slab = fcc111("Cu", size=SLAB_SIZE, vacuum=10.0)
    bare_slab.calc = calc
    e_slab_bare = bare_slab.get_potential_energy()

    print(json.dumps({"ready": True, "dim": 3 * n_mobile, "e_slab_bare": e_slab_bare}), flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("cmd") == "shutdown":
            break

        x = np.array(req["x"], dtype=float).reshape(n_mobile, 3)
        atoms = ref_slab.copy()
        positions = ref_positions.copy()
        positions[mobile_indices] += x * PERTURBATION_BOUND_ANGSTROM
        atoms.set_positions(positions)
        atoms.set_constraint(FixAtoms(mask=fixed_mask))
        atoms.calc = calc

        energies = []
        max_forces = []
        try:
            opt = BFGS(atoms, logfile=None)
            checkpoint_every = max(1, N_RELAX_STEPS // N_CHECKPOINTS)
            for step in range(N_RELAX_STEPS):
                opt.step()
                if (step + 1) % checkpoint_every == 0:
                    e = atoms.get_potential_energy()
                    f = atoms.get_forces()
                    fmax = float(np.abs(f[mobile_indices]).max())
                    energies.append(float(e))
                    max_forces.append(fmax)
            while len(energies) < N_CHECKPOINTS:
                energies.append(energies[-1] if energies else float(atoms.get_potential_energy()))
                max_forces.append(max_forces[-1] if max_forces else 0.0)
            ok = True
        except Exception as exc:  # noqa: BLE001 -- report failure to caller, don't crash the worker
            energies = [float(e_slab_bare)] * N_CHECKPOINTS
            max_forces = [100.0] * N_CHECKPOINTS
            ok = False

        print(json.dumps({
            "ok": ok,
            "energies": energies[:N_CHECKPOINTS],
            "max_forces": max_forces[:N_CHECKPOINTS],
        }), flush=True)


if __name__ == "__main__":
    main()
