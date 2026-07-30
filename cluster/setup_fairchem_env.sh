#!/usr/bin/env bash
# Builds the isolated venv composite_oc20.py needs to drive a real
# GemNet-OC (fairchem-core) potential without touching this project's
# own pinned torch==2.12.0. See morbo/problems/composite_oc20.py's
# module docstring for the full story of why this has to be a separate
# environment (fairchem-core's dependency chain pins an incompatible
# torch, discovered the hard way: installing it directly into the main
# env silently downgraded torch and was reverted).
#
# fairchem-core==1.10.0 is used deliberately, not the latest release:
# newer versions (>=2.1) moved to a gated `facebook/UMA` checkpoint
# requiring a HuggingFace account and accepting Meta's license
# agreement -- neither of which this script can do on the user's
# behalf. 1.10.0 still has the old `OCPCalculator` class and the
# `model_name_to_local_file` public model registry, which downloads the
# `GemNet-OC-S2EF-OC20-2M` checkpoint with no authentication at all
# (verified directly).
#
# Usage: bash cluster/setup_fairchem_env.sh
set -euo pipefail
cd "$(dirname "$0")/.."

ENV_DIR=.fairchem_env
CACHE_DIR=.fairchem_cache

python3 -m venv "$ENV_DIR"
PY="$ENV_DIR/bin/python"
# Windows venvs put the interpreter under Scripts/, not bin/
[ -f "$PY" ] || PY="$ENV_DIR/Scripts/python.exe"

"$PY" -m pip install --upgrade pip -q
"$PY" -m pip install "fairchem-core==1.10.0" ase -q

# torch_scatter/torch_sparse/torch_cluster: compiled extensions matched
# to fairchem-core==1.10.0's own torch pin (2.4.1). PyG's wheel index
# serves both Linux and Windows builds from the same URL.
"$PY" -m pip install torch_scatter torch_sparse torch_cluster \
  -f https://data.pyg.org/whl/torch-2.4.1+cpu.html -q

# fairchem 1.10.0's basis-function code calls scipy.special.sph_harm,
# renamed/removed in scipy>=1.15.
"$PY" -m pip install "scipy<1.15" -q

# Downloads the (no-auth-required) GemNet-OC checkpoint into $CACHE_DIR
# and verifies the worker script's own model-loading path end to end.
"$PY" -c "
from fairchem.core.models.model_registry import model_name_to_local_file
path = model_name_to_local_file('GemNet-OC-S2EF-OC20-2M', local_cache='$CACHE_DIR/')
print('Checkpoint ready at', path)
"

echo "fairchem environment ready at $ENV_DIR"
echo "Smoke test: echo '{\"x\": $(python3 -c 'print([0]*57)')}' | $PY morbo/problems/oc20_fairchem_worker.py"
