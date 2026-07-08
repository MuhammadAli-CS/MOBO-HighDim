#!/usr/bin/env bash
# One-time environment setup on the Unicorn cluster (it.coecis.cornell.edu).
# Run this from an interactive allocation, NOT the login node — the login
# node kills resource-intensive processes automatically, and building a
# conda env / installing torch counts as one.
#
# Usage (from unicorn-login-01):
#   salloc --mem=8g --cpus-per-task=4 --time=01:00:00 --partition=default_partition-interactive --account=kilian
#   bash cluster/setup_env.sh
set -euo pipefail

ENV_PATH="$HOME/morbo-env"

if [ ! -d "/share/apps/software/anaconda3" ]; then
  echo "WARNING: expected conda install at /share/apps/software/anaconda3 not found." >&2
  echo "Check 'module avail' or the Unicorn docs — path may have changed." >&2
fi

# Only needs to run once per account; harmless to re-run. `conda init bash`
# only takes effect in a NEW interactive shell (it edits ~/.bashrc, which
# most distros guard behind an "if not interactive, return" check near the
# top) -- this script runs as a non-interactive subshell, so `source
# ~/.bashrc` here doesn't actually pull conda's `conda`/`activate` shell
# functions into scope. Source conda's own profile script directly instead,
# which works regardless of interactive/non-interactive shell state.
/share/apps/software/anaconda3/bin/conda init bash || true
source /share/apps/software/anaconda3/etc/profile.d/conda.sh

if conda env list | grep -q "$ENV_PATH"; then
  echo "Env already exists at $ENV_PATH — skipping creation."
else
  conda create -y -p "$ENV_PATH" python=3.11
fi

conda activate "$ENV_PATH"

# torch: unpinned, no custom --index-url. The cu121 build channel is frozen
# at torch<=2.5.1 and, more importantly, CUDA 12.1 predates Blackwell
# (compute capability 10.0) support entirely -- the aimi partition's B200
# GPUs need a current CUDA build, which the default PyPI wheels provide.
#
# botorch/gpytorch: DELIBERATELY pinned to the exact versions this codebase
# was ported against (see README.md "Fork notes" -- the port removed/renamed
# botorch APIs that changed between the archived upstream's target, ~0.6,
# and 0.9.5). "Just use latest" risks botorch having moved its API again
# since 0.9.5, silently breaking the same way the original archived-repo
# port had to fix -- pinning trades a possibly-newer botorch for guaranteed
# API compatibility with the code as actually written.
pip install --upgrade pip
pip install torch
pip install "gpytorch==1.11" "botorch==0.9.5" scipy matplotlib jupyter anthropic

# Install this repo (morbo/, botier_llm/, run_comparison.py, etc.) in editable mode.
cd "$(dirname "$0")/.."
pip install -e .

echo "Done. Activate with: conda activate $ENV_PATH"
