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

# Only needs to run once per account; harmless to re-run.
/share/apps/software/anaconda3/bin/conda init bash || true
source ~/.bashrc

if conda env list | grep -q "$ENV_PATH"; then
  echo "Env already exists at $ENV_PATH — skipping creation."
else
  conda create -y -p "$ENV_PATH" python=3.11
fi

conda activate "$ENV_PATH"

# Versions pinned to match what's validated locally (see README.md "Fork
# notes" section) so cluster runs are numerically comparable to laptop runs.
pip install --upgrade pip
pip install "torch==2.12.*" --index-url https://download.pytorch.org/whl/cu121
pip install "gpytorch==1.11" "botorch==0.9.5" scipy matplotlib jupyter anthropic

# Install this repo (morbo/, botier_llm/, run_comparison.py, etc.) in editable mode.
cd "$(dirname "$0")/.."
pip install -e .

echo "Done. Activate with: conda activate $ENV_PATH"
