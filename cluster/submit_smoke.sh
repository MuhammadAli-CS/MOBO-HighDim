#!/usr/bin/env bash
# Runs smoke_test_tr_shape.py on the cluster (tiny dim=10 / 40-eval BO loops
# for every new tr_shape variant + linear-kernel + dim-scaled-prior paths).
# Cheap (~a few min on one GPU) and worth running first to confirm every new
# code path completes end-to-end on the cluster's environment before
# committing to the full multi-hour sweeps -- catches env/version issues and
# the "shape update silently never fires" class of bug.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

sbatch --requeue \
  --job-name=smoke-tr-shape \
  --partition=aimi --account=kilian \
  --cpus-per-task=8 --mem=32g --gres=gpu:1 --time=00:30:00 \
  --output=cluster/logs/smoke-tr-shape_%j.out \
  --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python smoke_test_tr_shape.py"

echo "Smoke-test job submitted. Check cluster/logs/smoke-tr-shape_*.out for"
echo "'All smoke tests passed.' before running the full sweeps."
