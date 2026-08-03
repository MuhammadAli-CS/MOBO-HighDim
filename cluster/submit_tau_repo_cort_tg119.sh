#!/usr/bin/env bash
# Runs tau315/composite-mobo's CORT TG119 radiotherapy benchmark (real
# clinical dose-influence data, IMRT beamlet-intensity optimization,
# 418D, 3 objectives, high suite: spherical/morbo only). Split out from
# cluster/submit_tau_repo_new_benchmarks.sh because its own
# benchmark_cort_tg119.py downloads the CORT dataset from GigaDB
# (Wasabi-hosted mirror, ~25.5MB, public) on first use -- confirmed small
# and license-clean before submitting.
#
# Usage: bash cluster/submit_tau_repo_cort_tg119.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs composite_ablation/results_tau_repo

TRIALS=20
BM=cort_tg119_3obj_418d

submit() {
  local pair=$1 mem=$2
  sbatch --requeue \
    --job-name="tau-repo-${BM}-${pair}" \
    --output="cluster/logs/tau-repo-${BM}-${pair}_%j.out" \
    --error="cluster/logs/tau-repo-${BM}-${pair}_%j.err" \
    --partition=aimi --account=kilian \
    --cpus-per-task=32 --mem="$mem" --time=0 \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python -m composite_ablation.run_from_tau_repo --benchmark $BM --pair $pair --trials $TRIALS --num-threads 8 --num-interop-threads 4 --out-dir composite_ablation/results_tau_repo/$BM"
}

submit spherical 128g
submit morbo      128g

echo "Submitted 2 jobs. Check with: squeue -u \$USER"
