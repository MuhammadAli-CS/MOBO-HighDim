#!/usr/bin/env bash
# Same tau315/composite-mobo ablation as cluster/submit_tau_repo_full.sh
# (composite_ablation/run_from_tau_repo.py: standard_mobo/qEHVI,
# chebyshev_bo, batched_morbo/morbo -- each direct vs. composite),
# applied to this project's own SnAr plug-flow-reactor benchmark
# (morbo/problems/composite_snar.py, d=4, M=2). Low-suite only (d=4<=10,
# not a spherical-kernel benchmark): standard, chebyshev, morbo.
#
# This is a separate ablation from cluster/submit_snar_2x3.sh, which
# tests composite-modeling x trust-region-shape-adaptation for just the
# morbo family. This script instead tests direct-vs-composite across
# multiple SOLVER FAMILIES (qEHVI, chebyshev scalarization, morbo) with
# isotropic trust regions only -- no PCA/ARD-PCA shape variants here.
#
# 3 jobs total. dim=4<=10 keeps the unscaled 45-eval default
# (see run_from_tau_repo.py's _eval_budget).
#
# Usage: bash cluster/submit_tau_repo_snar.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs composite_ablation/results_tau_repo

TRIALS=20

submit() {
  local pair=$1
  sbatch --requeue \
    --job-name="tau-repo-snar_4d_2obj_ours-${pair}" \
    --output="cluster/logs/tau-repo-snar_4d_2obj_ours-${pair}_%j.out" \
    --error="cluster/logs/tau-repo-snar_4d_2obj_ours-${pair}_%j.err" \
    --partition=aimi --account=kilian \
    --cpus-per-task=32 --mem=64g --time=0 \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python -m composite_ablation.run_from_tau_repo --benchmark snar_4d_2obj_ours --pair ${pair} --trials $TRIALS --num-threads 8 --num-interop-threads 4 --out-dir composite_ablation/results_tau_repo/snar_4d_2obj_ours"
}

submit standard
submit chebyshev
submit morbo

echo "Submitted 3 jobs. Check with: squeue -u \$USER"
echo "Per-job console logs: cluster/logs/tau-repo-snar_4d_2obj_ours-<pair>_<jobid>.out"
echo "Raw HV traces + per-pair summary: composite_ablation/results_tau_repo/snar_4d_2obj_ours/"
