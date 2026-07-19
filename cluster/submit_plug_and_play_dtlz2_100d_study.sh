#!/usr/bin/env bash
# Remakes the core tr_shape_dtlz2_100d comparison (DTLZ2, d=100, 600 and
# 2000 evals, 5 seeds) using plug_and_play/run.py -- which is a thin
# wrapper around this repo's REAL, validated morbo engine
# (morbo.run_one_replication -> TurboHParams(tr_shape=...)), the exact
# same code path run_comparison.py uses. This is NOT a simplified
# reimplementation -- results should match experiments/tr_shape_dtlz2_100d's
# recorded numbers up to ordinary floating-point/hardware nondeterminism
# (see plug_and_play/verify_reproduction.py for the direct comparison).
#
# Since this is the identical computation to run_comparison.py's own jobs
# (same engine, same hyperparameters), resource requests mirror
# cluster/run_experiment.sub's established pattern rather than a fresh
# guess.
#
# 7 methods x 2 budgets x 5 seeds = 70 jobs.
#
# Usage: bash cluster/submit_plug_and_play_dtlz2_100d_study.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

submit() {
  local method=$1 seed=$2 budget=$3
  sbatch --requeue \
    --job-name="pnp-dtlz2100d-${method}-b${budget}-s${seed}" \
    --output="cluster/logs/pnp-dtlz2100d-${method}-b${budget}-s${seed}_%j.out" \
    --error="cluster/logs/pnp-dtlz2100d-${method}-b${budget}-s${seed}_%j.err" \
    --partition=aimi --account=kilian \
    --cpus-per-task=32 --mem=128g --gres=gpu:1 --time=08:00:00 \
    --parsable \
    --wrap="cd $(pwd)/plug_and_play; . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python run_and_save.py --study dtlz2_100d --benchmark dtlz2 --dim 100 --method $method --seed $seed --budget $budget --n-initial-points 200 --batch-size 50 --benchmark-kwargs '{\"num_objectives\": 2}'"
}

submit_plot() {
  local budget=$1; shift
  local deps; deps=$(IFS=:; echo "$*")
  sbatch --requeue \
    --job-name="pnp-plot-dtlz2100d-b${budget}" \
    --dependency=afterok:"$deps" \
    --partition=aimi --account=kilian \
    --cpus-per-task=1 --mem=4g --time=00:15:00 \
    --output="cluster/logs/pnp-plot-dtlz2100d-b${budget}_%j.out" \
    --wrap="cd $(pwd)/plug_and_play; . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_study.py dtlz2_100d $budget"
}

SEEDS=(0 1 2 3 4)
METHODS=(isotropic ard_box pca_ellipsoid ard_pca_ellipsoid cma_ellipsoid labcat_style mab_shape)
BUDGETS=(600 2000)

for BUDGET in "${BUDGETS[@]}"; do
  echo "=== budget=$BUDGET ==="
  IDS=()
  for METHOD in "${METHODS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      J=$(submit "$METHOD" "$SEED" "$BUDGET"); IDS+=("$J")
    done
  done
  echo "  submitted ${#IDS[@]} jobs"
  submit_plot "$BUDGET" "${IDS[@]}"
done

echo
echo "Done (${#METHODS[@]} methods x ${#BUDGETS[@]} budgets x ${#SEEDS[@]} seeds = $((${#METHODS[@]} * ${#BUDGETS[@]} * ${#SEEDS[@]})) jobs)."
echo "Check with: squeue -u \$USER"
echo "Plots land in plug_and_play/results/dtlz2_100d/<budget>ev/comparison.png"
