#!/usr/bin/env bash
# Remakes the core tr_shape_dtlz2_100d comparison (DTLZ2, d=100, 600 and
# 2000 evals, 5 seeds) using the NEW self-contained plug_and_play/
# implementation (optimizer.py -- a single-trust-region engine, not this
# repo's full coordinated multi-region MORBO) instead of run_comparison.py.
#
# Honesty note, load-bearing for how to read the results: this is a
# DIFFERENT (deliberately simpler) algorithm than the one that produced
# experiments/tr_shape_dtlz2_100d's recorded numbers -- don't expect the
# absolute hypervolume values to match. What this DOES test is whether
# the same QUALITATIVE ranking (pca_ellipsoid/ard_pca_ellipsoid beating
# isotropic, ard_box's known failure mode, etc.) still emerges from the
# same shape functions (methods.py) plugged into a much smaller reference
# BO loop. See plug_and_play/optimizer.py's module docstring.
#
# 7 methods x 2 budgets x 5 seeds = 70 jobs, no GPU needed (each run took
# ~12s/~1min locally at 600/2000 evals on CPU alone -- this is cheap
# compute, not a reason to skip parallelizing across jobs).
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
    --cpus-per-task=8 --mem=16g --time=00:30:00 \
    --parsable \
    --wrap="cd $(pwd)/plug_and_play; . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python run_and_save.py --study dtlz2_100d --benchmark dtlz2 --dim 100 --method $method --seed $seed --budget $budget --n-init 200 --batch-size 50 --benchmark-kwargs '{\"num_objectives\": 2}'"
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
