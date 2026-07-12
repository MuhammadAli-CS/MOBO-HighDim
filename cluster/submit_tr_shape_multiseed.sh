#!/usr/bin/env bash
# Multi-seed robustness sweep. Everything reported so far is seed 0 only;
# the headline high-d effects (+64-72%) are far outside single-seed noise,
# but the small-d and Rover effects (+/-3-10%) are not, so those claims need
# seeds to be defensible. This reruns the established experiments across
# seeds 1-4 (seed 0 already committed) for the four core methods.
#
# After these land: `python aggregate_seeds.py <exp>` prints mean +/- std.
#
# NOTE: this is the largest single submission in the project -- 4 experiments
# x 4 methods x 4 seeds = 64 jobs (d=100 DTLZ2 ~a few min each on a B200;
# Rover 2000-eval jobs are the slow ones). Submit when the aimi partition has
# headroom (check `sinfo -p aimi`); all land in the priority kilian account.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

submit() {
  local exp=$1 label=$2 seed=$3
  sbatch --requeue \
    --job-name="${label}-${exp}-s${seed}" \
    --export=EXP="$exp",LABEL="$label",SEED="$seed" \
    --parsable \
    cluster/run_experiment.sub
}

SEEDS=(1 2 3 4)
METHODS=(morbo ard_box pca_ellipsoid ard_pca_ellipsoid)
EXPERIMENTS=(tr_shape_dtlz2_50d tr_shape_dtlz2_100d tr_shape_dtlz2_150d tr_shape_rover)

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=== $EXP ==="
  for SEED in "${SEEDS[@]}"; do
    for LABEL in "${METHODS[@]}"; do
      J=$(submit "$EXP" "$LABEL" "$SEED")
      echo "  ${LABEL} seed=${SEED} -> $J"
    done
  done
done

echo
echo "All jobs submitted (${#EXPERIMENTS[@]} exps x ${#METHODS[@]} methods x ${#SEEDS[@]} seeds)."
echo "Check with: squeue -u \$USER"
echo "Aggregate once done: python aggregate_seeds.py <experiment_name>"
