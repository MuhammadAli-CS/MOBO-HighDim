#!/usr/bin/env bash
# Pure random-search baseline: a single Sobol low-discrepancy sequence over
# the whole search space, no trust regions or GP fitting at all (see
# run_one_replication.py's `label == "sobol"` branch). Answers a more basic
# question than any tr_shape variant: is TuRBO/MORBO's local-modeling
# machinery earning its keep at all, on each of these problems?
#
# Reuses the existing tr_shape_dtlz2_{50,100,150,200}d and tr_shape_rover
# experiment dirs (same config.json each already has) -- just adds the
# "sobol" label alongside the methods already there, seeds 0-4 to match the
# multi-seed core sweep (DTLZ2) or seed 0 only where existing baselines are
# also single-seed (nothing here is single-seed except Rover, which already
# has 5 seeds too).
#
# Usage: bash cluster/submit_sobol_baseline.sh
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

SEEDS=(0 1 2 3 4)
EXPERIMENTS=(tr_shape_dtlz2_50d tr_shape_dtlz2_100d tr_shape_dtlz2_150d tr_shape_dtlz2_200d tr_shape_rover)

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=== $EXP ==="
  IDS=()
  for SEED in "${SEEDS[@]}"; do
    J=$(submit "$EXP" "sobol" "$SEED")
    IDS+=("$J")
    echo "  sobol seed=${SEED} -> $J"
  done
  deps=$(IFS=:; echo "${IDS[*]}")
  sbatch --requeue \
    --job-name="plot-${EXP}-sobol" \
    --dependency=afterok:"$deps" \
    --partition=aimi --account=kilian \
    --cpus-per-task=1 --mem=4g --time=00:15:00 \
    --output="cluster/logs/plot-${EXP}-sobol_%j.out" \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_comparison.py $EXP 0"
done

echo
echo "All jobs submitted (${#EXPERIMENTS[@]} experiments x ${#SEEDS[@]} seeds)."
echo "Check with: squeue -u \$USER"
echo "Aggregate with: python aggregate_seeds.py <experiment_name>"
