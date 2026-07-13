#!/usr/bin/env bash
# DTLZ landscape variants at d=100: does the DTLZ2 shape-adaptation win
# generalize across landscape characters at the same nominal dimension, or
# is it specific to DTLZ2's smooth unimodal geometry?
#   DTLZ1 -- multimodal (11^k local fronts), linear front
#   DTLZ3 -- DTLZ2's spherical front geometry + severe multimodality
#   DTLZ5 -- degenerate (curve-shaped) front
#   DTLZ7 -- disconnected front regions
# All share DTLZ2's low-effective-dimension structure (position dims +
# scalar g over distance dims), so the effective-dimension mechanism
# predicts shape adaptation should still help; the landscape character is
# the variable under test.
#
# Fresh experiment dirs with CORRECT evalfns -- the legacy dtlz5_m2/dtlz7_m2
# dirs have a known evalfn swap bug (see README "Known pre-existing bug").
# 600 evals / batch 50 matching the main tr_shape sweep; the 4 core
# methods; seeds 0-4. 4 problems x 5 seeds x 4 methods = 80 jobs.
#
# Usage: bash cluster/submit_dtlz_variants.sh
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

submit_plot() {
  local exp=$1; shift
  local deps; deps=$(IFS=:; echo "$*")
  sbatch --requeue \
    --job-name="plot-${exp}" \
    --dependency=afterok:"$deps" \
    --partition=aimi --account=kilian \
    --cpus-per-task=1 --mem=4g --time=00:15:00 \
    --output="cluster/logs/plot-${exp}_%j.out" \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_comparison.py $exp 0"
}

SEEDS=(0 1 2 3 4)
METHODS=(morbo ard_box pca_ellipsoid ard_pca_ellipsoid)
EXPERIMENTS=(tr_shape_dtlz1_100d tr_shape_dtlz3_100d tr_shape_dtlz5_100d tr_shape_dtlz7_100d)

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=== $EXP ==="
  IDS=()
  for SEED in "${SEEDS[@]}"; do
    for LABEL in "${METHODS[@]}"; do
      J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
    done
  done
  echo "  submitted ${#IDS[@]} jobs"
  submit_plot "$EXP" "${IDS[@]}"
done

echo
echo "Done (80 jobs total). Check with: squeue -u \$USER"
echo "Aggregate with: python aggregate_seeds.py <experiment_name>"
