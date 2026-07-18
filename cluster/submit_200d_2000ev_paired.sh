#!/usr/bin/env bash
# Targeted budget extension #3: tr_shape_dtlz2_200d_2000ev already has
# labcat_style at 5 seeds (from submit_labcat_style.sh), but morbo/
# pca_ellipsoid/ard_pca_ellipsoid/ard_box only have seed 0 (this
# experiment predates the labcat_style ablation, originally a single-seed
# companion to the d=150/2000ev point). Fills in seeds 1-4 for those four
# labels so there's a proper paired 5-seed comparison at the budget where
# the base d=200 collapse (RESULTS.md §3/§4: baseline morbo=0.00 at 600
# evals) is known to partially resolve.
#
# 4 methods x 4 seeds (1-4; seed 0 already exists) = 16 jobs.
#
# Usage: bash cluster/submit_200d_2000ev_paired.sh
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

EXP=tr_shape_dtlz2_200d_2000ev
SEEDS=(1 2 3 4)
LABELS=(morbo pca_ellipsoid ard_pca_ellipsoid ard_box)

echo "Submitting $EXP jobs (dim=200, 2000 evals, batch 50, seeds 1-4)..."
IDS=()
for SEED in "${SEEDS[@]}"; do
  for LABEL in "${LABELS[@]}"; do
    J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
  done
done

deps=$(IFS=:; echo "${IDS[*]}")
sbatch --requeue \
  --job-name="plot-${EXP}" \
  --dependency=afterok:"$deps" \
  --partition=aimi --account=kilian \
  --cpus-per-task=1 --mem=4g --time=00:15:00 \
  --output="cluster/logs/plot-${EXP}_%j.out" \
  --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_aggregate.py $EXP"

echo
echo "Done (${#LABELS[@]} methods x ${#SEEDS[@]} new seeds = $((${#LABELS[@]} * ${#SEEDS[@]})) jobs)."
echo "Check with: squeue -u \$USER"
