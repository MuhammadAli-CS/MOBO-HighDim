#!/usr/bin/env bash
# Targeted budget extension #5: tr_shape_dtlz1_100d/600ev is a flat null
# for every method -- "DTLZ1 uninformative (all methods at exactly 0 in
# 600 evals)" (RESULTS.md, methods.tex §7.1's DTLZ landscape variants
# paragraph). The "in 600 evals" qualifier is exactly the kind of hedge
# that should be tested rather than assumed: is DTLZ1 at d=100 genuinely
# unreachable within any reasonable budget (a landscape-difficulty null,
# like Rover/LassoBench), or does everyone eventually break through given
# enough evals, the same pattern already confirmed for the base d=150/200
# DTLZ2 points? Unlike DTLZ2/3/5/7, DTLZ1 is not one of the original
# MORBO paper's own benchmarks (see the initial commit's experiment
# configs) -- this isn't a budget-matching correction, just direct
# verification of an assumption the paper currently states as fact.
#
# New experiment directory (not an overwrite) so the original 600-eval
# point stays intact.
#
# 5 methods (morbo, ard_box, pca_ellipsoid, ard_pca_ellipsoid,
# labcat_style) x 5 seeds = 25 jobs.
#
# Usage: bash cluster/submit_dtlz1_100d_2000ev.sh
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

EXP=tr_shape_dtlz1_100d_2000ev
SEEDS=(0 1 2 3 4)
LABELS=(morbo ard_box pca_ellipsoid ard_pca_ellipsoid labcat_style)

echo "Submitting $EXP jobs (dim=100, 2000 evals, batch 50)..."
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
echo "Done (${#LABELS[@]} methods x ${#SEEDS[@]} seeds = $((${#LABELS[@]} * ${#SEEDS[@]})) jobs)."
echo "Check with: squeue -u \$USER"
