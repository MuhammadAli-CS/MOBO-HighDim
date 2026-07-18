#!/usr/bin/env bash
# Targeted budget extension #2 (of 2): tr_shape_dtlz2_150d_2000ev already
# has labcat_style at 5 seeds (from submit_labcat_style.sh's Group 1), and
# the seed-0-only data already shows a striking pattern -- labcat_style's
# well-documented d=150/600-eval collapse (RESULTS.md §13: exactly 0.0 HV
# in 3/5 seeds) fully recovers at 2000 evals, landing close to
# ard_pca_ellipsoid (labcat_style ~33-34 across its 5 seeds vs.
# ard_pca_ellipsoid's single seed-0 value of 34.26). But morbo/
# pca_ellipsoid/ard_pca_ellipsoid only have seed 0 here (this experiment
# was originally a single-seed extended-budget companion to
# tr_shape_dtlz2_200d_2000ev, predating the labcat_style ablation), so
# there's no paired multi-seed comparison yet -- this fills in seeds 1-4
# for those three labels to make one.
#
# Answers directly: was labcat_style's d=150 failure a fundamental flaw in
# the whiten-then-PCA construction, or a budget-starvation artifact that
# affects everyone similarly at d=150 (morbo/pca_ellipsoid/ard_pca_ellipsoid
# all also improve substantially from their own 600-eval d=150 numbers)?
#
# 3 methods x 4 seeds (1-4; seed 0 already exists) = 12 jobs.
#
# Usage: bash cluster/submit_150d_2000ev_paired.sh
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

EXP=tr_shape_dtlz2_150d_2000ev
SEEDS=(1 2 3 4)
LABELS=(morbo pca_ellipsoid ard_pca_ellipsoid)

echo "Submitting $EXP jobs (dim=150, 2000 evals, batch 50, seeds 1-4)..."
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
echo "Compare against labcat_style (already 5 seeds here) with:"
echo "  python compute_significance.py tr_shape_dtlz2_150d_2000ev labcat_style --baseline-label pca_ellipsoid"
