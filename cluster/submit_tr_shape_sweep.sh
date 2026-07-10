#!/usr/bin/env bash
# Submits the trust-region shape-adaptation dimension sweep + cross-problem
# check, following up on tr_shape_dtlz2_100d's headline result:
# pca_ellipsoid/ard_pca_ellipsoid won by +64.6%/+66.9% HV at d=100, while
# ard_box (naive per-dimension ARD rescaling, the original TuRBO technique)
# collapsed to -34.6% -- diagnosed as a curse-of-dimensionality effect (real
# fitted lengthscales at d=100 gave axis_lengths a 15.2x ratio, leaving only
# 1/200 locally accumulated points inside ard_box's region, vs 41/200 for
# pca_ellipsoid's data-covariance-derived shape; see morbo/utils.py's
# compute_ard_box_shape/compute_pca_ellipsoid_shape docstrings).
#
# This submits:
#   - A DTLZ2 dimension sweep at FIXED budget (600 evals, batch 50,
#     n_initial=200, min_tr_size=200 -- identical to tr_shape_dtlz2_100d's
#     own scale, so d is the only thing that varies): d in {20, 50, 150, 200}.
#     (d=100 is already done in tr_shape_dtlz2_100d; not resubmitted here.)
#   - tr_shape_rover: a real, non-DTLZ2 benchmark (d=60, Rover navigation,
#     already wired into this repo, existing config reused as-is) to check
#     the finding isn't a DTLZ2-specific artifact.
#
# All 5 experiments need all 4 labels run fresh (morbo, ard_box,
# pca_ellipsoid, ard_pca_ellipsoid) -- none of these configs match any
# existing baseline closely enough to reuse.
#
# Usage: from the repo root on unicorn-login-01 (or an interactive shell):
#   bash cluster/submit_tr_shape_sweep.sh
#
# Requires cluster/setup_env.sh to have been run once already.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

submit() {
  local exp=$1 label=$2 seed=$3
  sbatch --requeue \
    --job-name="${label}-${exp}" \
    --export=EXP="$exp",LABEL="$label",SEED="$seed" \
    --parsable \
    cluster/run_experiment.sub
}

submit_plot() {
  local exp=$1
  shift
  local deps
  deps=$(IFS=:; echo "$*")
  sbatch --requeue \
    --job-name="plot-${exp}" \
    --dependency=afterok:"$deps" \
    --partition=aimi --account=kilian \
    --cpus-per-task=1 --mem=4g --time=00:15:00 \
    --output="cluster/logs/plot-${exp}_%j.out" \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_comparison.py $exp 0"
}

LABELS=(morbo ard_box pca_ellipsoid ard_pca_ellipsoid)

for EXP in tr_shape_dtlz2_20d tr_shape_dtlz2_50d tr_shape_dtlz2_150d tr_shape_dtlz2_200d tr_shape_rover; do
  echo "Submitting $EXP jobs..."
  JOB_IDS=()
  for LABEL in "${LABELS[@]}"; do
    J=$(submit "$EXP" "$LABEL" 0)
    JOB_IDS+=("$J")
    echo "  $LABEL=$J"
  done
  submit_plot "$EXP" "${JOB_IDS[@]}"
done

echo "All jobs submitted. Check with: squeue -u \$USER"
echo "5 experiments x 4 labels = 20 experiment jobs, plus 5 dependent plot jobs."
echo "tr_shape_rover uses 2000 evals (its own existing config) -- expect it to"
echo "take longer than the 600-eval DTLZ2 sweep jobs."
