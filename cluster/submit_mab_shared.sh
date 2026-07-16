#!/usr/bin/env bash
# mab_shape_ducb_shared: D-UCB bandit whose CMA covariance advances at
# EVERY shape update regardless of which arm is played (the cma arm merely
# consumes the shared state). Targets the one regime the plain ducb bandit
# could not fix (RESULTS.md sec 11g): at d=200/600ev, arm-switching starves
# cma_ellipsoid's covariance of updates, and only full-rate CMA breaks
# through there. Prediction: with sharing, the bandit should recover at
# least part of cma's 21.72 at d=200 while keeping ducb's strong d=100 and
# tv_keff49 behavior (sharing shouldn't hurt those -- the cma arm just gets
# a better-maintained state).
#
#   tr_shape_methods_dtlz2_200d (5 seeds)  -- THE test this variant exists for
#   tr_shape_methods_dtlz2_100d (20 seeds) -- regression check vs plain ducb (31.41+/-3.50)
#   tv_sparse_dtlz2_d100_keff49 (20 seeds) -- regression check vs plain ducb (+9.3%, 16/20)
#
# 45 jobs. Usage: bash cluster/submit_mab_shared.sh
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
    --job-name="plot-${exp}-shared" \
    --dependency=afterok:"$deps" \
    --partition=aimi --account=kilian \
    --cpus-per-task=1 --mem=4g --time=00:15:00 \
    --output="cluster/logs/plot-${exp}-shared_%j.out" \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_aggregate.py $exp"
}

for SPEC in "tr_shape_methods_dtlz2_200d 5" "tr_shape_methods_dtlz2_100d 20" "tv_sparse_dtlz2_d100_keff49 20"; do
  EXP=${SPEC% *}; NSEEDS=${SPEC#* }
  echo "=== $EXP (${NSEEDS} seeds) ==="
  IDS=()
  for ((SEED=0; SEED<NSEEDS; SEED++)); do
    J=$(submit "$EXP" "mab_shape_ducb_shared" "$SEED"); IDS+=("$J")
  done
  echo "  submitted ${#IDS[@]} jobs"
  submit_plot "$EXP" "${IDS[@]}"
done

echo
echo "Done (45 jobs). Check with: squeue -u \$USER"
