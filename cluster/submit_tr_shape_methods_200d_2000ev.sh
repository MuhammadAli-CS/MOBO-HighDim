#!/usr/bin/env bash
# Targeted budget extension #4: tr_shape_methods_dtlz2_200d/600ev is
# where the mab_shape bandit line's one unsolved regime lives -- RESULTS.md
# §11h diagnoses it as information-theoretic ("the reward signal itself
# carries zero information until something breaks through"), not a fixable
# policy defect, based entirely on 600-eval data where mab_shape_ducb_shared
# scores exactly 0.00 on 4/5 seeds. The base d=200 dimension-sweep point
# (tr_shape_dtlz2_200d) already showed morbo/ard_box/ard_pca_ellipsoid
# collapse similarly at 600 evals and partially recover at 2000
# (tr_shape_dtlz2_200d_2000ev). This is the direct test of whether the
# same is true here: does a breakthrough ever occur with more budget (the
# reward signal was just delayed, not absent), or does the diagnosis hold
# even at 2000 evals (the bandit's mechanism, not the budget, is the
# limit)?
#
# New experiment directory (not an overwrite of tr_shape_methods_dtlz2_200d)
# so the original 600-eval point stays intact for the controlled
# cross-budget comparison, matching the existing _2000ev convention.
#
# All 9 labels currently in tr_shape_methods_dtlz2_200d, 5 seeds each
# (some already have partial seeds there, but this is a fresh directory so
# all are submitted from scratch) = 45 jobs.
#
# Usage: bash cluster/submit_tr_shape_methods_200d_2000ev.sh
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

EXP=tr_shape_methods_dtlz2_200d_2000ev
SEEDS=(0 1 2 3 4)
LABELS=(morbo cma_ellipsoid mab_shape mab_shape_ducb mab_shape_ducb_shared linear_gp linear_gp_pca ard_pca_ellipsoid ard_pca_dimprior)

echo "Submitting $EXP jobs (dim=200, 2000 evals, batch 50)..."
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
echo "Compare against tr_shape_methods_dtlz2_200d (600 evals) with:"
echo "  python compute_significance.py $EXP mab_shape_ducb_shared --baseline-label morbo"
