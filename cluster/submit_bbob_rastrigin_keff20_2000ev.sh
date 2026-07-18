#!/usr/bin/env bash
# Targeted budget extension #1 (of 2): bbob_rastrigin_rastrigin_keff20 was
# a flat null at 600 evals for every shape variant (RESULTS.md §12 Group
# B) -- indistinguishable from zero effect, CIs straddling zero, win-rates
# at or below chance. That's consistent with either "no benefit exists at
# this k_eff" or "600 evals just isn't enough budget for anyone to find
# the k_eff=20 informative subspace on a genuinely multimodal Rastrigin
# landscape." This reruns the same config at Rover/d=200's 2000-eval
# budget (same n_initial_points/batch/min_tr_size, only max_evals changes)
# to distinguish the two: does a real signal ever appear, or does it stay
# null with more budget?
#
# Kept as a SEPARATE experiment directory (bbob_rastrigin_rastrigin_keff20_2000ev)
# so the original 600-eval Group B point stays intact for the controlled
# cross-budget comparison.
#
# 5 methods (morbo, pca_ellipsoid, ard_pca_ellipsoid, cma_ellipsoid,
# labcat_style) x 5 seeds = 25 jobs.
#
# Usage: bash cluster/submit_bbob_rastrigin_keff20_2000ev.sh
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

EXP=bbob_rastrigin_rastrigin_keff20_2000ev
SEEDS=(0 1 2 3 4)
LABELS=(morbo pca_ellipsoid ard_pca_ellipsoid cma_ellipsoid labcat_style)

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
echo "Compare against bbob_rastrigin_rastrigin_keff20 (600 evals) with:"
echo "  python compute_significance.py bbob_rastrigin_rastrigin_keff20_2000ev <label> --baseline-label morbo"
