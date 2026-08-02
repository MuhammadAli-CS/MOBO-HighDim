#!/usr/bin/env bash
# Composite modeling on Penicillin (d=7, M=3, morbo/problems/composite_penicillin.py),
# a proper 5-seed direct-vs-composite A/B. This benchmark was implemented
# early in the project and used as the precedent for the checkpointed-
# trajectory composite pattern later reused by SnAr/GRI-Mech/RCM40/OC20, but
# was itself never run past a single seed (cluster/submit_penicillin_2x2.sh
# runs each of its 5 labels once, for the composite x shape factorial, not
# for a significance test) -- so whether composite modeling actually helps
# here was, until now, untested at scale. Fed-batch fermentation dynamics
# (biomass/substrate/product/CO2 over time) are non-monotonic (growth then
# decline), unlike GRI-Mech/OC20's monotonically-converging trajectories,
# so this is a real test of whether that distinction matters.
#
# 2 labels x 5 seeds = 10 jobs. No _pca/_ard_pca shape variants (same
# reason as submit_gri_mech_2x1.sh/submit_rcm40_2x1.sh/submit_oc20_2x1.sh).
#
# Usage: bash cluster/submit_penicillin_5seed.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

submit() {
  local exp=$1 label=$2 seed=$3
  sbatch --requeue \
    --job-name="${label}-${exp}-s${seed}" \
    --export=EXP="$exp",LABEL="$label",SEED="$seed" \
    --parsable \
    --time=0 \
    cluster/run_experiment.sub
}

EXP=penicillin_composite
SEEDS=(0 1 2 3 4)
LABELS=(morbo composite_penicillin)

echo "Submitting $EXP jobs (Penicillin fed-batch fermentation, d=7, M=3)..."
IDS=()
for SEED in "${SEEDS[@]}"; do
  for LABEL in "${LABELS[@]}"; do
    J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
  done
done

deps=$(IFS=:; echo "${IDS[*]}")
sbatch --requeue \
  --job-name="plot-${EXP}" \
  --output="cluster/logs/plot-${EXP}_%j.out" \
  --error="cluster/logs/plot-${EXP}_%j.err" \
  --dependency=afterany:$deps \
  --partition=aimi --account=kilian \
  --cpus-per-task=4 --mem=8g --time=1:00:00 \
  --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_aggregate.py $EXP --labels ${LABELS[*]}"

echo "Submitted ${#IDS[@]} jobs + 1 dependent plot job."
echo "Check with: squeue -u \$USER"
