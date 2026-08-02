#!/usr/bin/env bash
# Composite modeling on RCM46 Optimal Power Flow -- the same real
# IEEE 14-bus system and 34-dim design space as RCM40 (CEC2021 RWCMOP
# suite, morbo/problems/composite_rcm46.py), but with 4 objectives
# (fuel cost, active/reactive power loss, voltage deviation) instead of
# 2. Tests whether more, more-decoupled objectives on the same real
# system amplify RCM40's own modest composite win (see
# writeup/methods.tex's RCM40 subsection). Same direct-vs-composite A/B
# pattern as submit_rcm40_2x1.sh -- pure admittance-matrix linear
# algebra, no simulator dependency, no thread-contention risk.
#
# 2 labels x 5 seeds = 10 jobs. No _pca/_ard_pca shape variants (same
# reason as submit_rcm40_2x1.sh/submit_gri_mech_2x1.sh: the pre-existing
# TR-index-logging assertion crash found during the SnAr shape ablation).
#
# Usage: bash cluster/submit_rcm46_2x1.sh
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

EXP=rcm46_composite
SEEDS=(0 1 2 3 4)
LABELS=(morbo composite_rcm46)

echo "Submitting $EXP jobs (RCM46 Optimal Power Flow, d=34, M=4)..."
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
