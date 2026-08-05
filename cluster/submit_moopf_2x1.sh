#!/usr/bin/env bash
# Composite modeling on 5-objective many-objective OPF (MOOPF) -- the same
# real IEEE 14-bus system and 34-dim design space as RCM40/RCM46 (CEC2021
# RWCMOP suite, morbo/problems/composite_moopf.py), but with FIVE objectives:
# RCM46's four (fuel cost, active/reactive power loss, voltage deviation)
# plus the Kessel-Glavitsch L-index voltage-stability objective. Tests
# whether pushing past RCM46's 4 decoupled objectives (the CEC2021 suite's
# maximum) to 5 amplifies the strongest composite win in the project
# further. Same direct-vs-composite A/B pattern as submit_rcm46_2x1.sh --
# pure admittance-matrix linear algebra, no simulator dependency.
#
# 2 labels x 5 seeds = 10 jobs. No _pca/_ard_pca shape variants (same
# reason as submit_rcm40_2x1.sh/submit_gri_mech_2x1.sh: the pre-existing
# TR-index-logging assertion crash found during the SnAr shape ablation).
#
# Usage: bash cluster/submit_moopf_2x1.sh
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

EXP=moopf_composite
SEEDS=(0 1 2 3 4)
LABELS=(morbo composite_moopf)

echo "Submitting $EXP jobs (5-objective MOOPF, IEEE 14-bus, d=34, M=5)..."
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
