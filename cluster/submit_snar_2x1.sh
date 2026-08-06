#!/usr/bin/env bash
# Re-run of composite SnAr after the raw-response cleanup: the 6th raw
# component is now the product molar flow F_product = C_product * q_tot (a
# genuine, uncertain reactor throughput) instead of the bare, closed-form-
# known q_tot = 5/tau -- so no per-output GP is wasted modeling a quantity
# that is already known exactly (see morbo/problems/composite_snar.py's
# docstring; objectives are bit-identical, only the composite raw response
# changed). Direct-vs-composite A/B, same pattern as submit_rcm46_2x1.sh.
#
# 2 labels x 5 seeds = 10 jobs. No _pca/_ard_pca shape variants (the
# pre-existing TR-index-logging assertion crash from the SnAr shape ablation).
#
# Usage: bash cluster/submit_snar_2x1.sh
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

EXP=snar_composite
SEEDS=(0 1 2 3 4)
LABELS=(morbo composite_snar)

echo "Submitting $EXP jobs (Summit SnAr, d=4, M=2, cleaned raw response)..."
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
