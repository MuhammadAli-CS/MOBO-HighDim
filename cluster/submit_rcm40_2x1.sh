#!/usr/bin/env bash
# Composite modeling on RCM40 Optimal Power Flow (CEC2021 RWCMOP suite,
# Kumar et al. 2021; real IEEE 14-bus test system, Biswas et al. 2020;
# morbo/problems/composite_rcm40.py -- see writeup/methods.tex's RCM40
# subsection). Same direct-vs-composite A/B pattern as
# submit_gri_mech_2x1.sh, on this project's next real high-dimensional
# (34D) benchmark: no chemistry-simulator dependency at all here (pure
# admittance-matrix linear algebra, no ODE/PDE), so no thread-contention
# risk like Cantera's -- eval cost is trivial, this is purely testing
# whether composite modeling of the 29-dim per-bus raw response helps
# vs. the 2-dim direct objectives.
#
# 2 labels x 5 seeds = 10 jobs. No _pca/_ard_pca shape variants (same
# reason as submit_gri_mech_2x1.sh: the pre-existing TR-index-logging
# assertion crash found during the SnAr shape ablation).
#
# Usage: bash cluster/submit_rcm40_2x1.sh
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

EXP=rcm40_composite
SEEDS=(0 1 2 3 4)
LABELS=(morbo composite_rcm40)

echo "Submitting $EXP jobs (RCM40 Optimal Power Flow, d=34, M=2)..."
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
