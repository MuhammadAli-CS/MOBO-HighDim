#!/usr/bin/env bash
# Controlled decoupling experiment: DTLZ2 with the objective (and Pareto front)
# held FIXED while the composite raw response's effective input-dimension is
# dialed via block_size -- see morbo/problems/composite_dtlz2_blocked.py. If
# the composite advantage falls monotonically as block_size (= effective dim)
# rises, decoupling *causes* the composite advantage rather than merely
# correlating with it across heterogeneous benchmarks.
#
# One experiment per block size (experiments/decoupling_b{1,2,4,8,16,48},
# dim=49 so effective dim = block/49 spans 0.02 -> 0.98). Each runs the
# block-size-independent direct baseline (dtlz2_blocked) and the composite
# label (composite_dtlz2_blocked) at 5 seeds, plus a dependent plot job.
#
# Usage: bash cluster/submit_decoupling_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

submit() {
  local exp=$1 label=$2 seed=$3
  sbatch --requeue \
    --job-name="${label}-${exp}-s${seed}" \
    --export=EXP="$exp",LABEL="$label",SEED="$seed" \
    --parsable --time=0 \
    cluster/run_experiment.sub
}

SEEDS=(0 1 2 3 4)
LABELS=(dtlz2_blocked composite_dtlz2_blocked)
BLOCKS=(1 2 4 8 16 48)

for b in "${BLOCKS[@]}"; do
  EXP="decoupling_b${b}"
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
done

echo "Submitted 6 block sizes x (2 labels x 5 seeds + 1 plot) = 66 jobs."
echo "After they finish, run: python analyze_decoupling_sweep.py  (aggregates the sweep + makes the advantage-vs-effdim plot)"
