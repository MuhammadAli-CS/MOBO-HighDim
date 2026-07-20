#!/usr/bin/env bash
# New benchmark: DTLZ2 at 5 objectives, 6 dimensions (task requirement),
# run through the SAME real morbo engine every other experiment in this
# study uses (run_comparison.py / morbo.run_one_replication) -- not the
# plug_and_play wrapper, no special-casing. All core shape methods run
# against isotropic base MORBO ("morbo" label) as the required baseline.
#
# Honesty note: morbo/problems/composite_dtlz2.py (the actual "composite
# modeling" GP technique) only supports num_objectives=2 -- its reduction
# formula is hardcoded for that case. This experiment uses evalfn="DTLZ2"
# (direct objective modeling), which is mathematically identical to
# composite_dtlz2's final objectives at the same dim/M (verified
# numerically in morbo/problems/composite_dtlz2_general.py) but does not
# exercise composite-GP modeling itself. If the task specifically needs
# composite modeling tested at M=5 (not just the underlying DTLZ2
# problem), that needs a new evalfn branch in run_one_replication.py
# wired to composite_dtlz2_general.py -- not done here.
#
# Confirmed working locally at toy scale (30 evals) for both "morbo" and
# "pca_ellipsoid" before submitting this. Note: 5-objective hypervolume
# computation is measurably slower than this study's usual bi-objective
# runs (toy-scale 30 evals took ~84s locally, vs. a few seconds for
# similar bi-objective toy runs) -- run_experiment.sub's standard 8h time
# limit / GPU allocation should comfortably absorb this at the full
# 300-eval budget, but flagging it since it's a real, measured difference
# from every other experiment in this study, not assumed identical cost.
#
# 7 methods (morbo, ard_box, pca_ellipsoid, ard_pca_ellipsoid,
# cma_ellipsoid, labcat_style, mab_shape_ducb_shared) x 5 seeds = 35 jobs.
#
# Usage: bash cluster/submit_dtlz2_5obj_6d.sh
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

EXP=dtlz2_5obj_6d
SEEDS=(0 1 2 3 4)
LABELS=(morbo ard_box pca_ellipsoid ard_pca_ellipsoid cma_ellipsoid labcat_style mab_shape_ducb_shared)

echo "Submitting $EXP jobs (dim=6, num_objectives=5, 300 evals, batch 10)..."
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
echo "Aggregate with: python aggregate_seeds.py $EXP"
