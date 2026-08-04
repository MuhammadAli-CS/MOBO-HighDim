#!/usr/bin/env bash
# Composite modeling on 2D SIMP structural topology optimization (1152D
# half-MBB-beam per-element density design, morbo/problems/composite_topopt.py
# -- see writeup/methods.tex's topopt subsection). Element stiffness matrix,
# DOF assembly, and boundary conditions ported verbatim from the verified,
# peer-reviewed "88 lines" MATLAB code (Andreassen et al. 2011), not
# retyped from the paper -- no MPB/MEEP-style compiled dependency needed
# (pure NumPy/SciPy sparse solve). Same direct-vs-composite A/B pattern as
# submit_rcm40_2x1.sh/submit_photonic_2x1.sh.
#
# 2 labels x 5 seeds = 10 jobs. No _pca/_ard_pca shape variants (same
# reason as the other composite benchmarks: the pre-existing
# TR-index-logging assertion crash found during the SnAr shape ablation).
#
# Usage: bash cluster/submit_topopt_2x1.sh
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

EXP=topopt_composite
SEEDS=(0 1 2 3 4)
LABELS=(morbo composite_topopt)

echo "Submitting $EXP jobs (2D SIMP topology optimization, half-MBB beam, d=1152, M=2)..."
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
