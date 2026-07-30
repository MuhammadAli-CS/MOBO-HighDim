#!/usr/bin/env bash
# Composite modeling on OC20 catalyst-adsorbate relaxation (Chanussot
# et al. 2021), using a live, no-authentication GemNet-OC potential
# (fairchem-core, isolated in .fairchem_env/ -- see
# cluster/setup_fairchem_env.sh, which MUST be run once on this cluster
# checkout before submitting this script) as a proxy for the true DFT
# relaxation this task is normally defined against. See
# writeup/methods.tex's OC20 subsection and
# morbo/problems/composite_oc20.py's module docstring for the full story
# (why a live DFT oracle and the newer gated facebook/UMA checkpoint were
# both ruled out, and why this needs a separate venv at all).
#
# 2 labels x 5 seeds = 10 jobs. No _pca/_ard_pca shape variants (same
# reason as submit_gri_mech_2x1.sh/submit_rcm40_2x1.sh: the pre-existing
# TR-index-logging assertion crash found during the SnAr shape ablation).
# CPU-only (the isolated env's torch/torch_scatter/torch_sparse/
# torch_cluster wheels are all +cpu builds, deliberately, to avoid
# matching a specific CUDA version across environments) -- per-evaluation
# cost is a real ML relaxation (~5-10s locally), not free, so no SLURM
# time limit here either.
#
# Usage:
#   bash cluster/setup_fairchem_env.sh   # once, before this script
#   bash cluster/submit_oc20_2x1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

if [ ! -f .fairchem_env/bin/python ] && [ ! -f .fairchem_env/Scripts/python.exe ]; then
  echo "Isolated fairchem env not found -- run cluster/setup_fairchem_env.sh first." >&2
  exit 1
fi

submit() {
  local exp=$1 label=$2 seed=$3
  sbatch --requeue \
    --job-name="${label}-${exp}-s${seed}" \
    --export=EXP="$exp",LABEL="$label",SEED="$seed" \
    --parsable \
    --time=0 \
    cluster/run_experiment.sub
}

EXP=oc20_composite
SEEDS=(0 1 2 3 4)
LABELS=(morbo composite_oc20)

echo "Submitting $EXP jobs (OC20 catalyst relaxation via GemNet-OC, d=57, M=2)..."
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
