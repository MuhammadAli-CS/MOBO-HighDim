#!/usr/bin/env bash
# Composite modeling x trust-region shape adaptation on Summit's SnAr
# plug-flow reactor (d=4, M=2: space-time yield, E-factor; Felton et al.
# 2021, kinetics from Hone et al. 2017 -- see writeup/methods.tex
# \S9.22.5). Same question as submit_penicillin_2x2.sh, on a genuinely
# different (chemistry, not fermentation) simulator and at a much lower
# input dimension: does composite modeling (GP over the 6-dim raw
# final-concentration/flow-rate response, morbo/problems/composite_snar.py)
# help at low d, does shape adaptation still do anything at d=4, and do the
# two compose.
#
# 5 labels (config's evalfn is "SnAr", so the direct cells use it as-is;
# the composite cells override to CompositeSnAr):
#   direct    + isotropic  -> morbo
#   direct    + pca        -> pca_ellipsoid
#   composite + isotropic  -> composite_snar
#   composite + pca        -> composite_snar_pca
#   composite + ard_pca    -> composite_snar_ard_pca
# x 5 seeds (SnAr's ODE solve is cheap -- a few ms/eval -- so a real seed
# count costs little, unlike Penicillin's 2500-step integration).
#
# Usage: bash cluster/submit_snar_2x3.sh
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

EXP=snar_composite
SEEDS=(0 1 2 3 4)
LABELS=(morbo pca_ellipsoid composite_snar composite_snar_pca composite_snar_ard_pca)

echo "Submitting $EXP jobs (SnAr d=4, M=2, composite x shape)..."
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
