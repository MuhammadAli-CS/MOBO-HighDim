#!/usr/bin/env bash
# Composite modeling x trust-region shape adaptation, a 2x2 factorial on
# Penicillin (d=7, M=3): {direct, composite} x {isotropic, pca_ellipsoid}.
# Tests whether the two orthogonal MORBO extensions built in this project
# interact -- does composite modeling make shape adaptation redundant, does
# shape adaptation amplify the composite gain, or are they independent?
#
# The four cells map to labels (config's evalfn is "Penicillin", so the
# direct cells use it directly; the composite cells override to
# CompositePenicillin):
#   direct    + isotropic  -> morbo
#   direct    + pca        -> pca_ellipsoid
#   composite + isotropic  -> composite_penicillin
#   composite + pca        -> composite_penicillin_pca
# (composite_penicillin_ard_pca is included as a fifth arm -- the strongest
#  shape variant crossed with composite -- since it's nearly free.)
#
# Usage: bash cluster/submit_penicillin_2x2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

submit() {
  local exp=$1 label=$2 seed=$3
  sbatch --requeue \
    --job-name="${label}-${exp}" \
    --export=EXP="$exp",LABEL="$label",SEED="$seed" \
    --parsable \
    cluster/run_experiment.sub
}

EXP=tr_shape_penicillin_2x2
echo "Submitting $EXP jobs (Penicillin d=7, M=3, composite x shape)..."
IDS=()
for LABEL in morbo pca_ellipsoid composite_penicillin composite_penicillin_pca composite_penicillin_ard_pca; do
  J=$(submit "$EXP" "$LABEL" 0); IDS+=("$J"); echo "  $LABEL=$J"
done

deps=$(IFS=:; echo "${IDS[*]}")
sbatch --requeue \
  --job-name="plot-${EXP}" \
  --dependency=afterok:"$deps" \
  --partition=aimi --account=kilian \
  --cpus-per-task=1 --mem=4g --time=00:15:00 \
  --output="cluster/logs/plot-${EXP}_%j.out" \
  --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_comparison.py $EXP 0"

echo "All jobs submitted. Check with: squeue -u \$USER"
