#!/usr/bin/env bash
# Composite modeling x shape adaptation at HIGH dimension -- the version of
# the 2x2 the Penicillin run couldn't answer (d=7 is a shape-null regime,
# so no interaction was detectable there; RESULTS.md sec 5).
#
# At d=100 shape adaptation is worth +66%. CompositeDTLZ2's GP models the
# raw response [g, cos, sin] -- i.e. the scalar g that IS the problem's
# low-dimensional structure -- instead of the 2 objectives. Two competing
# hypotheses, both interesting:
#   (a) redundancy: the g-GP already captures the effective structure, so
#       rotating the trust region adds nothing on top;
#   (b) stacking: better-conditioned local models give shape adaptation a
#       cleaner signal, and the improvements compound.
#
# Runs INTO the existing tr_shape_dtlz2_100d experiment dir (CompositeDTLZ2
# is mathematically identical to DTLZ2, same config applies -- the label
# overrides swap evalfn), giving a controlled 6-cell comparison against the
# already-committed morbo / pca_ellipsoid / ard_pca_ellipsoid seeds 0-4:
#   direct    x {isotropic, pca, ard_pca}   <- committed
#   composite x {isotropic, pca, ard_pca}   <- this script
# 3 labels x 5 seeds = 15 jobs.
#
# Usage: bash cluster/submit_composite_shape_100d.sh
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

EXP=tr_shape_dtlz2_100d
echo "=== $EXP: composite x shape cells (5 seeds each) ==="
IDS=()
for SEED in 0 1 2 3 4; do
  for LABEL in composite_morbo composite_dtlz2_pca composite_dtlz2_ard_pca; do
    J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
  done
done
echo "  submitted ${#IDS[@]} jobs"

deps=$(IFS=:; echo "${IDS[*]}")
sbatch --requeue \
  --job-name="plot-${EXP}-composite" \
  --dependency=afterok:"$deps" \
  --partition=aimi --account=kilian \
  --cpus-per-task=1 --mem=4g --time=00:15:00 \
  --output="cluster/logs/plot-${EXP}-composite_%j.out" \
  --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_aggregate.py $EXP"

echo "Done (15 jobs + plot). Check with: squeue -u \$USER"
