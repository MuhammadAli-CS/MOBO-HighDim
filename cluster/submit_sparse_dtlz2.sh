#!/usr/bin/env bash
# Partial-effective-dimension DTLZ2 ("SparseDTLZ2", morbo/problems/sparse_dtlz2.py):
# masks all but `k_eff` of DTLZ2's `k = dim - M + 1` distance dimensions out
# of `g(x)` entirely (the rest are literal no-ops on every objective), so
# nominal dim and true effective dim can be varied independently -- standard
# DTLZ2 confounds the two, since its `k` (and the volume an isotropic box
# wastes) necessarily grows in lockstep with nominal dim.
#
# Two sweeps, both testing whether shape adaptation's benefit tracks the GAP
# between nominal and effective dimension (this project's own mechanism,
# from experiments/tr_shape_dtlz2_100d/RESULTS.md), not nominal dimension
# alone:
#
#   Group A (sparse_dtlz2_d{60,80,100,150,200}_keff5): effective dim held
#     fixed at (M-1)+k_eff = 1+5 = 6 while nominal dim scales 60->200. If the
#     benefit is really about the GAP, it should keep growing with nominal
#     dim here even though it did not for effective-dim-collapse reasons in
#     the plain DTLZ2 sweep (there, more nominal dim also means more true
#     signal dims). If the benefit instead plateaus once effective dim is
#     pinned, that's an important correction to the "gap" framing.
#   Group B (sparse_dtlz2_d100_keff{2,10,20,50}): nominal dim held fixed at
#     100 while effective dim varies (via k_eff) from small to
#     near-plain-DTLZ2 -- the direct dose-response test of the same claim.
#
# Each experiment gets the 4 core methods (morbo, pca_ellipsoid,
# ard_pca_ellipsoid, cma_ellipsoid) at seed 0.
#
# Usage: bash cluster/submit_sparse_dtlz2.sh
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

submit_plot() {
  local exp=$1; shift
  local deps; deps=$(IFS=:; echo "$*")
  sbatch --requeue \
    --job-name="plot-${exp}" \
    --dependency=afterok:"$deps" \
    --partition=aimi --account=kilian \
    --cpus-per-task=1 --mem=4g --time=00:15:00 \
    --output="cluster/logs/plot-${exp}_%j.out" \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_comparison.py $exp 0"
}

METHODS=(morbo pca_ellipsoid ard_pca_ellipsoid cma_ellipsoid)

EXPERIMENTS=(
  sparse_dtlz2_d60_keff5
  sparse_dtlz2_d80_keff5
  sparse_dtlz2_d100_keff5
  sparse_dtlz2_d150_keff5
  sparse_dtlz2_d200_keff5
  sparse_dtlz2_d100_keff2
  sparse_dtlz2_d100_keff10
  sparse_dtlz2_d100_keff20
  sparse_dtlz2_d100_keff50
)

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=== $EXP ==="
  IDS=()
  for LABEL in "${METHODS[@]}"; do
    J=$(submit "$EXP" "$LABEL" 0); IDS+=("$J"); echo "  ${LABEL} -> $J"
  done
  submit_plot "$EXP" "${IDS[@]}"
done

echo
echo "All jobs submitted (${#EXPERIMENTS[@]} experiments x ${#METHODS[@]} methods)."
echo "Check with: squeue -u \$USER"
echo "Aggregate any experiment with: python aggregate_seeds.py <experiment_name>"
