#!/usr/bin/env bash
# Extend to 20 seeds (5-19; 0-4 committed) where it matters, staged:
#
# Stage A (default) -- CONCLUSION-CHANGING: the noisy, coin-flip results
# where 5 seeds cannot resolve the +/-2-5% effects:
#   tr_shape_rover, sparse_rover_d{120,180}   (the Rover-family question)
#   rotated_sparse_dtlz2_d100_keff50 + sparse_dtlz2_d100_keff50
#       (the PCA-under-rotation question -- assumes submit_followups.sh's
#        seeds 1-4 for the axis-aligned one already ran)
#   tv_sparse_dtlz2_d100_keff49               (cma-memory hypothesis)
#   ~435 jobs.
#
# Stage B ("paper") -- PROTOCOL-MATCHING: results already unanimous at 5
# seeds, extended to 20 to match the MORBO paper's own 20-replication
# protocol for headline tables:
#   tr_shape_dtlz2_{50,100,150}d, tr_shape_dtlz{3,5,7}_100d,
#   tr_shape_methods_dtlz2_100d (cma/linear_gp_pca/mab_shape)
#   ~465 jobs.
#
# Deliberately NOT extended: LassoBench (already 30 seeds, its paper's own
# protocol), 2000-eval extended-budget runs (answered a binary question),
# SparseDTLZ2 nulls (more seeds on a <0.3% null adds nothing).
#
# Usage:
#   bash cluster/submit_20seed.sh          # stage A only
#   bash cluster/submit_20seed.sh paper    # stage B only
#   bash cluster/submit_20seed.sh all      # both
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

STAGE="${1:-a}"

submit() {
  local exp=$1 label=$2 seed=$3
  sbatch --requeue \
    --job-name="${label}-${exp}-s${seed}" \
    --export=EXP="$exp",LABEL="$label",SEED="$seed" \
    --parsable \
    cluster/run_experiment.sub
}

batch() {
  local exp=$1 first_seed=$2; shift 2
  local labels=("$@")
  echo "=== $exp (seeds ${first_seed}-19, ${#labels[@]} methods) ==="
  local n=0
  for ((SEED=first_seed; SEED<20; SEED++)); do
    for LABEL in "${labels[@]}"; do
      submit "$exp" "$LABEL" "$SEED" > /dev/null; n=$((n+1))
    done
  done
  echo "  submitted $n jobs"
}

CORE=(morbo ard_box pca_ellipsoid ard_pca_ellipsoid)
ROVER_SET=(morbo pca_ellipsoid ard_pca_ellipsoid cma_ellipsoid mab_shape sobol)

if [[ "$STAGE" == "a" || "$STAGE" == "all" ]]; then
  echo "--- Stage A: conclusion-changing ---"
  batch tr_shape_rover 5 morbo ard_box pca_ellipsoid ard_pca_ellipsoid
  batch sparse_rover_d120 5 "${ROVER_SET[@]}"
  batch sparse_rover_d180 5 "${ROVER_SET[@]}"
  batch rotated_sparse_dtlz2_d100_keff50 5 morbo ard_box pca_ellipsoid ard_pca_ellipsoid cma_ellipsoid
  batch sparse_dtlz2_d100_keff50 5 morbo pca_ellipsoid ard_pca_ellipsoid cma_ellipsoid
  batch tv_sparse_dtlz2_d100_keff49 5 morbo pca_ellipsoid cma_ellipsoid mab_shape
fi

if [[ "$STAGE" == "paper" || "$STAGE" == "all" ]]; then
  echo "--- Stage B: protocol-matching (MORBO paper's 20 replications) ---"
  batch tr_shape_dtlz2_50d 5 "${CORE[@]}"
  batch tr_shape_dtlz2_100d 5 "${CORE[@]}"
  batch tr_shape_dtlz2_150d 5 "${CORE[@]}"
  # tr_shape_dtlz1_100d deliberately excluded: all methods at exactly 0.00
  # (budget artifact) -- 20 seeds of a null adds nothing.
  batch tr_shape_dtlz3_100d 5 "${CORE[@]}"
  batch tr_shape_dtlz5_100d 5 "${CORE[@]}"
  batch tr_shape_dtlz7_100d 5 "${CORE[@]}"
  batch tr_shape_methods_dtlz2_100d 5 cma_ellipsoid linear_gp_pca mab_shape
fi

echo
echo "Done. Check with: squeue -u \$USER"
echo "Aggregate with: python aggregate_seeds.py <experiment_name>"
