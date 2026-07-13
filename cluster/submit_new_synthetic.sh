#!/usr/bin/env bash
# New synthetic benchmarks probing specific mechanisms (see the problem
# files' docstrings in morbo/problems/ for full rationale):
#
# 1. RotatedSparseDTLZ2 (rotated_sparse_dtlz2_d100_keff{5,50}) -- closes
#    the axis-alignment gap in the SparseDTLZ2 study: informative subspace
#    is a random ROTATED subspace, so rotation-based shapes (pca/ard_pca/
#    cma) should be ~invariant while ard_box (axis-aligned by construction)
#    should get strictly worse and isotropic stay unaffected. ard_box is
#    deliberately included here -- it's the method this problem exists to
#    discriminate against. Protocol matches sparse_dtlz2_* (400 evals),
#    5 seeds. 2 exps x 5 seeds x 5 methods = 50 jobs.
#
# 2. TimeVaryingSparseDTLZ2 (tv_sparse_dtlz2_d100_keff5) -- informative
#    dims switch at 50% budget; probes re-adaptation. cma_ellipsoid's
#    persistent covariance (its strength at d=200) should HURT here;
#    memoryless pca_ellipsoid should recover fast. 5 seeds x 4 methods
#    = 20 jobs. NOTE: analysis metric is post-switch HV recovery, not the
#    final mixed-history HV (see the problem file's METRIC CAVEAT).
#
# Usage: bash cluster/submit_new_synthetic.sh
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

SEEDS=(0 1 2 3 4)

# RotatedSparseDTLZ2: ard_box included on purpose (the discriminating case).
ROT_METHODS=(morbo ard_box pca_ellipsoid ard_pca_ellipsoid cma_ellipsoid)
for EXP in rotated_sparse_dtlz2_d100_keff5 rotated_sparse_dtlz2_d100_keff50; do
  echo "=== $EXP ==="
  IDS=()
  for SEED in "${SEEDS[@]}"; do
    for LABEL in "${ROT_METHODS[@]}"; do
      J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
    done
  done
  echo "  submitted ${#IDS[@]} jobs"
  submit_plot "$EXP" "${IDS[@]}"
done

# TimeVaryingSparseDTLZ2: the re-adaptation quartet.
TV_METHODS=(morbo pca_ellipsoid cma_ellipsoid mab_shape)
EXP=tv_sparse_dtlz2_d100_keff5
echo "=== $EXP ==="
IDS=()
for SEED in "${SEEDS[@]}"; do
  for LABEL in "${TV_METHODS[@]}"; do
    J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
  done
done
echo "  submitted ${#IDS[@]} jobs"
submit_plot "$EXP" "${IDS[@]}"

echo
echo "Done (70 jobs total). Check with: squeue -u \$USER"
