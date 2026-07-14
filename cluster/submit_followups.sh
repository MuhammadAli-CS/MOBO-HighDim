#!/usr/bin/env bash
# Three follow-ups demanded directly by the benchmark-battery results
# (RESULTS.md sec 10):
#
# 1. Multi-seed the AXIS-ALIGNED sparse_dtlz2_d100_keff50 (seeds 1-4; seed 0
#    committed). Resolves sec 10c's open question: pca_ellipsoid's
#    axis-aligned +6.7% was single-seed, and it vanished under rotation
#    (5-seed, -1.9%). If the multi-seed axis-aligned number also shrinks to
#    ~0, the "+6.7%" was seed noise; if it holds at ~+7% (5/5), rotation
#    genuinely hurts PCA's estimation and that's a real finding.
#    16 jobs (4 methods x 4 seeds).
#
# 2. Rerun TimeVaryingSparseDTLZ2 at k_eff=49 (tv_sparse_dtlz2_d100_keff49)
#    -- the k_eff=5 version was null-by-design (no adaptation advantage to
#    disrupt; sec 10d). k_eff=49 is the maximum allowing disjoint pre/post
#    masks at d=100 (2*k_eff <= 99), squarely in the regime where shape
#    adaptation matters (sec 7 Group B). Tests the cma-memory-liability
#    hypothesis properly. 20 jobs (4 methods x 5 seeds).
#
# 3. Multi-seed the new-methods sweep at d=100 (seeds 1-4; seed 0 committed)
#    for the headline single-seed claims: cma_ellipsoid, linear_gp_pca,
#    mab_shape. 12 jobs (3 methods x 4 seeds).
#
# Total: 48 jobs.
# Usage: bash cluster/submit_followups.sh
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

# 1. Axis-aligned keff50, seeds 1-4 (seed 0 already committed).
EXP=sparse_dtlz2_d100_keff50
echo "=== $EXP (seeds 1-4) ==="
IDS=()
for SEED in 1 2 3 4; do
  for LABEL in morbo pca_ellipsoid ard_pca_ellipsoid cma_ellipsoid; do
    J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
  done
done
echo "  submitted ${#IDS[@]} jobs"
submit_plot "$EXP" "${IDS[@]}"

# 2. TimeVarying at k_eff=49, seeds 0-4, fresh experiment dir.
EXP=tv_sparse_dtlz2_d100_keff49
echo "=== $EXP (seeds 0-4) ==="
IDS=()
for SEED in 0 1 2 3 4; do
  for LABEL in morbo pca_ellipsoid cma_ellipsoid mab_shape; do
    J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
  done
done
echo "  submitted ${#IDS[@]} jobs"
submit_plot "$EXP" "${IDS[@]}"

# 3. Multi-seed the headline single-seed new-methods claims at d=100.
EXP=tr_shape_methods_dtlz2_100d
echo "=== $EXP (seeds 1-4, headline methods) ==="
IDS=()
for SEED in 1 2 3 4; do
  for LABEL in cma_ellipsoid linear_gp_pca mab_shape; do
    J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
  done
done
echo "  submitted ${#IDS[@]} jobs"
submit_plot "$EXP" "${IDS[@]}"

echo
echo "Done (48 jobs). Check with: squeue -u \$USER"
echo "Aggregate with: python aggregate_seeds.py <experiment_name>"
