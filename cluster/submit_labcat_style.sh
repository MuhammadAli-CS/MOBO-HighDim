#!/usr/bin/env bash
# labcat_style: LABCAT's own construction (Visser et al. 2023) implemented
# directly as a tr_shape mode -- fitness-weighted PCA computed genuinely IN
# lengthscale-whitened coordinates, rotation kept directly rather than
# reweighted afterward. The opposite ordering from ard_pca_ellipsoid; see
# writeup/methods.tex sec 7.1's "Relation to prior work" / the
# "labcat_style" subsection, and RESULTS.md sec 13, for the exact
# mechanism and the no-op this avoids.
#
# For completeness, this runs labcat_style against EVERY experiment in the
# tr_shape study that already has other shape-variant baselines to compare
# against (morbo/pca_ellipsoid/ard_pca_ellipsoid/cma_ellipsoid etc.) --
# not just the two headline comparisons (DTLZ2 d=100, BBOB Rosenbrock).
# Excludes experiments outside the shape study entirely (composite-modeling-
# only dirs: fig2_dtlz2_100d, penicillin_composite, correlation_ablation_
# dtlz2curve, composite_curve_dtlz2_100d; and bare unshaped baselines with
# no other tr_shape variant to compare against: dtlz2_10d/30d/100d, rover,
# vehicle_safety, welded_beam, llm_morbo_vehicle_safety).
#
# Groups (see inline comments) x 5 seeds = jobs per group; LassoBench
# requires the LassoBench package (cluster-only, as with every other label
# run against those 3 experiments).
#
# Usage: bash cluster/submit_labcat_style.sh
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
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_aggregate.py $exp"
}

SEEDS=(0 1 2 3 4)
LABEL=labcat_style

# Group 1: DTLZ2 dimension sweep (core comparison, RESULTS.md sec 1/3).
G1=(
  tr_shape_dtlz2_20d
  tr_shape_dtlz2_50d
  tr_shape_dtlz2_100d
  tr_shape_dtlz2_150d
  tr_shape_dtlz2_150d_2000ev
  tr_shape_dtlz2_200d
  tr_shape_dtlz2_200d_2000ev
)
# Group 2: other DTLZ landscapes (RESULTS.md sec 10e/11a's ard_box
# rugged-g/smooth-g finding).
G2=(
  tr_shape_dtlz1_100d
  tr_shape_dtlz3_100d
  tr_shape_dtlz5_100d
  tr_shape_dtlz7_100d
)
# Group 3: the new-methods experiments (cma_ellipsoid, mab_shape, linear_gp
# variants) -- the most direct comparison points, since these already host
# multiple non-baseline shape mechanisms.
G3=(
  tr_shape_methods_dtlz2_100d
  tr_shape_methods_dtlz2_150d
  tr_shape_methods_dtlz2_200d
)
# Group 4: real (non-DTLZ) problems.
G4=(
  tr_shape_rover
  tr_shape_penicillin
)
# Group 5: SparseDTLZ2 effective-dimension family (RESULTS.md sec 7).
G5=(
  sparse_dtlz2_d100_keff2
  sparse_dtlz2_d100_keff5
  sparse_dtlz2_d100_keff10
  sparse_dtlz2_d100_keff20
  sparse_dtlz2_d100_keff50
  sparse_dtlz2_d60_keff5
  sparse_dtlz2_d80_keff5
  sparse_dtlz2_d150_keff5
  sparse_dtlz2_d200_keff5
)
# Group 6: RotatedSparseDTLZ2 (rotation-under-effective-dimension).
G6=(
  rotated_sparse_dtlz2_d100_keff5
  rotated_sparse_dtlz2_d100_keff50
)
# Group 7: TimeVaryingSparseDTLZ2 (non-stationary effective dimension).
G7=(
  tv_sparse_dtlz2_d100_keff5
  tv_sparse_dtlz2_d100_keff49
)
# Group 8: SparseRover (real-problem effective-dimension analog).
G8=(
  sparse_rover_d120
  sparse_rover_d180
)
# Group 9: LassoBench (RESULTS.md sec 10a) -- requires LassoBench installed
# on the cluster node, as with every other label run against these.
G9=(
  lasso_dna_mo
  lasso_synt_high_mo
  lasso_synt_medium_mo
)
# Group 10: BBOB-style landscape taxonomy (RESULTS.md sec 12) -- all 8
# pairs, not just rosenbrock_rosenbrock, now that the baseline batch
# (submit_bbob_style.sh) has been submitted separately.
G10=(
  bbob_sphere_sphere
  bbob_ellipsoidal_ellipsoidal
  bbob_rosenbrock_rosenbrock
  bbob_rastrigin_rastrigin
  bbob_peaks_peaks
  bbob_sphere_peaks
  bbob_rastrigin_rastrigin_keff20
  bbob_rastrigin_rastrigin_keff80
)

EXPERIMENTS=("${G1[@]}" "${G2[@]}" "${G3[@]}" "${G4[@]}" "${G5[@]}" "${G6[@]}" "${G7[@]}" "${G8[@]}" "${G9[@]}" "${G10[@]}")

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=== $EXP ==="
  IDS=()
  for SEED in "${SEEDS[@]}"; do
    J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
  done
  echo "  submitted ${#IDS[@]} jobs"
  submit_plot "$EXP" "${IDS[@]}"
done

echo
echo "Done (${#EXPERIMENTS[@]} experiments x ${#SEEDS[@]} seeds = $((${#EXPERIMENTS[@]} * ${#SEEDS[@]})) jobs)."
echo "Check with: squeue -u \$USER"
echo "Aggregate with: python aggregate_seeds.py <experiment_name>"
