#!/usr/bin/env bash
# BBOB-style landscape taxonomy: does the ard_box "wins on rugged-g,
# fails on smooth-g" finding (RESULTS.md sec 10e/11a, from just 4 DTLZ
# variants) generalize to genuinely non-algebraic BBOB-family landscapes?
# Also the closest thing in this study to LABCAT's own evaluation suite
# (COCO/BBOB) -- see morbo/problems/bbob_style.py's docstring for the
# honesty caveats (faithful-in-spirit reimplementation, not official
# cocoex; not bit-comparable to published COCO hypervolume tables).
#
# Group A (6 experiments): one pair per curated landscape category --
# sphere_sphere (trivial control), ellipsoidal_ellipsoidal (rotated,
# high-conditioning unimodal -- closest analog to RotatedSparseDTLZ2),
# rosenbrock_rosenbrock (moderate conditioning, curved valley),
# rastrigin_rastrigin (multimodal, adequate global structure -- DTLZ3/7
# analog), peaks_peaks (multimodal, weak global structure -- Rover/
# Gallagher analog), sphere_peaks (genuinely mixed bi-objective landscape:
# one smooth objective, one deceptive one -- no DTLZ variant in this study
# tests this, since DTLZ's objectives always share one g).
#
# Group B (2 experiments): effective-dimension dose-response on
# rastrigin_rastrigin (k_eff=20,80 of 100) -- does SparseDTLZ2's sec 7
# dose-response generalize to a non-algebraic, genuinely multimodal
# landscape?
#
# 8 experiments x 5 seeds x 4 methods (morbo, pca_ellipsoid,
# ard_pca_ellipsoid, cma_ellipsoid) = 160 jobs.
#
# Usage: bash cluster/submit_bbob_style.sh
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
METHODS=(morbo pca_ellipsoid ard_pca_ellipsoid cma_ellipsoid)
EXPERIMENTS=(
  bbob_sphere_sphere
  bbob_ellipsoidal_ellipsoidal
  bbob_rosenbrock_rosenbrock
  bbob_rastrigin_rastrigin
  bbob_peaks_peaks
  bbob_sphere_peaks
  bbob_rastrigin_rastrigin_keff20
  bbob_rastrigin_rastrigin_keff80
)

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=== $EXP ==="
  IDS=()
  for SEED in "${SEEDS[@]}"; do
    for LABEL in "${METHODS[@]}"; do
      J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
    done
  done
  echo "  submitted ${#IDS[@]} jobs"
  submit_plot "$EXP" "${IDS[@]}"
done

echo
echo "Done (${#EXPERIMENTS[@]} experiments x ${#SEEDS[@]} seeds x ${#METHODS[@]} methods = 160 jobs)."
echo "Check with: squeue -u \$USER"
echo "Aggregate with: python aggregate_seeds.py <experiment_name>"
