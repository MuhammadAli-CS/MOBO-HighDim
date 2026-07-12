#!/usr/bin/env bash
# Runs the new tr_shape methods -- cma_ellipsoid (AS-SMEA-style covariance
# adaptation), the spherically-projected linear kernel (linear-bo challenge
# baseline) alone and crossed with shape, and the dimension-scaled
# lengthscale prior (Hvarfner) as a candidate fix for ard_box's high-d
# collapse -- against the isotropic baseline, at the dimensions where the
# original shape effects were largest (d=100, and the d=150/200 crossover).
#
# Reuses the committed isotropic `morbo` baseline in tr_shape_methods_dtlz2_100d
# (identical to fig2_dtlz2_100d's). For d=150/200 the baseline is rerun since
# no committed baseline exists at those dims in a methods-comparison dir.
#
# Usage: bash cluster/submit_tr_shape_new_methods.sh
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

# d=100: reuse the committed morbo baseline; run only the new methods.
EXP=tr_shape_methods_dtlz2_100d
echo "Submitting $EXP new-method jobs (baseline reused)..."
IDS=()
for LABEL in cma_ellipsoid linear_gp linear_gp_pca linear_gp_cma ard_box_dimprior ard_pca_dimprior; do
  J=$(submit "$EXP" "$LABEL" 0); IDS+=("$J"); echo "  $LABEL=$J"
done
submit_plot "$EXP" "${IDS[@]}"

# d=150 and d=200: the crossover regime where the shape effect was largest.
# Full label set including the isotropic baseline (no committed baseline here).
for EXP in tr_shape_methods_dtlz2_150d tr_shape_methods_dtlz2_200d; do
  echo "Submitting $EXP jobs..."
  IDS=()
  for LABEL in morbo cma_ellipsoid linear_gp linear_gp_pca ard_pca_ellipsoid ard_pca_dimprior; do
    J=$(submit "$EXP" "$LABEL" 0); IDS+=("$J"); echo "  $LABEL=$J"
  done
  submit_plot "$EXP" "${IDS[@]}"
done

echo "All jobs submitted. Check with: squeue -u \$USER"
