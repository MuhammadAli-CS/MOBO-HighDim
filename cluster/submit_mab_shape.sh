#!/usr/bin/env bash
# Per-trust-region multi-armed bandit over shapes ("mab_shape"): AS-SMEA's
# own answer (Wang et al. 2026, Sec. 3.3, LS-IMA/MASS) to this project's own
# finding that no single shape wins everywhere (PCA wins on DTLZ2, no shape
# robustly wins on Rover) -- let each trust region learn per-arm reward
# estimates (epsilon-greedy over {isotropic, ard_box, pca_ellipsoid,
# ard_pca_ellipsoid, cma_ellipsoid}) from its own success-streak history,
# rather than fixing one shape globally for the whole run.
#
# Runs mab_shape alongside the already-established core methods (morbo,
# pca_ellipsoid, cma_ellipsoid) at d=100/150/200 (reusing the isotropic
# baseline already committed at d=100) plus Rover, where the DTLZ2-optimal
# shape (PCA) showed no robust benefit -- the case mab_shape is specifically
# meant to handle well by learning to fall back toward isotropic/whichever
# arm actually helps locally.
#
# Usage: bash cluster/submit_mab_shape.sh
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

# d=100: reuse the committed morbo/pca_ellipsoid/cma_ellipsoid baselines
# already in these experiment dirs; just add mab_shape.
EXP=tr_shape_methods_dtlz2_100d
echo "Submitting $EXP mab_shape job (other methods reused)..."
J=$(submit "$EXP" "mab_shape" 0)
submit_plot "$EXP" "$J"
echo "  mab_shape=$J"

# d=150/200: full comparison set including mab_shape (no committed baseline
# for mab_shape at these dims yet).
for EXP in tr_shape_methods_dtlz2_150d tr_shape_methods_dtlz2_200d; do
  echo "Submitting $EXP mab_shape job (other methods reused)..."
  J=$(submit "$EXP" "mab_shape" 0)
  submit_plot "$EXP" "$J"
  echo "  mab_shape=$J"
done

# Rover: the case mab_shape is specifically meant to help -- no shape
# variant showed a robust win there, so a bandit that can learn to prefer
# isotropic per-TR is the direct test of whether adaptivity recovers what
# fixed-shape variants couldn't.
EXP=tr_shape_rover
echo "Submitting $EXP mab_shape job (other methods reused)..."
J=$(submit "$EXP" "mab_shape" 0)
submit_plot "$EXP" "$J"
echo "  mab_shape=$J"

echo "All jobs submitted. Check with: squeue -u \$USER"
