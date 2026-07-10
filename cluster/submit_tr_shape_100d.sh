#!/usr/bin/env bash
# Submits the trust-region shape-adaptation comparison:
# experiments/tr_shape_dtlz2_100d (dim=100, max_evals=600, batch=50 --
# same scale as fig2_dtlz2_100d). MORBO's trust regions were purely
# isotropic hypercubes (no lengthscale-based rescaling at all, a step below
# even the original TuRBO paper's own technique). Tests three alternative
# geometries against the isotropic baseline:
#   - ard_box: axis-aligned box rescaled per-dim by the TR's own fitted GP
#     ARD lengthscales.
#   - pca_ellipsoid: box rotated into the PCA frame of the TR's local data.
#   - ard_pca_ellipsoid: PCA rotation with axis widths reweighted by
#     lengthscales projected onto each principal axis.
# See morbo/trust_region.py's TurboHParams.tr_shape docstring and
# run_comparison.py's module docstring for the full design rationale.
#
# The isotropic baseline (morbo) is NOT resubmitted here -- it's the exact
# same evalfn/config/seed as fig2_dtlz2_100d's own morbo result, already
# copied into experiments/tr_shape_dtlz2_100d/morbo/0000_morbo.pt and
# committed to git, so `git pull` on the cluster already has it.
#
# Usage: from the repo root on unicorn-login-01 (or an interactive shell):
#   bash cluster/submit_tr_shape_100d.sh
#
# Requires cluster/setup_env.sh to have been run once already.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

submit() {
  local exp=$1 label=$2 seed=$3
  sbatch --requeue \
    --job-name="${label}-100d" \
    --export=EXP="$exp",LABEL="$label",SEED="$seed" \
    --parsable \
    cluster/run_experiment.sub
}

EXP=tr_shape_dtlz2_100d

echo "Submitting $EXP jobs (dim=100, 600 evals, batch 50)..."
J1=$(submit "$EXP" ard_box 0)
J2=$(submit "$EXP" pca_ellipsoid 0)
J3=$(submit "$EXP" ard_pca_ellipsoid 0)
echo "  ard_box=$J1 pca_ellipsoid=$J2 ard_pca_ellipsoid=$J3"

echo "Submitting plot job (runs after all three finish)..."
sbatch --requeue \
  --job-name=plot-tr-shape-100d \
  --dependency=afterok:"$J1":"$J2":"$J3" \
  --partition=aimi --account=kilian \
  --cpus-per-task=1 --mem=4g --time=00:15:00 \
  --output=cluster/logs/plot-tr-shape-100d_%j.out \
  --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_comparison.py $EXP 0"

echo "All jobs submitted. Check with: squeue -u \$USER"
echo "morbo (isotropic baseline) is reused from the committed"
echo "experiments/tr_shape_dtlz2_100d/morbo/0000_morbo.pt -- no need to run it."
