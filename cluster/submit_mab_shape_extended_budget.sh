#!/usr/bin/env bash
# Is mab_shape's d=150/200 failure (RESULTS.md #6) a budget artifact or a
# real design flaw? Precedent for "budget artifact": at d=200/600 evals,
# morbo and ard_box ALSO showed exactly 0.00, and that turned out to be
# purely a budget effect -- tr_shape_dtlz2_200d_2000ev (2000 evals) showed
# every method eventually breaking through, same ranking as d=100/150. This
# reruns mab_shape at the same 2000-eval budget at d=150 and d=200 to see if
# it recovers the same way, or whether its own fixed 15% exploration cost
# (mab_epsilon=0.15) keeps taxing the run regardless of how long it goes --
# which would mean the flaw isn't "not enough time to learn" but "constant
# tax that never amortizes."
#
# d=200: reuses the already-committed morbo/pca_ellipsoid/ard_pca_ellipsoid
# baselines in tr_shape_dtlz2_200d_2000ev; only mab_shape needs to run.
# d=150: no extended-budget experiment existed yet (the 600-eval version
# already broke through for pca_ellipsoid/ard_pca_ellipsoid there), so this
# submits the full 4-method comparison in a fresh tr_shape_dtlz2_150d_2000ev
# directory for a complete apples-to-apples table.
#
# Usage: bash cluster/submit_mab_shape_extended_budget.sh
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

# d=200: baselines already committed, just add mab_shape.
EXP=tr_shape_dtlz2_200d_2000ev
echo "Submitting $EXP mab_shape job (other methods reused)..."
J=$(submit "$EXP" "mab_shape" 0)
submit_plot "$EXP" "$J"
echo "  mab_shape=$J"

# d=150: fresh experiment dir, full comparison set.
EXP=tr_shape_dtlz2_150d_2000ev
echo "Submitting $EXP jobs (dim=150, 2000 evals, batch 50)..."
IDS=()
for LABEL in morbo pca_ellipsoid ard_pca_ellipsoid mab_shape; do
  J=$(submit "$EXP" "$LABEL" 0); IDS+=("$J"); echo "  $LABEL=$J"
done
submit_plot "$EXP" "${IDS[@]}"

echo "All jobs submitted. Check with: squeue -u \$USER"
