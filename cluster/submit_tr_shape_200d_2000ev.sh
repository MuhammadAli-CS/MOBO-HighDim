#!/usr/bin/env bash
# Follow-up to the tr_shape dimension sweep: at d=200/600 evals, ALL FOUR
# methods (morbo, ard_box, pca_ellipsoid, ard_pca_ellipsoid) showed exactly
# zero true hypervolume the entire run -- not a failed job (600 real evals,
# real non-trivial Pareto fronts, just none beating the reference point
# within budget). At d=150/600 evals, pca_ellipsoid/ard_pca_ellipsoid broke
# through late (eval ~300-400) and recovered to real HV by eval 600, while
# morbo/ard_box never did. This reruns d=200 at Rover's 2000-eval budget
# (same everything else -- same n_initial_points/batch/min_tr_size as the
# 600-eval version) to see if the same late-breakthrough pattern repeats at
# a longer horizon, or whether d=200 genuinely can't be cracked by any
# variant at this scale.
#
# Kept as a SEPARATE experiment directory from tr_shape_dtlz2_200d (not an
# overwrite) so the original fixed-600-eval sweep point stays intact for the
# controlled cross-d comparison.
#
# Usage: from the repo root on unicorn-login-01 (or an interactive shell):
#   bash cluster/submit_tr_shape_200d_2000ev.sh
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

EXP=tr_shape_dtlz2_200d_2000ev
LABELS=(morbo ard_box pca_ellipsoid ard_pca_ellipsoid)

echo "Submitting $EXP jobs (dim=200, 2000 evals, batch 50)..."
JOB_IDS=()
for LABEL in "${LABELS[@]}"; do
  J=$(submit "$EXP" "$LABEL" 0)
  JOB_IDS+=("$J")
  echo "  $LABEL=$J"
done

deps=$(IFS=:; echo "${JOB_IDS[*]}")
sbatch --requeue \
  --job-name="plot-${EXP}" \
  --dependency=afterok:"$deps" \
  --partition=aimi --account=kilian \
  --cpus-per-task=1 --mem=4g --time=00:15:00 \
  --output="cluster/logs/plot-${EXP}_%j.out" \
  --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_comparison.py $EXP 0"

echo "All jobs submitted. Check with: squeue -u \$USER"
echo "3.3x the eval budget of the main sweep's d=200 point -- expect this to"
echo "take proportionally longer than tr_shape_dtlz2_200d's own jobs did."
