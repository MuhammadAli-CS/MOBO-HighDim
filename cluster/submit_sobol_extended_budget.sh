#!/usr/bin/env bash
# submit_sobol_baseline.sh only covered the 600-eval experiments. This adds
# "sobol" to the 2000-eval extended-budget experiments too
# (tr_shape_dtlz2_150d_2000ev, tr_shape_dtlz2_200d_2000ev) -- the more
# interesting comparison, arguably, since that's exactly the budget where
# mab_shape and the PCA variants break through and separate from each
# other (RESULTS.md's mab_shape section). Does MORBO/shape-adaptation's
# advantage over pure random search hold up once budget is no longer the
# bottleneck, or does random search catch up given enough evals?
#
# Single seed (0), matching the existing seed-0-only convention in both of
# these experiment dirs.
#
# Usage: bash cluster/submit_sobol_extended_budget.sh
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

EXPERIMENTS=(tr_shape_dtlz2_150d_2000ev tr_shape_dtlz2_200d_2000ev)

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=== $EXP ==="
  J=$(submit "$EXP" "sobol" 0)
  echo "  sobol=$J"
  sbatch --requeue \
    --job-name="plot-${EXP}-sobol" \
    --dependency=afterok:"$J" \
    --partition=aimi --account=kilian \
    --cpus-per-task=1 --mem=4g --time=00:15:00 \
    --output="cluster/logs/plot-${EXP}-sobol_%j.out" \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_comparison.py $EXP 0"
done

echo
echo "All jobs submitted. Check with: squeue -u \$USER"
echo "Aggregate with: python aggregate_seeds.py <experiment_name>"
