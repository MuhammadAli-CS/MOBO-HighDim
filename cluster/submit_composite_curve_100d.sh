#!/usr/bin/env bash
# Submits the fig2-scale correlation-ablation follow-up:
# experiments/composite_curve_dtlz2_100d (dim=100, max_evals=600, batch=50,
# n_curve_points=8 -- same scale as fig2_dtlz2_100d, but using the
# genuinely-correlated 8-point curve raw response instead of DTLZ2's
# g/cos/sin decomposition). Purpose: test whether the correlation-ablation's
# small composite-vs-direct margin at d=20 (see
# experiments/correlation_ablation_dtlz2curve/RESULTS.md) was a dimensionality
# artifact -- fig2_dtlz2_100d's own g/cos/sin composite won by +25% HV at
# d=100, so this checks whether the *curve* composite construction shows a
# similarly large margin once d matches.
#
# Runs all three labels (morbo, independent_gp_composite,
# kronecker_gp_composite) on the cluster rather than the laptop --
# kronecker_gp_composite was ~100x the fit cost of independent_gp_composite
# at d=20/200 evals (2255s vs 22.7s, see correlation_ablation_dtlz2curve's
# RESULTS.md); at d=100/600 evals it is expected to be substantially more
# expensive still, which is why this one specifically was moved off the
# laptop rather than risk another multi-hour thermal situation.
#
# Usage: from the repo root on unicorn-login-01 (or an interactive shell):
#   bash cluster/submit_composite_curve_100d.sh
#
# Requires cluster/setup_env.sh to have been run once already, and
# experiments/composite_curve_dtlz2_100d/config.json to exist (already
# committed to the repo -- no local setup needed beyond git pull).
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

EXP=composite_curve_dtlz2_100d

echo "Submitting $EXP jobs (dim=100, 600 evals, batch 50)..."
J1=$(submit "$EXP" morbo 0)
J2=$(submit "$EXP" independent_gp_composite 0)
J3=$(submit "$EXP" kronecker_gp_composite 0)
echo "  morbo=$J1 independent=$J2 kronecker=$J3"

echo "Submitting plot job (runs after all three finish)..."
sbatch --requeue \
  --job-name=plot-composite-curve-100d \
  --dependency=afterok:"$J1":"$J2":"$J3" \
  --partition=kilian \
  --cpus-per-task=1 --mem=4g --gres=gpu:0 --time=00:15:00 \
  --output=cluster/logs/plot-composite-curve-100d_%j.out \
  --wrap="cd $SLURM_SUBMIT_DIR; source ~/.bashrc; conda activate \$HOME/morbo-env; python plot_comparison.py $EXP 0"

echo "All jobs submitted. Check with: squeue -u \$USER"
echo "Note: morbo is cheap to rerun (~14-25 min at this scale) rather than"
echo "syncing the laptop's already-completed fig2_dtlz2_100d/morbo result up --"
echo "reruns at the same seed/config should land on the same n_evals-indexed HV"
echo "trajectory regardless of which machine ran it."
