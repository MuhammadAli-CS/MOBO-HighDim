#!/usr/bin/env bash
# Submits the remaining Part 1 experiments (the ones killed on the laptop
# last night: correlation ablation + composite Penicillin) as independent
# SLURM jobs. Unlike the laptop's sequential run, these run in PARALLEL on
# the cluster — that's the whole point of moving here. Each plot job waits
# (--dependency=afterok) on its group's runs before overlaying results.
#
# Usage: from the repo root on unicorn-login-01 (or an interactive shell):
#   bash cluster/submit_all.sh
#
# Requires cluster/setup_env.sh to have been run once already.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

submit() {
  local exp=$1 label=$2 seed=$3
  sbatch --requeue \
    --job-name="${label}" \
    --export=EXP="$exp",LABEL="$label",SEED="$seed" \
    --parsable \
    cluster/run_experiment.sub
}

echo "Submitting correlation_ablation_dtlz2curve jobs..."
J1=$(submit correlation_ablation_dtlz2curve morbo 0)
J2=$(submit correlation_ablation_dtlz2curve independent_gp_composite 0)
J3=$(submit correlation_ablation_dtlz2curve kronecker_gp_composite 0)
echo "  morbo=$J1 independent=$J2 kronecker=$J3"

echo "Submitting penicillin_composite jobs..."
J4=$(submit penicillin_composite morbo 0)
J5=$(submit penicillin_composite composite_penicillin 0)
echo "  morbo=$J4 composite_penicillin=$J5"

echo "Submitting plot jobs (run after their group finishes)..."
sbatch --requeue \
  --job-name=plot-correlation \
  --dependency=afterok:"$J1":"$J2":"$J3" \
  --partition=kilian \
  --cpus-per-task=1 --mem=4g --gres=gpu:0 --time=00:15:00 \
  --output=cluster/logs/plot-correlation_%j.out \
  --wrap="cd $SLURM_SUBMIT_DIR; source ~/.bashrc; conda activate \$HOME/morbo-env; python plot_comparison.py correlation_ablation_dtlz2curve 0"

sbatch --requeue \
  --job-name=plot-penicillin \
  --dependency=afterok:"$J4":"$J5" \
  --partition=kilian \
  --cpus-per-task=1 --mem=4g --gres=gpu:0 --time=00:15:00 \
  --output=cluster/logs/plot-penicillin_%j.out \
  --wrap="cd $SLURM_SUBMIT_DIR; source ~/.bashrc; conda activate \$HOME/morbo-env; python plot_comparison.py penicillin_composite 0"

echo "All jobs submitted. Check with: squeue -u \$USER"
