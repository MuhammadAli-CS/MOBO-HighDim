#!/usr/bin/env bash
# Submits Parts 2 (LLM-assisted MORBO) and 3 (LLM-automated BoTier) — both
# need ANTHROPIC_API_KEY and both make outbound HTTPS calls per job, so
# confirm the cluster's compute nodes allow external network access before
# relying on this (login nodes usually do; compute nodes vary by site).
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-ant-...
#   bash cluster/submit_llm.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
  echo "ANTHROPIC_API_KEY is not set in this shell. Export it first:" >&2
  echo "  export ANTHROPIC_API_KEY=sk-ant-..." >&2
  exit 1
fi

submit() {
  local exp=$1 label=$2 seed=$3
  sbatch --requeue \
    --job-name="${label}" \
    --export=EXP="$exp",LABEL="$label",SEED="$seed",ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
    --parsable \
    cluster/run_experiment.sub
}

echo "Submitting llm_morbo_vehicle_safety jobs..."
J1=$(submit llm_morbo_vehicle_safety morbo 0)
J2=$(submit llm_morbo_vehicle_safety llm_morbo 0)
echo "  morbo=$J1 llm_morbo=$J2"

sbatch --requeue \
  --job-name=plot-llm-morbo \
  --dependency=afterok:"$J1":"$J2" \
  --partition=default_partition --account=kilian \
  --cpus-per-task=1 --mem=4g --time=00:15:00 \
  --output=cluster/logs/plot-llm-morbo_%j.out \
  --wrap="cd $SLURM_SUBMIT_DIR; source /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_comparison.py llm_morbo_vehicle_safety 0"

echo "NOTE: Part 3 (botier_llm) does not go through run_comparison.py — it's a"
echo "standalone script (run_botier_comparison.py), not yet wired into this"
echo "sbatch template. Run it via a one-off job if/when needed:"
echo "  sbatch --partition=default_partition --account=kilian --cpus-per-task=2 --mem=8g --time=01:00:00 \\"
echo "    --export=ANTHROPIC_API_KEY=\$ANTHROPIC_API_KEY \\"
echo "    --wrap=\"cd \$PWD; source /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \\\$HOME/morbo-env; python run_botier_comparison.py 10 60 0\""

echo "All jobs submitted. Check with: squeue -u \$USER"
