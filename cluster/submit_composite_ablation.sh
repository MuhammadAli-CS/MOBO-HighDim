#!/usr/bin/env bash
# Direct-vs-composite ablation (composite_ablation/run_ablation.py): does
# adding known composite structure help each of standard_mobo/chebyshev_bo/
# spherical_chebyshev_bo (and their composite counterparts)? Runs this
# project's own composite_dtlz2 benchmark plus all six benchmarks from a
# collaborator's repo (https://github.com/tau315/composite-mobo),
# deliberately excluding that repo's own MORBO/composite_morbo/
# batched_morbo variants -- this project already has its own authoritative
# MORBO comparison (morbo/, plug_and_play/, experiments/composite_curve_*,
# experiments/correlation_ablation_dtlz2curve/); the point of this ablation
# is the solvers this project did NOT already have a composite comparison
# for. See composite_ablation/solvers.py and composite_ablation/
# tau_benchmarks.py module docstrings for the vendoring/attribution note.
#
# One job per benchmark (not per-trial/seed): `run_ablation.py` already
# loops over --trials internally per solver pair, so a single job covers
# a benchmark's full study. `projected_langermann_2obj_500d` (the one
# high-dimensional collaborator benchmark) gets its own longer time
# budget; everything else is low-dim (<=50D) and should finish well
# within 2 hours.
#
# 7 jobs total: this project's own composite_dtlz2 (dim=6, M=5, matching
# plug_and_play/benchmarks.py's default) + all 6 tau benchmarks.
#
# Usage: bash cluster/submit_composite_ablation.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs composite_ablation/results

TRIALS=20

submit() {
  local job_name=$1 time=$2 mem=$3
  shift 3
  local args=("$@")
  sbatch --requeue \
    --job-name="composite-ablation-${job_name}" \
    --output="cluster/logs/composite-ablation-${job_name}_%j.out" \
    --error="cluster/logs/composite-ablation-${job_name}_%j.err" \
    --partition=aimi --account=kilian \
    --cpus-per-task=32 --mem="$mem" --time="$time" \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python -m composite_ablation.run_ablation ${args[*]} --trials $TRIALS --out-dir composite_ablation/results/${job_name}"
}

# This project's own composite_dtlz2 (dim=6, M=5 default -- see
# plug_and_play/benchmarks.py's _make_composite_dtlz2).
submit composite_dtlz2_ours   02:00:00 64g --source ours --benchmark composite_dtlz2 --dim 6 --num-objectives 5

# tau315/composite-mobo's own six benchmarks, "low"-suite ones share a
# time budget; the two "high"-suite ones (50D, 500D) get more.
submit dtlz2_2obj_6d               02:00:00 64g  --source tau --benchmark dtlz2_2obj_6d
submit ackley_griewank_2obj_6d     02:00:00 64g  --source tau --benchmark ackley_griewank_2obj_6d
submit five_ackley_5obj_6d         02:00:00 64g  --source tau --benchmark five_ackley_5obj_6d
submit langermann3_ackley_2obj_6d  02:00:00 64g  --source tau --benchmark langermann3_ackley_2obj_6d
submit ackley_griewank_2obj_50d    04:00:00 64g  --source tau --benchmark ackley_griewank_2obj_50d
submit projected_langermann_500d   12:00:00 96g  --source tau --benchmark projected_langermann_2obj_500d

echo "Submitted. Check with: squeue -u \$USER"
echo "Per-pair console logs: cluster/logs/composite-ablation-<job>_<jobid>.out"
echo "Raw HV traces + summary.json: composite_ablation/results/<job>/"
