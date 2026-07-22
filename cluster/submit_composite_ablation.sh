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
# ONE JOB PER (BENCHMARK, SOLVER PAIR) -- NOT one job per benchmark.
# The first version of this script ran both applicable pairs sequentially
# in a single 2-hour job; real per-trial cost turned out to be 300-550s
# (not the seconds-scale toy-config timing this was first estimated from),
# so 20 trials x 2 pairs routinely takes 3-6+ hours -- 4 of 7 first-run
# jobs were killed by the 2h SLURM limit mid-pair with NOTHING saved
# (run_ablation.py has since been changed to checkpoint after every trial,
# not just at the end of a pair, so a future timeout only loses partial
# progress, not everything -- but generous time budgets below are still
# the main fix). 5-objective benchmarks (composite_dtlz2_ours,
# five_ackley_5obj_6d) get the most time: hypervolume computation cost
# scales badly with objective count (same lesson learned early in this
# project's own dtlz2_5obj_6d run).
#
# 12 jobs total: 5 benchmarks x 2 pairs each (composite_dtlz2_ours,
# dtlz2_2obj_6d, ackley_griewank_2obj_6d, langermann3_ackley_2obj_6d,
# five_ackley_5obj_6d) + 2 high-suite tau benchmarks x 1 pair each
# (ackley_griewank_2obj_50d, projected_langermann_2obj_500d).
#
# Usage: bash cluster/submit_composite_ablation.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs composite_ablation/results

TRIALS=20

submit() {
  local job_name=$1 pair=$2 time=$3 mem=$4
  shift 4
  local args=("$@")
  sbatch --requeue \
    --job-name="composite-ablation-${job_name}-${pair}" \
    --output="cluster/logs/composite-ablation-${job_name}-${pair}_%j.out" \
    --error="cluster/logs/composite-ablation-${job_name}-${pair}_%j.err" \
    --partition=aimi --account=kilian \
    --cpus-per-task=32 --mem="$mem" --time="$time" \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python -m composite_ablation.run_ablation ${args[*]} --pair $pair --trials $TRIALS --out-dir composite_ablation/results/${job_name}"
}

# This project's own composite_dtlz2 (dim=6, M=5 default -- see
# plug_and_play/benchmarks.py's _make_composite_dtlz2). 5 objectives:
# generously budgeted.
submit composite_dtlz2_ours standard   10:00:00 64g --source ours --benchmark composite_dtlz2 --dim 6 --num-objectives 5
submit composite_dtlz2_ours chebyshev  10:00:00 64g --source ours --benchmark composite_dtlz2 --dim 6 --num-objectives 5

# tau315/composite-mobo's own six benchmarks. Low-suite: one job per pair.
submit dtlz2_2obj_6d               standard   06:00:00 64g  --source tau --benchmark dtlz2_2obj_6d
submit dtlz2_2obj_6d               chebyshev  06:00:00 64g  --source tau --benchmark dtlz2_2obj_6d
submit ackley_griewank_2obj_6d     standard   06:00:00 64g  --source tau --benchmark ackley_griewank_2obj_6d
submit ackley_griewank_2obj_6d     chebyshev  06:00:00 64g  --source tau --benchmark ackley_griewank_2obj_6d
submit langermann3_ackley_2obj_6d  standard   06:00:00 64g  --source tau --benchmark langermann3_ackley_2obj_6d
submit langermann3_ackley_2obj_6d  chebyshev  06:00:00 64g  --source tau --benchmark langermann3_ackley_2obj_6d
# 5 objectives: generously budgeted, same as composite_dtlz2_ours above.
submit five_ackley_5obj_6d         standard   10:00:00 64g  --source tau --benchmark five_ackley_5obj_6d
submit five_ackley_5obj_6d         chebyshev  10:00:00 64g  --source tau --benchmark five_ackley_5obj_6d

# High-suite: one pair only (spherical), already confirmed to comfortably
# finish within these budgets on the first run.
submit ackley_griewank_2obj_50d    spherical  04:00:00 64g  --source tau --benchmark ackley_griewank_2obj_50d
submit projected_langermann_500d   spherical  12:00:00 96g  --source tau --benchmark projected_langermann_2obj_500d

echo "Submitted 12 jobs. Check with: squeue -u \$USER"
echo "Per-job console logs: cluster/logs/composite-ablation-<job>-<pair>_<jobid>.out"
echo "Raw HV traces + per-pair summary: composite_ablation/results/<job>/"
echo "Once all pairs for a benchmark finish: python -m composite_ablation.plot_ablation composite_ablation/results/<job>"
