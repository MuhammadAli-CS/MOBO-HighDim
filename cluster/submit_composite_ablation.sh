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
# THREAD CAP: passes --num-threads 8 --num-interop-threads 4 to every job
# (matching --cpus-per-task=32 below). Without this, three jobs sharing a
# node with five others were observed to either get OOM-killed or have
# per-trial time balloon 10x mid-run before timing out -- torch/MKL spawn
# thread pools sized to the node's full core count, not this job's actual
# 32-cpu SLURM allocation, causing severe contention on a shared node. See
# composite_ablation/run_ablation.py's module docstring ("Threading"
# section) -- this is the same class of problem already fixed once before
# in this project, in run_comparison.py, for one specific experiment.
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
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python -m composite_ablation.run_ablation ${args[*]} --pair $pair --trials $TRIALS --num-threads 8 --num-interop-threads 4 --out-dir composite_ablation/results/${job_name}"
}

# This project's own composite_dtlz2 (dim=6, M=5 default -- see
# plug_and_play/benchmarks.py's _make_composite_dtlz2).
#
# KNOWN ISSUE, unresolved: the "standard" (standard_mobo/composite_mobo)
# pair on BOTH 5-objective benchmarks here (composite_dtlz2_ours,
# five_ackley_5obj_6d below) has OOM-killed at 64g twice now -- once
# sharing a node with 5 others (plausibly contention-related) AND once
# running alone with the thread-cap fix already applied (NOT
# contention -- this is a real memory scaling problem, most likely
# NondominatedPartitioning's box decomposition, which is known to blow
# up combinatorially with objective count; standard_mobo calls it
# directly every iteration on qLogEHVI's exact acquisition path).
# 2-objective benchmarks (langermann3_ackley_2obj_6d, ackley_griewank_2obj_6d)
# complete fine. Before rerunning either 5-obj "standard" job, either bump
# memory well past 64g or investigate whether this is inherent to
# standard_mobo at M=5 regardless of memory (i.e. genuinely needs a
# different/approximate HV formulation, not just more RAM).
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

# High-suite: one pair only (spherical). EVAL BUDGET SCALED BY DIMENSION --
# the first run used tau315's own fixed 45-eval protocol (5 init + 40
# adaptive) unchanged even at d=50/d=500, which is far too thin at that
# scale (this project's own convention elsewhere uses 600+ evals even at
# d=100). Low-dim (d<=10) benchmarks above are intentionally left at the
# unscaled 45-eval default -- it's already ~7.5x dim there, reasonable,
# and changing it would obsolete the 8 already-running low-dim jobs from
# the previous fix for no benefit.
#
# Scaling rule (dim > 10 only): n_init = clamp(round(dim/5), 10, 100),
# remaining = clamp(8*dim, 80, 400). d=50 -> n_init=10, remaining=400
# (410 total, ~9x the old budget). d=500 -> n_init=100, remaining=400
# (500 total, capped -- scaling remaining uncapped, 8*500=4000, would need
# an estimated 250+ hours per trial given observed per-eval cost, not
# tractable at any reasonable job length).
#
# IMPORTANT: only the "standard" pair (standard_mobo/composite_mobo)
# spends its budget as n_init + n_iter. "chebyshev"/"spherical" instead
# spend theirs as n_init + weights*per_weight -- --n-iter is SILENTLY
# IGNORED for those two pairs (verified directly: the first attempt at
# this scaling passed --n-iter here and it did nothing, since the two
# high-suite benchmarks only ever run the "spherical" pair -- their
# results came back at 50/140 total evals instead of the intended
# 410/500). Below, "remaining" is spent as weights=8 (unchanged pareto/
# simplex coverage), per_weight=round(remaining/8).
#
# Empirical basis for the time estimates below:
# the ORIGINAL 45-eval run's per-trial time (direct+composite combined,
# 90 evals total) was ~450-500s at BOTH d=50 and d=500 -- i.e. cost is
# not dimension-driven in this range (the spherical kernel handles that
# fine), it's driven by iteration count and the exact GP's accumulating
# training set. Scaling linearly from that 90-evals/~500s baseline to
# 820/1000 combined evals gives ~75-90 min/trial; a real 32-core timing
# probe of the accumulating-training-set cost itself did not finish in a
# reasonable check-in window, so these times use a conservative LINEAR
# extrapolation (if per-iteration cost instead grows worse than linear as
# the training set accumulates, actual time will be higher than this
# estimate -- the generous time budgets below have headroom for that, but
# if a job runs past its limit, that means growth is worse than assumed
# and the budget needs revisiting, not just extending).
submit ackley_griewank_2obj_50d    spherical  48:00:00 64g  --source tau --benchmark ackley_griewank_2obj_50d --n-init 10 --weights 8 --per-weight 50
submit projected_langermann_500d   spherical  48:00:00 96g  --source tau --benchmark projected_langermann_2obj_500d --n-init 100 --weights 8 --per-weight 50

echo "Submitted 12 jobs. Check with: squeue -u \$USER"
echo "Per-job console logs: cluster/logs/composite-ablation-<job>-<pair>_<jobid>.out"
echo "Raw HV traces + per-pair summary: composite_ablation/results/<job>/"
echo "Once all pairs for a benchmark finish: python -m composite_ablation.plot_ablation composite_ablation/results/<job>"
