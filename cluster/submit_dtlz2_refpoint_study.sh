#!/usr/bin/env bash
# DTLZ2 (the paper's "Synthetic 1") reference-point study.
#
# WHY THIS EXISTS
# The paper's Appendix E.2 states d=6 with reference point (1.1, 1.1) and
# exact maximum hypervolume 1.1^2 - pi/4.  But 1.1 is the d=1 case of the
# upstream repo's own rule,
#     REFERENCE_VALUE = 1.0 + 0.25 * (DIM - 1) + 0.1
# (tau315_repo/benchmark_dtlz2_100d.py:11), which gives 2.35 at d=6; the 6-d
# benchmark file hardcodes 2.50.  At d=6 the distance term q reaches 1.25, so
# the objectives reach 2.25 -- far outside a reference of 1.1.  Running there
# for 45 evaluations, the STCH arm reached only 23.5% of the attainable
# hypervolume and SIX of 80 runs finished at exactly zero, which makes every
# ratio metric undefined.  So either the paper's reference point is wrong for
# d=6, or 45 evaluations is too small a budget for it.  This settles which.
#
# THE REFERENCE POINT IS NOT JUST A SCORE
# run_ablation.py passes problem.ref_point into the solver, and qLogEHVI uses
# it inside the acquisition function.  Re-scoring stored Y at another
# reference answers "what would these runs be worth there", NOT "what would
# running there produce" -- verified: block b0 shares its seeds with the
# original 20 runs, yet Direct's mean hypervolume at ref 2.5 is 5.3539 when
# driven by ref 1.1 against 5.2350 when driven by ref 2.5.  Each candidate
# therefore needs its own runs.
#
# ARMS (we already hold n=80 at ref 1.1, 45 evals, from
# cluster/submit_dtlz2_more_seeds.sh):
#   A  ref 2.50, 45 evals, 4 blocks -> n=80   the upstream 6-d file's value
#   B  ref 2.35, 45 evals, 2 blocks -> n=40   the upstream rule at d=6
#   C  ref 1.10, ~100 evals, 2 blocks -> n=40 is the paper's reference merely
#                                            starved of budget?
#
# SEEDS: run_ablation.py uses seed = seed0 + 10_007 * trial.  Block b takes
# seed0 = 20 * 10007 * b, so blocks are disjoint slices of one sequence and
# concatenate cleanly.  Direct and Composite share the seed within a trial.
#
# Usage:  bash cluster/submit_dtlz2_refpoint_study.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs composite_ablation/results_blocks

TRIALS=20
STRIDE=10007

# Jobs already queued or running, so re-running this script tops the grid up
# instead of double-submitting.  `|| true` because an empty queue makes
# squeue exit non-zero under `set -e`.
QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

submit() {
  local tag=$1 pair=$2 ref=$3 blocks=$4 time=$5
  shift 5
  local extra=("$@")
  for b in $(seq 0 $((blocks - 1))); do
    local seed0=$(( TRIALS * STRIDE * b ))
    local out="composite_ablation/results_blocks/dtlz2_${tag}_b${b}"
    local name="dtlz2-${tag}-${pair}-b${b}"
    if grep -qx "$name" <<<"$QUEUED"; then
      echo "skip $name (already queued)"
      continue
    fi
    mkdir -p "$out"
    # slurmctld intermittently drops a connection ("Socket timed out on
    # send/recv operation").  Under `set -e` that one hiccup abandoned the
    # rest of the grid -- 14 of 16 jobs were lost to it once.  Retry instead.
    local try
    for try in 1 2 3 4 5; do
      if sbatch --requeue \
      --job-name="$name" \
      --output="cluster/logs/${name}_%j.out" \
      --error="cluster/logs/${name}_%j.err" \
      --partition=aimi --account=kilian \
      --cpus-per-task=32 --mem=64g --time="$time" \
      --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python -m composite_ablation.run_ablation --source tau --benchmark dtlz2_2obj_6d --pair ${pair} --trials ${TRIALS} --seed ${seed0} --ref-point ${ref} --num-threads 8 --num-interop-threads 4 --out-dir ${out} ${extra[*]}"
      then
        break
      fi
      echo "  sbatch failed for $name (attempt $try), retrying in 20s"
      sleep 20
    done
  done
}

# --- A: upstream's hardcoded 6-d reference, full 80 runs -------------------
submit ref250 standard  2.5  4 08:00:00
submit ref250 chebyshev 2.5  4 08:00:00

# --- B: upstream's dimension rule at d=6, sensitivity check ----------------
submit ref235 standard  2.35 2 08:00:00
submit ref235 chebyshev 2.35 2 08:00:00

# --- C: the paper's reference with a real budget ---------------------------
# "standard" spends budget as n_init + n_iter; "chebyshev" spends it as
# n_init + weights * per_weight and SILENTLY IGNORES --n-iter (see
# cluster/submit_composite_ablation.sh).  10+90=100 and 10+8*11=98.
# Longer wall clock: per-iteration cost grows with the training set, so
# roughly doubling the budget costs far more than double.
submit ref110big standard  1.1 2 20:00:00 --n-init 10 --n-iter 90
submit ref110big chebyshev 1.1 2 20:00:00 --n-init 10 --weights 8 --per-weight 11

echo "Submitted 16 jobs."
echo "Pool each arm with, e.g.:"
echo "  python -m composite_ablation.pool_blocks composite_ablation/results_blocks/dtlz2_ref250_b* "
