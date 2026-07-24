#!/usr/bin/env bash
# Runs tau315/composite-mobo's OWN code directly (cloned into
# composite_ablation/tau315_repo/) -- ALL solvers in that repo's
# solvers.py (including MORBO/composite_morbo this time, driven by THIS
# PROJECT'S OWN morbo engine -- see composite_ablation/run_from_tau_repo.py's
# module docstring for exactly how/why that's safe) against ALL SIX of
# that repo's own benchmarks, plus this project's own five-objective/
# six-dimension DTLZ2 benchmark (dtlz2_5obj_6d_ours).
#
# Supersedes cluster/submit_composite_ablation.sh's hand-vendored
# approach (composite_ablation/solvers.py/tau_benchmarks.py) for the
# non-MORBO solvers: running his own files directly instead of a
# hand-vendored copy avoids re-deriving each solver family's own scaling
# knobs by hand, which already produced two real, cluster-hours-wasting
# bugs (see run_from_tau_repo.py's docstring). MORBO is a genuinely new
# addition here, not previously run through this ablation at all.
#
# ONE JOB PER (BENCHMARK, PAIR). Suite-gated exactly like tau315's own
# protocol (benchmark_common.py's _solver_jobs): "low"-suite benchmarks
# run standard/chebyshev/morbo; "high"-suite (his 50D/500D benchmarks)
# run spherical/morbo only -- the plain-kernel solvers at 500D would not
# be a meaningful test (that's exactly the pathology the spherical kernel
# exists to fix, and why his own repo never runs them there either).
#
# Eval budget scales with dimension (dim<=10 keeps the fixed 45-eval
# default; dim>10 scales up) -- see run_from_tau_repo.py's docstring for
# the exact formula per solver family.
#
# NO TIME LIMIT (--time=0, honored since the aimi partition's own limit
# is "infinite" per `sinfo`) -- these are expected to take a long time,
# especially the 500D and 5-objective benchmarks.
#
# 19 jobs total: 5 low-suite benchmarks (dtlz2_2obj_6d,
# ackley_griewank_2obj_6d, five_ackley_5obj_6d, langermann3_ackley_2obj_6d,
# dtlz2_5obj_6d_ours) x 3 pairs (standard, chebyshev, morbo) = 15, plus
# 2 high-suite benchmarks (ackley_griewank_2obj_50d,
# projected_langermann_2obj_500d) x 2 pairs (spherical, morbo) = 4.
#
# Usage: bash cluster/submit_tau_repo_full.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs composite_ablation/results_tau_repo

TRIALS=20

submit() {
  local job_name=$1 pair=$2 mem=$3
  shift 3
  local args=("$@")
  sbatch --requeue \
    --job-name="tau-repo-${job_name}-${pair}" \
    --output="cluster/logs/tau-repo-${job_name}-${pair}_%j.out" \
    --error="cluster/logs/tau-repo-${job_name}-${pair}_%j.err" \
    --partition=aimi --account=kilian \
    --cpus-per-task=32 --mem="$mem" --time=0 \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python -m composite_ablation.run_from_tau_repo --benchmark ${args[*]} --pair $pair --trials $TRIALS --num-threads 8 --num-interop-threads 4 --out-dir composite_ablation/results_tau_repo/${job_name}"
}

# Low-suite: standard, chebyshev, morbo.
for bm in dtlz2_2obj_6d ackley_griewank_2obj_6d langermann3_ackley_2obj_6d; do
  submit "$bm" standard   128g "$bm"
  submit "$bm" chebyshev  128g "$bm"
  submit "$bm" morbo      128g "$bm"
done

# 5-objective low-suite benchmarks: OOM'd at 64g even solo in the earlier
# hand-vendored run (likely NondominatedPartitioning's box decomposition
# blowing up with objective count) -- generously over-provisioned here.
for bm in five_ackley_5obj_6d dtlz2_5obj_6d_ours; do
  submit "$bm" standard   256g "$bm"
  submit "$bm" chebyshev  256g "$bm"
  submit "$bm" morbo      256g "$bm"
done

# High-suite: spherical, morbo only.
submit ackley_griewank_2obj_50d    spherical  128g ackley_griewank_2obj_50d
submit ackley_griewank_2obj_50d    morbo      128g ackley_griewank_2obj_50d
submit projected_langermann_2obj_500d spherical 192g projected_langermann_2obj_500d
submit projected_langermann_2obj_500d morbo     192g projected_langermann_2obj_500d

echo "Submitted 19 jobs. Check with: squeue -u \$USER"
echo "Per-job console logs: cluster/logs/tau-repo-<job>-<pair>_<jobid>.out"
echo "Raw HV traces + per-pair summary: composite_ablation/results_tau_repo/<job>/"
