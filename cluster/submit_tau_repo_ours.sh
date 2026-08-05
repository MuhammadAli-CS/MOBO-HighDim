#!/usr/bin/env bash
# Runs THIS PROJECT'S OWN composite benchmarks through tau315/composite-mobo's
# multi-method pipeline -- the same standard/chebyshev/spherical/morbo x
# direct/composite comparison already run on tau315's own set
# (cluster/submit_tau_repo_new_benchmarks.sh), now on our RCM40/RCM46/
# Penicillin as well, via the ports in composite_ablation/tau315_repo/
# benchmark_rcm40.py, benchmark_rcm46.py, benchmark_penicillin.py (registered
# in run_from_tau_repo.py's TAU_PROBLEMS).
#
# rcm40_2obj_34d / rcm46_4obj_34d: IEEE 14-bus Optimal Power Flow (CEC2021
#   RWCMOP suite), 34D, "high" suite -> spherical + morbo only (same gating
#   rule as dtlz2_100d/600d: plain-kernel solvers aren't a meaningful test
#   at this dimension).
# penicillin_3obj_7d: penicillin fermentation (7D, 3 objectives), "low"
#   suite -> standard + chebyshev + morbo.
#
# Same direct-execution pattern as submit_tau_repo_new_benchmarks.sh: tau315's
# own solvers.py/benchmark_common.py, our own morbo engine for the morbo pair.
#
# Usage: bash cluster/submit_tau_repo_ours.sh
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

# High-suite: spherical, morbo only.
for bm in rcm40_2obj_34d rcm46_4obj_34d; do
  submit "$bm" spherical  128g "$bm"
  submit "$bm" morbo      128g "$bm"
done

# Low-suite: standard, chebyshev, morbo.
submit penicillin_3obj_7d standard   128g penicillin_3obj_7d
submit penicillin_3obj_7d chebyshev  128g penicillin_3obj_7d
submit penicillin_3obj_7d morbo      128g penicillin_3obj_7d

echo "Submitted 7 jobs. Check with: squeue -u \$USER"
echo "Per-job console logs: cluster/logs/tau-repo-<job>-<pair>_<jobid>.out"
echo "Raw HV traces + per-pair summary: composite_ablation/results_tau_repo/<job>/"
