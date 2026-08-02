#!/usr/bin/env bash
# Runs tau315/composite-mobo's newest benchmarks (added upstream in commit
# 41ab952, "add new benchmarks and scripts", which also RETIRED the five
# benchmarks cluster/submit_tau_repo_full.sh targets -- ackley_griewank_*,
# five_ackley_5obj_6d, langermann3_ackley_2obj_6d,
# projected_langermann_2obj_500d -- see composite_ablation/
# run_from_tau_repo.py's TAU_PROBLEMS comment). Same direct-execution
# pattern as submit_tau_repo_full.sh: his own solvers.py/benchmark_*.py,
# our own morbo engine for the morbo pair.
#
# nanoparticle_rgb_3obj_6d: real Mie-scattering multilayer-nanoparticle
#   structural-color benchmark (6D, 3 objectives, low suite).
# summit_snar_2obj_4d: tau315's OWN independent RK4 reimplementation of
#   Summit's SnAr reactor (4D, 2 objectives, low suite) -- distinct from
#   this project's own composite_snar.py/"snar_4d_2obj_ours" wrapper
#   already in submit_tau_repo_full.sh; a genuinely different
#   implementation of the same real chemistry as an independent check.
# dtlz2_2obj_100d / dtlz2_2obj_600d: synthetic DTLZ2 at two more scales
#   (high suite: spherical/morbo only, same gating rule as elsewhere).
#
# NOT included here: cort_tg119_3obj_418d (real CORT TG-119 radiotherapy
# dose-optimization benchmark, 418D) -- its own benchmark_cort_tg119.py
# downloads the CORT dataset from GigaDB on first use, which needs
# explicit go-ahead before running on the cluster (see
# cluster/submit_tau_repo_cort_tg119.sh, submitted separately once
# confirmed).
#
# Usage: bash cluster/submit_tau_repo_new_benchmarks.sh
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
for bm in nanoparticle_rgb_3obj_6d summit_snar_2obj_4d; do
  submit "$bm" standard   128g "$bm"
  submit "$bm" chebyshev  128g "$bm"
  submit "$bm" morbo      128g "$bm"
done

# High-suite: spherical, morbo only.
for bm in dtlz2_2obj_100d dtlz2_2obj_600d; do
  submit "$bm" spherical  128g "$bm"
  submit "$bm" morbo      128g "$bm"
done

echo "Submitted 10 jobs. Check with: squeue -u \$USER"
echo "Per-job console logs: cluster/logs/tau-repo-<job>-<pair>_<jobid>.out"
echo "Raw HV traces + per-pair summary: composite_ablation/results_tau_repo/<job>/"
