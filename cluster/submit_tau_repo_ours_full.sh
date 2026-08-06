#!/usr/bin/env bash
# Run ALL of this project's own composite benchmarks through tau315's
# multi-method pipeline with the FULL solver set, for a uniform cross-method
# comparison ("all three" plain solvers -- standard/chebyshev/morbo -- plus
# spherical STCH on the high-dim ones). Uses the ports in
# composite_ablation/tau_benchmarks_ours/ (registered in
# run_from_tau_repo.py's TAU_PROBLEMS), our own morbo engine for the morbo
# pair via import-caching.
#
# Benchmarks and pairs (trimmed to the three we still need):
#   snar_4d_2obj_ours (4D, low)   : standard, chebyshev, morbo
#       -- the CLEANED SnAr (6th raw component is now the product molar flow
#          F_product, not the closed-form-known q_tot; objectives bit-identical)
#   rcm46_4obj_34d (34D, high)    : standard, chebyshev, spherical, morbo
#   moopf_5obj_34d (34D, high)    : standard, chebyshev, spherical, morbo
#       -- MOOPF-5, RCM46's 4 objectives + L-index (constructed benchmark)
# (RCM40 and Penicillin are already covered / not needed, so left out.)
#
# The 34D benchmarks are "high" suite, so standard/chebyshev are off-suite --
# passed --allow-any-pair to run them anyway (34D is well within the plain
# solvers' reach; the gate exists for the 500D+ regime). --allow-any-pair is
# a harmless no-op on on-suite pairs.
#
# Usage: bash cluster/submit_tau_repo_ours_full.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs composite_ablation/results_tau_repo

TRIALS=20

submit() {
  local job_name=$1 pair=$2
  sbatch --requeue \
    --job-name="tau-repo-${job_name}-${pair}" \
    --output="cluster/logs/tau-repo-${job_name}-${pair}_%j.out" \
    --error="cluster/logs/tau-repo-${job_name}-${pair}_%j.err" \
    --partition=aimi --account=kilian \
    --cpus-per-task=32 --mem=128g --time=0 \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python -m composite_ablation.run_from_tau_repo --benchmark ${job_name} --pair ${pair} --trials ${TRIALS} --allow-any-pair --num-threads 8 --num-interop-threads 4 --out-dir composite_ablation/results_tau_repo/${job_name}"
}

# Low-suite (4D): the three plain solvers. Cleaned SnAr only.
for bm in snar_4d_2obj_ours; do
  for pair in standard chebyshev morbo; do submit "$bm" "$pair"; done
done

# High-suite (34D): the three plain solvers PLUS spherical STCH.
for bm in rcm46_4obj_34d moopf_5obj_34d; do
  for pair in standard chebyshev spherical morbo; do submit "$bm" "$pair"; done
done

echo "Submitted 11 jobs (1x3 SnAr + 2x4 RCM46/MOOPF-5). Check with: squeue -u \$USER"
echo "Results: composite_ablation/results_tau_repo/<benchmark>/<pair>_-_*.npz"
