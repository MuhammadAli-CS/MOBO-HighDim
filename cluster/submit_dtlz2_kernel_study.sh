#!/usr/bin/env bash
# Does the surrogate kernel explain the DTLZ2 discrepancy?
#
# The paper reports DTLZ2 at LogEHVI +38.4 +- 14.7 and STCH +69.9 +- 38.9.
# Nothing in either repo reproduces that: our ablation gives +3.60/+0.20 at
# ref 2.5, and tau315/composite-mobo's own stored results give +3.60/+0.06
# (ablation_results/) and +11.05/+5.68 (results/production-31f21c3/).
#
# But the two pipelines were never fitting the same model.  Neither repo
# passes covar_module to SingleTaskGP, so each inherits its BoTorch default:
#
#   this repo, botorch 0.9.5    ScaleKernel(Matern 5/2), Gamma priors
#   tau315,    botorch 0.18.0   bare RBF, LogNormal(sqrt2 + log(d)/2, sqrt3)
#                               lengthscale prior >= 0.025, no outputscale
#
# composite_ablation/solvers.py now exposes that as --surrogate, so this runs
# the identical pipeline under both kernels and attributes the difference.
#
# Held fixed against the ref250 arm of submit_dtlz2_refpoint_study.sh: same
# benchmark, same 45-evaluation budget, same seeds, same reference point.
# Only the kernel differs, so the comparison is clean.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs composite_ablation/results_blocks

TRIALS=20
STRIDE=10007
QUEUED=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)

for pair in standard chebyshev; do
  for b in 0 1 2 3; do
    seed0=$(( TRIALS * STRIDE * b ))
    name="dtlz2-rbf250-${pair}-b${b}"
    if grep -qx "$name" <<<"$QUEUED"; then echo "skip $name"; continue; fi
    out="composite_ablation/results_blocks/dtlz2_rbf250_b${b}"
    mkdir -p "$out"
    for try in 1 2 3 4 5; do
      if sbatch --requeue \
        --job-name="$name" \
        --output="cluster/logs/${name}_%j.out" \
        --error="cluster/logs/${name}_%j.err" \
        --partition=aimi --account=kilian \
        --cpus-per-task=32 --mem=64g --time=08:00:00 \
        --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python -m composite_ablation.run_ablation --source tau --benchmark dtlz2_2obj_6d --pair ${pair} --trials ${TRIALS} --seed ${seed0} --ref-point 2.5 --surrogate dimscaled_rbf --num-threads 8 --num-interop-threads 4 --out-dir ${out}"
      then break; fi
      echo "  sbatch failed for $name (attempt $try), retrying in 20s"; sleep 20
    done
  done
done
echo "Submitted up to 8 jobs (2 pairs x 4 blocks, n=80)."
echo "Compare against composite_ablation/results_blocks/dtlz2_ref250_b* (Matern 5/2)."
