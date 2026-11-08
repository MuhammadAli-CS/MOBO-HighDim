#!/usr/bin/env bash
# More paired runs for DTLZ2 ("Synthetic 1" in the paper), at the paper's
# own reference point, to shrink the error bars that made that row
# uninformative (LogEHVI +38.4 +- 14.7, STCH +69.9 +- 38.9).
#
# WHY A RE-RUN AND NOT A RE-ANALYSIS
# The paper measures hypervolume against ref (1.1, 1.1) -- the standard
# DTLZ2 reference, exact maximum 1.1^2 - pi/4 = 0.4246.  The stored runs in
# composite_ablation/results_tau_repo/dtlz2_2obj_6d used the upstream repo's
# (2.5, 2.5) (tau_benchmarks.py:DTLZ2_2OBJ_6D), and run_ablation.py used to
# save ONLY the hypervolume traces, not the objective values they were
# computed from.  Hypervolume is a function of (Y, ref), so with Y gone the
# old traces cannot be re-referenced.  run_ablation.py now stores Y, so this
# is the last time a reference-point change costs a re-run.
#
# SEEDS
# run_ablation.py: seed = seed0 + 10_007 * trial.  Block b uses
# seed0 = 20 * 10007 * b, i.e. trials 20b..20b+19 of the same sequence --
# disjoint from the original --seed 0 runs and from each other, so the
# blocks concatenate into one clean paired sample.  Direct and Composite
# share the seed within a trial, so pairing is preserved.
#
# Block 0 re-runs the ORIGINAL 20 trials at the new reference point; the old
# numbers are at the wrong reference and cannot be pooled with these.
#
# Usage:  bash cluster/submit_dtlz2_more_seeds.sh [n_blocks]   (default 4)
#         4 blocks x 20 trials = 80 paired runs per arm.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs composite_ablation/results_blocks

BLOCKS=${1:-4}
TRIALS=20
STRIDE=10007          # must match run_ablation.py's seed stride
REF="1.1"             # paper's DTLZ2 reference point, both objectives

for pair in standard chebyshev; do
  for b in $(seq 0 $((BLOCKS - 1))); do
    seed0=$(( TRIALS * STRIDE * b ))
    out="composite_ablation/results_blocks/dtlz2_2obj_6d_ref11_b${b}"
    mkdir -p "$out"
    sbatch --requeue \
      --job-name="dtlz2-ref11-${pair}-b${b}" \
      --output="cluster/logs/dtlz2-ref11-${pair}-b${b}_%j.out" \
      --error="cluster/logs/dtlz2-ref11-${pair}-b${b}_%j.err" \
      --partition=aimi --account=kilian \
      --cpus-per-task=32 --mem=64g --time=08:00:00 \
      --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python -m composite_ablation.run_ablation --source tau --benchmark dtlz2_2obj_6d --pair ${pair} --trials ${TRIALS} --seed ${seed0} --ref-point ${REF} --num-threads 8 --num-interop-threads 4 --out-dir ${out}"
  done
done

echo "Submitted $(( 2 * BLOCKS )) jobs: 2 pairs x $BLOCKS blocks x $TRIALS trials."
echo "Pool when they finish:"
echo "  python -m composite_ablation.pool_blocks composite_ablation/results_blocks/dtlz2_2obj_6d_ref11_b*"
