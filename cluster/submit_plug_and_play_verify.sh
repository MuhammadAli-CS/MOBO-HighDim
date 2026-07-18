#!/usr/bin/env bash
# Verifies plug_and_play/run.py reproduces this project's own recorded
# results -- i.e. that the modularized methods.py/benchmarks.py interface
# is wired correctly against the real MORBO engine, not just "runs
# without crashing" at toy scale (already checked locally). Runs all 7
# methods on DTLZ2 (d=100, 600 evals, seed 0 -- the exact
# tr_shape_dtlz2_100d / tr_shape_methods_dtlz2_100d configuration) and
# compares each against the corresponding recorded seed-0 .pt value.
#
# A local run of just pca_ellipsoid already landed within ~0.4% of the
# recorded value (33.08 vs 32.95) -- consistent with ordinary
# floating-point non-determinism across different hardware/thread counts,
# not a wiring bug, since both paths call the identical, identically-seeded
# run_one_replication function. Running this ON THE CLUSTER (the same
# hardware the reference values were produced on) is the stronger version
# of that check, across every method, not just one.
#
# One job (not per-method): all 7 methods share DTLZ2's construction cost
# and run in well under the time limit sequentially; splitting into 7 jobs
# would just add SLURM overhead for a one-off verification, not a
# permanent result meant to be re-run/aggregated like the study's other
# experiments.
#
# Usage: bash cluster/submit_plug_and_play_verify.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

sbatch --requeue \
  --job-name=plug-and-play-verify \
  --output=cluster/logs/plug-and-play-verify_%j.out \
  --error=cluster/logs/plug-and-play-verify_%j.err \
  --partition=aimi --account=kilian \
  --cpus-per-task=32 --mem=64g --time=02:00:00 \
  --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plug_and_play/verify_reproduction.py --tolerance-pct 10"

echo "Submitted. Check with: squeue -u \$USER"
echo "Results land in cluster/logs/plug-and-play-verify_<jobid>.out"
