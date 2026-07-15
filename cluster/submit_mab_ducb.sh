#!/usr/bin/env bash
# mab_shape_ducb: discounted-UCB bandit over shapes, targeting the exact
# regimes where epsilon-greedy mab_shape was measured to fail
# (RESULTS.md sec 11d/11e):
#
# 1. tv_sparse_dtlz2_d100_keff49 (20 seeds) -- THE discriminating test:
#    epsilon-greedy scored +2.9% (10/20, std 3.2) here while both fixed
#    shapes hit +20% at 20/20, because its per-arm reward estimates go
#    stale at the mid-run switch. D-UCB's regrowing exploration bonus is
#    the textbook fix; if it doesn't close most of that gap, the design
#    argument is wrong.
# 2. tr_shape_methods_dtlz2_100d (20 seeds) -- the variance test:
#    epsilon-greedy averaged +33.5% with std 7.2 (some seeds lock onto
#    good arms, some don't). D-UCB's guaranteed round-robin init + bonus
#    schedule should cut the tail risk.
# 3. tr_shape_methods_dtlz2_200d (5 seeds) -- the tight-budget test:
#    epsilon-greedy tied the failing baseline at d=200/600ev (fixed
#    exploration tax); UCB bonuses anneal, so ducb may keep more of
#    cma_ellipsoid's breakthrough (the arm that wins there).
#
# 20 + 20 + 5 = 45 jobs.
# Usage: bash cluster/submit_mab_ducb.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

submit() {
  local exp=$1 label=$2 seed=$3
  sbatch --requeue \
    --job-name="${label}-${exp}-s${seed}" \
    --export=EXP="$exp",LABEL="$label",SEED="$seed" \
    --parsable \
    cluster/run_experiment.sub
}

submit_plot() {
  local exp=$1; shift
  local deps; deps=$(IFS=:; echo "$*")
  sbatch --requeue \
    --job-name="plot-${exp}-ducb" \
    --dependency=afterok:"$deps" \
    --partition=aimi --account=kilian \
    --cpus-per-task=1 --mem=4g --time=00:15:00 \
    --output="cluster/logs/plot-${exp}-ducb_%j.out" \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_aggregate.py $exp"
}

for SPEC in "tv_sparse_dtlz2_d100_keff49 20" "tr_shape_methods_dtlz2_100d 20" "tr_shape_methods_dtlz2_200d 5"; do
  EXP=${SPEC% *}; NSEEDS=${SPEC#* }
  echo "=== $EXP (${NSEEDS} seeds) ==="
  IDS=()
  for ((SEED=0; SEED<NSEEDS; SEED++)); do
    J=$(submit "$EXP" "mab_shape_ducb" "$SEED"); IDS+=("$J")
  done
  echo "  submitted ${#IDS[@]} jobs"
  submit_plot "$EXP" "${IDS[@]}"
done

echo
echo "Done (45 jobs). Check with: squeue -u \$USER"
