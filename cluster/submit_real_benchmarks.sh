#!/usr/bin/env bash
# Real-problem benchmarks for the tr_shape study:
#
# 1. LassoBench-MO (lasso_synt_medium_mo, lasso_synt_high_mo, lasso_dna_mo)
#    -- bi-objective LassoBench (validation loss + active-coefficient
#    fraction). Protocol MATCHES THE LASSOBENCH PAPER (Sehic et al., AutoML
#    2022) so our best-loss curves are directly comparable to their Table 2:
#    1000 evals for synt_medium (d=100, effective dim 5) and DNA (d=180,
#    effective dim 43), 5000 evals for synt_high (d=300, effective dim 15),
#    and THIRTY repetitions (seeds 0-29) per method, matching their "30
#    repetitions for each method".
#    REQUIRES LassoBench installed in the morbo-env (checked below).
#
# 2. SparseRover (sparse_rover_d120, sparse_rover_d180) -- real Rover
#    trajectory objective embedded in 2x/3x nominal dims (extra dims are
#    no-ops). Rover's own protocol (2000 evals), 5 seeds matching
#    tr_shape_rover.
#
# Job counts per stage (6 methods each):
#   synt_medium: 180   dna: 180   synt_high: 180 (5000-eval jobs, LONG)
#   sparse_rover: 60 (2 dims x 5 seeds x 6 methods)
#
# Usage:
#   bash cluster/submit_real_benchmarks.sh                # stage 1: synt_medium + sparse_rover (240 jobs)
#   bash cluster/submit_real_benchmarks.sh dna            # + lasso_dna_mo (180 jobs)
#   bash cluster/submit_real_benchmarks.sh synt_high      # + lasso_synt_high_mo (180 LONG jobs)
#   bash cluster/submit_real_benchmarks.sh all            # everything (600 jobs)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

STAGE="${1:-default}"

# LassoBench install check -- fail fast rather than filling the queue with
# 180 jobs that all die on the same ImportError.
if ! . /share/apps/software/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
   ! conda activate "$HOME/morbo-env" 2>/dev/null || \
   ! python -c "import LassoBench" 2>/dev/null; then
  echo "ERROR: LassoBench is not importable in ~/morbo-env." >&2
  echo "Install it first (from anywhere with internet, e.g. the login node):" >&2
  echo "  conda activate \$HOME/morbo-env" >&2
  echo "  git clone https://github.com/ksehic/LassoBench.git ~/LassoBench" >&2
  echo "  pip install -e ~/LassoBench" >&2
  echo "Then re-run this script. (SparseRover jobs don't need it -- to submit" >&2
  echo "only those, comment out this check.)" >&2
  exit 1
fi

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
    --job-name="plot-${exp}" \
    --dependency=afterok:"$deps" \
    --partition=aimi --account=kilian \
    --cpus-per-task=1 --mem=4g --time=00:15:00 \
    --output="cluster/logs/plot-${exp}_%j.out" \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_comparison.py $exp 0"
}

METHODS=(morbo pca_ellipsoid ard_pca_ellipsoid cma_ellipsoid mab_shape sobol)

run_experiment_batch() {
  local exp=$1 n_seeds=$2
  echo "=== $exp (${n_seeds} seeds x ${#METHODS[@]} methods) ==="
  IDS=()
  for ((SEED=0; SEED<n_seeds; SEED++)); do
    for LABEL in "${METHODS[@]}"; do
      J=$(submit "$exp" "$LABEL" "$SEED"); IDS+=("$J")
    done
  done
  echo "  submitted ${#IDS[@]} jobs (first=$( echo "${IDS[0]}" ), last=${IDS[-1]})"
  submit_plot "$exp" "${IDS[@]}"
}

# Stage 1 (default): the cheapest LassoBench benchmark + SparseRover.
if [[ "$STAGE" == "default" || "$STAGE" == "all" ]]; then
  run_experiment_batch lasso_synt_medium_mo 30
  run_experiment_batch sparse_rover_d120 5
  run_experiment_batch sparse_rover_d180 5
fi
if [[ "$STAGE" == "dna" || "$STAGE" == "all" ]]; then
  run_experiment_batch lasso_dna_mo 30
fi
if [[ "$STAGE" == "synt_high" || "$STAGE" == "all" ]]; then
  run_experiment_batch lasso_synt_high_mo 30
fi

echo
echo "Done. Check with: squeue -u \$USER"
echo "Aggregate with: python aggregate_seeds.py <experiment_name>"
