#!/usr/bin/env bash
# cma_turbo_style: a direct ablation of cma_ellipsoid replicating CMA-TuRBO's
# own mechanism (Ngo, Ha, Chan, Nguyen & Zhang, "High-dimensional Bayesian
# Optimization via Covariance Matrix Adaptation Strategy," TMLR 2024,
# arXiv:2402.03104) instead of our own simplification -- verified directly
# from the paper's PDF (Section 4.2.2, Eq. 4/6) AND the paper's own
# reference implementation (github.com/LamNgo1/cma-meta-algorithm):
#
#   - now wraps the actual `cma` PyPI package directly (a genuine
#     cma.CMAEvolutionStrategy per trust region, fed via ask()/tell() each
#     batch) instead of a hand-rolled rank-mu/evolution-path update --
#     requires the `cma` package, see cluster/README.md section 4k;
#   - candidates drawn by DIRECT multivariate-Gaussian sampling from that
#     object's own adapted mean/step-size/covariance
#     (sample_tr_gaussian_ellipsoid), reproducing the reference's
#     create_candidates closure line-for-line -- not rotated-box
#     perturbation, the representation every other tr_shape mode uses.
#
# See get_or_init_cma_turbo_es/update_cma_turbo_es/
# sample_tr_gaussian_ellipsoid's docstrings (morbo/utils.py) for the exact
# procedure and the multi-objective ranking substitution (the source paper
# is single-objective; we rank by mean per-objective min-max value, the
# same substitution already established for labcat_style).
#
# Targeted rollout (not a full-study sweep, matching how labcat_style was
# originally scoped before later being run everywhere): the core headline
# comparison, cma_ellipsoid's own strongest documented landscape (rotated
# effective subspace), and two other landscapes where cma_ellipsoid already
# has recorded results to compare against directly.
#
# 4 experiments x 5 seeds = 20 jobs.
#
# Usage: bash cluster/submit_cma_turbo_style.sh
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
    --job-name="plot-${exp}" \
    --dependency=afterok:"$deps" \
    --partition=aimi --account=kilian \
    --cpus-per-task=1 --mem=4g --time=00:15:00 \
    --output="cluster/logs/plot-${exp}_%j.out" \
    --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_aggregate.py $exp"
}

SEEDS=(0 1 2 3 4)
LABEL=cma_turbo_style
EXPERIMENTS=(
  tr_shape_dtlz2_100d
  tr_shape_methods_dtlz2_100d
  rotated_sparse_dtlz2_d100_keff50
  bbob_rosenbrock_rosenbrock
)

for EXP in "${EXPERIMENTS[@]}"; do
  echo "=== $EXP ==="
  IDS=()
  for SEED in "${SEEDS[@]}"; do
    J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
  done
  echo "  submitted ${#IDS[@]} jobs"
  submit_plot "$EXP" "${IDS[@]}"
done

echo
echo "Done (${#EXPERIMENTS[@]} experiments x ${#SEEDS[@]} seeds = $((${#EXPERIMENTS[@]} * ${#SEEDS[@]})) jobs)."
echo "Check with: squeue -u \$USER"
echo "Compare against cma_ellipsoid with, e.g. (cma_ellipsoid lives in"
echo "tr_shape_methods_dtlz2_100d, not tr_shape_dtlz2_100d):"
echo "  python compute_significance.py tr_shape_methods_dtlz2_100d cma_turbo_style --baseline-label cma_ellipsoid"
echo "  python compute_significance.py rotated_sparse_dtlz2_d100_keff50 cma_turbo_style --baseline-label cma_ellipsoid"
echo "  python compute_significance.py bbob_rosenbrock_rosenbrock cma_turbo_style --baseline-label cma_ellipsoid"
echo "  python compute_significance.py tr_shape_dtlz2_100d cma_turbo_style --baseline-label pca_ellipsoid"
