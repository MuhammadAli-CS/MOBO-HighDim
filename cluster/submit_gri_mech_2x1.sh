#!/usr/bin/env bash
# Composite modeling on a 64-dim GRI-Mech 3.0 rate-constant calibration
# benchmark (morbo/problems/composite_gri_mech.py; kinetics via Cantera,
# gri30.yaml -- see writeup/methods.tex's composite GRI-Mech subsection).
# Same direct-vs-composite A/B pattern as submit_snar_2x3.sh, but only
# 2 labels (morbo, composite_gri_mech) -- NOT the _pca/_ard_pca shape
# variants, since those hit a pre-existing crash in
# morbo/run_one_replication.py's TR-index-logging assertion
# (`assert len(tr_inds) == len(tr.X)`) discovered while running the SnAr
# composite x shape ablation; re-add once that's fixed.
#
# 2 labels x 5 seeds = 10 jobs. dim=64 (>50D, deliberately chosen: no
# chemistry benchmark surveyed reaches 50D without an artificial
# parameterization -- kinetic mechanism rate-constant calibration is the
# one that does so naturally, see composite_gri_mech.py's module
# docstring). Per-evaluation cost is cheap (six 0D constant-pressure
# Cantera ignition sims, ~1-2s combined), so 400 evals/trial is
# affordable; NO TIME LIMIT (--time=0) since composite mode's 48-output
# ModelListGP fit cost per iteration is unverified at cluster scale.
#
# Usage: bash cluster/submit_gri_mech_2x1.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs

submit() {
  local exp=$1 label=$2 seed=$3
  sbatch --requeue \
    --job-name="${label}-${exp}-s${seed}" \
    --export=EXP="$exp",LABEL="$label",SEED="$seed" \
    --parsable \
    --time=0 \
    cluster/run_experiment.sub
}

EXP=gri_mech_composite
SEEDS=(0 1 2 3 4)
LABELS=(morbo composite_gri_mech)

echo "Submitting $EXP jobs (GRI-Mech 3.0 rate calibration, d=64, M=2)..."
IDS=()
for SEED in "${SEEDS[@]}"; do
  for LABEL in "${LABELS[@]}"; do
    J=$(submit "$EXP" "$LABEL" "$SEED"); IDS+=("$J")
  done
done

deps=$(IFS=:; echo "${IDS[*]}")
sbatch --requeue \
  --job-name="plot-${EXP}" \
  --dependency=afterany:$deps \
  --partition=aimi --account=kilian \
  --cpus-per-task=4 --mem=8g --time=1:00:00 \
  --wrap="cd $(pwd); . /share/apps/software/anaconda3/etc/profile.d/conda.sh; conda activate \$HOME/morbo-env; python plot_aggregate.py $EXP --labels ${LABELS[*]}"

echo "Submitted ${#IDS[@]} jobs + 1 dependent plot job."
echo "Check with: squeue -u \$USER"
