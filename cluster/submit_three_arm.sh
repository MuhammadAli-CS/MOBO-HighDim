#!/usr/bin/env bash
# Three-arm composite experiment (knowing g vs observing h): one job per outer
# map, each running all seeds and all three arms.
#
# The map list is read from three_arm.test_functions.GROUPS so it cannot drift
# out of sync with the code.
#
# Usage:
#   bash cluster/submit_three_arm.sh              # every group
#   bash cluster/submit_three_arm.sh sweep        # one group
#   SEEDS=20 BUDGET=40 bash cluster/submit_three_arm.sh pairs
#
# Validate the arm-3 sampler before spending a queue on it:
#   python three_arm/run.py --check
#
# Collect the results once the jobs land:
#   python three_arm/run.py --collect
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cluster/logs three_arm/results

SEEDS="${SEEDS:-10}"
BUDGET="${BUDGET:-40}"
GROUPS_TO_RUN=("$@")
if [ ${#GROUPS_TO_RUN[@]} -eq 0 ]; then
  GROUPS_TO_RUN=(validate sweep pairs analogs)
fi

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$HOME/morbo-env"

# De-duplicate: maps appear in more than one group (shiftsq_c0 is in both the
# sweep and the matched pairs), and each one only needs to run once.
MAPS=$(python - "${GROUPS_TO_RUN[@]}" <<'PY'
import sys
from three_arm.test_functions import GROUPS
seen = dict.fromkeys(m for g in sys.argv[1:] for m in GROUPS[g])
print(" ".join(seen))
PY
)

echo "submitting $(echo "$MAPS" | wc -w) jobs (seeds=$SEEDS budget=$BUDGET): $MAPS"
for map in $MAPS; do
  sbatch --requeue \
    --job-name="three-arm-${map}" \
    --export=MAP="$map",SEEDS="$SEEDS",BUDGET="$BUDGET" \
    --parsable \
    cluster/three_arm.sub
done
