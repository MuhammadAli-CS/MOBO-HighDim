#!/usr/bin/env bash
# Two-objective experiment: x -> h1 -> g1 and x -> h2 -> g2, one job per
# (g1 map, inner function draw).
#
# g2 is the identity, so objective 2 is a plain GP and both arms model it
# identically.  Every difference between the arms therefore has to come from
# g1, which is what makes the comparison across g1 maps meaningful.
#
# Usage:
#   bash cluster/submit_two_stage.sh                      # every g1 map
#   DRAWS="0 1 2 3" TRIALS=5 bash cluster/submit_two_stage.sh
#   G1S="mono3 shiftsq_c0" bash cluster/submit_two_stage.sh
#
# Collect once the jobs land:
#   python composite_ablation/collect_two_stage.py --out composite_ablation/results/two_stage_grid
set -euo pipefail
cd "$(dirname "$0")/.."

DRAWS="${DRAWS:-0 1 2 3}"
TRIALS="${TRIALS:-5}"
DIM="${DIM:-6}"
N_INIT="${N_INIT:-10}"
N_ITER="${N_ITER:-40}"
G2="${G2:-identity}"
KERNEL="${KERNEL:-matern}"
JOBS="${JOBS:-5}"
OUT="${OUT:-composite_ablation/results/two_stage_grid}"
ENV_PATH="${ENV_PATH:-$HOME/morbo-env}"

mkdir -p cluster/logs "$OUT"

if [ ! -d "$ENV_PATH" ]; then
  echo "conda env not found at $ENV_PATH." >&2
  echo "Create it once with:      bash cluster/setup_env.sh" >&2
  echo "Or point at another one:  ENV_PATH=/path/to/env bash $0" >&2
  exit 1
fi

source /share/apps/software/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

# The default map list is READ FROM THE CODE, not repeated here: hardcoding it
# meant that renaming cube/square/sinh to mono3/shiftsq_c0/sinh_b1 left this
# script asking for maps that no longer existed.  An explicit G1S is still
# validated against the same source.  This doubles as the preflight, so a bad
# env fails on the login node rather than in every queued job.
G1S=$(python - "${G1S:-}" <<'PY'
import importlib.util, os, sys
missing = [m for m in ("torch", "botorch", "gpytorch")
           if importlib.util.find_spec(m) is None]
if missing:
    sys.exit("missing packages in the conda env: " + ", ".join(missing))
sys.path.insert(0, os.getcwd())
try:
    from composite_ablation.two_stage import OUTER_G1
except ImportError as exc:
    sys.exit(f"cannot import composite_ablation ({exc}) -- repo checked out here?")
asked = sys.argv[1].split()
if not asked:
    print(" ".join(OUTER_G1))
else:
    bad = [g for g in asked if g not in OUTER_G1]
    if bad:
        sys.exit(f"unknown g1 map(s) {bad}; choose from {sorted(OUTER_G1)}")
    print(" ".join(asked))
PY
) || exit 1

n=$(( $(echo "$G1S" | wc -w) * $(echo "$DRAWS" | wc -w) ))
echo "submitting $n jobs"
echo "  g1:     $G1S"
echo "  draws:  $DRAWS   trials: $TRIALS   dim: $DIM   n_iter: $N_ITER"
echo "  g2:     $G2   kernel: $KERNEL   out: $OUT"

# slurmctld intermittently returns "Socket timed out on send/recv operation".
# Under `set -e` that killed the whole loop, so one flaky call left 8 of 84
# jobs queued and no indication which.  Retry each submission, and on
# persistent failure record it and carry on rather than abandoning the rest.
submitted=0
failed=()
skipped=0
for g1 in $G1S; do
  for draw in $DRAWS; do
    if [ -n "${SKIP_EXISTING:-}" ] && [ -f "$OUT/${g1}_d${draw}.json" ]; then
      skipped=$((skipped + 1))
      continue
    fi
    id=""
    for attempt in 1 2 3 4 5; do
      if id=$(sbatch --requeue \
          --job-name="two-stage-${g1}-d${draw}" \
          --export=G1="$g1",DRAW="$draw",TRIALS="$TRIALS",DIM="$DIM",N_INIT="$N_INIT",N_ITER="$N_ITER",G2="$G2",KERNEL="$KERNEL",JOBS="$JOBS",OUT="$OUT",ENV_PATH="$ENV_PATH" \
          --parsable \
          cluster/two_stage.sub 2>&1); then
        break
      fi
      echo "  retry $attempt for ${g1} d${draw}: $id" >&2
      sleep $((attempt * 3))
      id=""
    done
    if [ -z "$id" ]; then
      failed+=("${g1}_d${draw}")
    else
      submitted=$((submitted + 1))
      echo "$id"
    fi
  done
done

echo
echo "submitted $submitted of $n$( [ "$skipped" -gt 0 ] && echo ", skipped $skipped already done" )"
if [ ${#failed[@]} -gt 0 ]; then
  echo "FAILED to submit ${#failed[@]}: ${failed[*]}" >&2
  echo "Re-run with SKIP_EXISTING=1 once the queued jobs have written their" >&2
  echo "output, and only the missing ones will be submitted." >&2
  exit 1
fi
