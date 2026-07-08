#!/usr/bin/env bash
# Sequential overnight run of the remaining Part 1 experiments (correlation
# ablation + composite Penicillin). No resume support exists in
# run_one_replication.py, so each label reruns from scratch. One process at a
# time to keep memory bounded (Kronecker GP + Penicillin's 2500-step
# simulator are the two suspected RAM/disk hogs from the killed run).
set -uo pipefail
cd "$(dirname "$0")"
LOG=overnight_run.log
exec > >(tee -a "$LOG") 2>&1

mem_report() {
  echo "--- $(date '+%Y-%m-%d %H:%M:%S') mem: $(systeminfo 2>/dev/null | grep 'Available Physical Memory')"
}

run_step() {
  local exp=$1 label=$2 seed=$3
  echo "=== [$(date '+%H:%M:%S')] START $exp/$label seed=$seed ==="
  mem_report
  python run_comparison.py "$exp" "$label" "$seed"
  status=$?
  echo "=== [$(date '+%H:%M:%S')] END $exp/$label exit=$status ==="
  mem_report
  return $status
}

echo "############ overnight run starting $(date) ############"

run_step correlation_ablation_dtlz2curve morbo 0 && \
run_step correlation_ablation_dtlz2curve independent_gp_composite 0 && \
run_step correlation_ablation_dtlz2curve kronecker_gp_composite 0
echo "=== plotting correlation_ablation_dtlz2curve ==="
python plot_comparison.py correlation_ablation_dtlz2curve 0

run_step penicillin_composite morbo 0 && \
run_step penicillin_composite composite_penicillin 0
echo "=== plotting penicillin_composite ==="
python plot_comparison.py penicillin_composite 0

echo "############ overnight run finished $(date) ############"
