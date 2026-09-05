#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
LOG=experiments/work_dirs/_v33_logs
mkdir -p "experiments/work_dirs/_v33_logs"

echo "=== v3.3 downstream chain started $(date) ==="

echo "[$(date +%H:%M:%S)] STAGE 1/3 fusion battery"
CUDA_VISIBLE_DEVICES=0 bash scripts/pipelines/run_fusion.sh > "$LOG/fusion.log" 2>&1
rc=$?; echo "[$(date +%H:%M:%S)] STAGE 1 rc=$rc"
if [ $rc -ne 0 ]; then echo "CHAIN_ABORTED_AT_FUSION"; exit 1; fi
if ! grep -q FUSION_V33_DONE "$LOG/fusion.log"; then
    echo "CHAIN_ABORTED_AT_FUSION (no completion marker)"; exit 1; fi

echo "[$(date +%H:%M:%S)] STAGE 2/3 followup (GPU0) + STAGE 3/3 OOD (GPU1), parallel"
( CUDA_VISIBLE_DEVICES=0 bash scripts/pipelines/run_followup.sh > "$LOG/followup.log" 2>&1
  rc=$?; echo "[$(date +%H:%M:%S)] STAGE 2 rc=$rc" ) & P2=$!
( GPU=1 CUDA_VISIBLE_DEVICES=1 bash scripts/pipelines/run_ood_evals.sh > "$LOG/ood.log" 2>&1
  rc=$?; echo "[$(date +%H:%M:%S)] STAGE 3 rc=$rc" ) & P3=$!
wait $P2 $P3

echo "=== V33_CHAIN_FINISHED $(date) ==="
grep -c . "$LOG/fusion.log" >/dev/null
for f in fusion followup ood; do
    printf "  %-10s %s\n" "$f" "$(tail -1 "$LOG/$f.log")"
done
