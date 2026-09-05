#!/bin/bash
set -u
cd "$(dirname "$0")/../.."

DATA=datasets/processed_v3.3/
LOG_DIR=experiments/work_dirs/_v33_logs
mkdir -p "$LOG_DIR"

train_one() {
    local cfg=$1 gpu=$2
    local work_dir="experiments/work_dirs/${cfg}_v33"
    echo "[$(date +%H:%M:%S)] START $cfg on GPU$gpu -> $work_dir"
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/detectors/train_mmdet.sh "$cfg" --amp         --work-dir "$work_dir"         --cfg-options             train_dataloader.dataset.data_root="$DATA"             val_dataloader.dataset.data_root="$DATA"             test_dataloader.dataset.data_root="$DATA"             val_evaluator.ann_file="${DATA}annotations/instances_val.json"             test_evaluator.ann_file="${DATA}annotations/instances_test.json"         > "$LOG_DIR/${cfg}.log" 2>&1
    local rc=$?
    echo "[$(date +%H:%M:%S)] DONE  $cfg rc=$rc"
    [ $rc -ne 0 ] && echo "[$(date +%H:%M:%S)] FAILED $cfg -- see $LOG_DIR/${cfg}.log"
    return $rc
}

track0() { train_one htc_x101 0; train_one mask_rcnn_r50 0; echo TRACK0_FINISHED; }
track1() { train_one htc_r50 1; train_one cascade_mask_rcnn_r50 1; train_one faster_rcnn_r50 1; echo TRACK1_FINISHED; }

echo "=== v3.3 retraining started $(date) ==="
track0 & P0=$!
track1 & P1=$!
wait $P0 $P1
echo "=== ALL_TRAINING_V33_FINISHED $(date) ==="
