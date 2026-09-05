#!/bin/bash
set -u
cd "$(dirname "$0")/../.."

DATA=datasets/processed_v3.3/
LOG_DIR=experiments/work_dirs/_v33_logs
mkdir -p "experiments/work_dirs/_v33_logs"
CONFIGS=(htc_x101 htc_r50 mask_rcnn_r50 cascade_mask_rcnn_r50 faster_rcnn_r50)

# ---- resolve best epochs first (cheap, log-only) ----------------------------
echo "config,best_epoch,val_bbox_mAP" > "$LOG_DIR/best_epochs.csv"
declare -A BEST_EP
for cfg in "${CONFIGS[@]}"; do
    log="$LOG_DIR/$cfg.log"
    [[ -f "$log" ]] || { echo "SKIP $cfg - no training log"; continue; }
    best=$(grep -oE "coco/bbox_mAP: [0-9.]+" "$log" \
           | awk '{print NR, $2}' | sort -k2 -rn | head -1)
    ep=$(echo "$best" | awk '{print $1}')
    score=$(echo "$best" | awk '{print $2}')
    ckpt="experiments/work_dirs/${cfg}_v33/epoch_${ep}.pth"
    [[ -f "$ckpt" ]] || { echo "SKIP $cfg - $ckpt missing"; continue; }
    BEST_EP[$cfg]=$ep
    echo "$cfg,$ep,$score" >> "$LOG_DIR/best_epochs.csv"
    echo "  $cfg -> epoch $ep (val bbox $score)"
done

test_one() {
    local cfg=$1 gpu=$2
    local ep=${BEST_EP[$cfg]:-}
    [[ -n "$ep" ]] || return 0
    echo "[$(date +%H:%M:%S)] TEST $cfg (epoch $ep) on GPU$gpu"
    CUDA_VISIBLE_DEVICES=$gpu "${MIM:-.venv/bin/mim}" test mmdet \
        "configs/detectors/$cfg.py" \
        --checkpoint "experiments/work_dirs/${cfg}_v33/epoch_${ep}.pth" \
        --work-dir "experiments/work_dirs/${cfg}_v33" \
        --cfg-options \
            test_dataloader.dataset.data_root="$DATA" \
            test_evaluator.ann_file="${DATA}annotations/instances_test.json" \
            test_evaluator.classwise=True \
        > "$LOG_DIR/${cfg}_test.log" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] DONE $cfg rc=$rc"
}

track0() { test_one htc_x101 0; test_one mask_rcnn_r50 0; test_one faster_rcnn_r50 0; }
track1() { test_one htc_r50 1; test_one cascade_mask_rcnn_r50 1; }

echo "=== v3.3 test evaluation started $(date) ==="
track0 & P0=$!
track1 & P1=$!
wait $P0 $P1

echo
echo "=== test-split summary (v3.3) ==="
for cfg in "${CONFIGS[@]}"; do
    f="$LOG_DIR/${cfg}_test.log"
    [[ -f "$f" ]] || continue
    b=$(grep -oE "coco/bbox_mAP: [0-9.]+" "$f" | tail -1 | awk '{print $2}')
    s=$(grep -oE "coco/segm_mAP: [0-9.]+" "$f" | tail -1 | awk '{print $2}')
    printf "  %-24s bbox %-8s segm %-8s\n" "$cfg" "${b:---}" "${s:---}"
done
echo "=== ALL_EVAL_V33_FINISHED $(date) ==="
