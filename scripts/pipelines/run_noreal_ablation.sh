#!/bin/bash
# Hybrid-v3.3 WITHOUT real495, then two detectors on it.
set -u
cd "$(dirname "$0")/../.."

DATA=datasets/processed_v3.3_noreal/
LOG_DIR=experiments/work_dirs/_v33_noreal_logs
mkdir -p "$LOG_DIR"

echo "=== [1/2] build $DATA (default sources minus beach_litter; real495 absent) ==="
if [ -f "${DATA}annotations/instances_train.json" ]; then
    echo "  already built, skipping"
else
    "${PYTHON:-.venv/bin/python}" scripts/build_data/build_hybrid_coco.py \
        --raw-dir datasets/raw/ \
        --output-dir "$DATA" \
        --exclude beach_litter \
        --zerowaste-cardboard-frac 0.33 \
        > "$LOG_DIR/build.log" 2>&1
    echo "  build rc=$?"
fi
"${PYTHON:-.venv/bin/python}" - <<'PY'
import json
for sp in ("train", "val", "test"):
    d = json.load(open("datasets/processed_v3.3_noreal/annotations/instances_%s.json" % sp))
    real = sum(1 for im in d["images"] if "real" in im["file_name"].lower())
    print("  %-5s imgs=%-6d anns=%-6d real495-frames=%d" % (sp, len(d["images"]), len(d["annotations"]), real))
PY

train_one() {
    local cfg=$1 gpu=$2
    local work_dir="experiments/work_dirs/${cfg}_v33_noreal"
    echo "[$(date +%H:%M:%S)] START $cfg on GPU$gpu -> $work_dir"
    CUDA_VISIBLE_DEVICES=$gpu bash scripts/detectors/train_mmdet.sh "$cfg" --amp \
        --work-dir "$work_dir" \
        --cfg-options \
            train_dataloader.dataset.data_root="$DATA" \
            val_dataloader.dataset.data_root="$DATA" \
            test_dataloader.dataset.data_root="$DATA" \
            val_evaluator.ann_file="${DATA}annotations/instances_val.json" \
            test_evaluator.ann_file="${DATA}annotations/instances_test.json" \
        > "$LOG_DIR/${cfg}.log" 2>&1
    rc=$?
    echo "[$(date +%H:%M:%S)] DONE  $cfg rc=$rc"
}

echo "=== [2/2] train ==="
train_one htc_x101 0 & P0=$!
train_one htc_r50       1 & P1=$!
wait $P0 $P1
echo "=== V33_NOREAL_TRAINING_FINISHED $(date) ==="
