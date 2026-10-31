#!/bin/bash
set -u
cd "$(dirname "$0")/../.."

PY="${PYTHON:-.venv/bin/python}"
LOG_DIR=experiments/work_dirs/_v33_noreal_logs
mkdir -p "experiments/work_dirs/_v33_noreal_logs"
MASTER=experiments/work_dirs/_v33_noreal_master.log
TAU=0.25

if [ -f "$MASTER" ]; then
    echo "=== waiting for V33_NOREAL_TRAINING_FINISHED ==="
    for i in $(seq 1 2400); do
        grep -q "V33_NOREAL_TRAINING_FINISHED" "$MASTER" 2>/dev/null && break
        sleep 60
    done
    if ! grep -q "V33_NOREAL_TRAINING_FINISHED" "$MASTER" 2>/dev/null; then
        echo "TIMEOUT waiting for training"; exit 1
    fi
    echo "[$(date +%H:%M:%S)] training finished, proceeding"
fi

# Resolve each config's best-validation epoch from its own training log,
# rather than assuming the last epoch is the best one.
best_epoch() {
    local wd=$1
    $PY - "$wd" <<'PY'
import json, sys, glob, os
wd = sys.argv[1]
best, best_ep = -1.0, None
for f in glob.glob(os.path.join(wd, "*", "vis_data", "scalars.json")):
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        m = d.get("coco/bbox_mAP")
        if m is not None and m > best:
            best, best_ep = m, d.get("step") or d.get("epoch")
print(best_ep if best_ep else "")
PY
}

run_one() {
    local cfg=$1 gpu=$2
    local wd="experiments/work_dirs/${cfg}_v33_noreal"
    local ep
    ep=$(best_epoch "$wd")
    if [ -z "$ep" ]; then
        set -- "$wd"/epoch_*.pth
        [ "$#" -eq 1 ] && [ -f "$1" ] && ep=$(basename "$1" .pth | sed 's/^epoch_//')
        set -- "$cfg" "$gpu"
    fi
    if [ -z "$ep" ] || [ ! -f "$wd/epoch_${ep}.pth" ]; then
        echo "  !! cannot resolve best epoch for $cfg (got '$ep'), skipping"; return 1
    fi
    echo "[$(date +%H:%M:%S)] $cfg: best val epoch $ep"

    CUDA_VISIBLE_DEVICES=$gpu $PY scripts/real495/detect.py \
        --config "configs/detectors/${cfg}.py" \
        --ckpt "$wd/epoch_${ep}.pth" \
        --out "datasets/real495/raw_detections_v33_${cfg}.json" \
        --device cuda:0 \
        > "$LOG_DIR/detect_${cfg}.log" 2>&1
    echo "  detect rc=$?"

    $PY scripts/real495/safety_net.py \
        --raw-detections "datasets/real495/raw_detections_v33_${cfg}.json" \
        --tau "$TAU" \
        > "$LOG_DIR/safetynet_${cfg}.log" 2>&1
    echo "  safety-net rc=$?"
    echo "--- $cfg safety-net summary ---"
    tail -40 "$LOG_DIR/safetynet_${cfg}.log"
}

run_one htc_x101 0
run_one htc_r50 1
echo "=== V33_REALNET_DONE $(date) ==="
