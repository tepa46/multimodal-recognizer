#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
PY="${PYTHON:-.venv/bin/python}"
D=datasets/multimodal/v3.4_folds/fold0; CFG=configs/folds/fold0/calibration_dev.yaml
OUT=experiments/fusion/v3_4_hparam; mkdir -p "$OUT"
COMMON="--tuples-train $D/train/tuples.csv --tuples-val $D/val/tuples.csv --tuples-test $D/test/tuples.csv \
        --cache-dir experiments/fusion/roi_cache_v33/ --density-yaml $CFG --seeds 42 43 44 45 46 --mu-cls 0.0 \
        --device cpu --num-workers 0 --threads 1 --input-repr log --lambdas 0 --skip-mlp"
for lr in 3e-4 1e-3 3e-3; do for ep in 30 60; do for dr in 0.1 0.3 0.5; do for wd in 1e-4 1e-3; do
  for head in gate nogate; do
    flag=""; [ $head = nogate ] && flag="--no-gate"
    tag="${head}_lr${lr}_ep${ep}_dr${dr}_wd${wd}"
    nohup "$PY" scripts/fusion/run_fusion_stats.py $COMMON $flag --lr $lr --epochs $ep --dropout $dr --weight-decay $wd \
      --output-dir "$OUT/$tag/" --stats-json "$OUT/stats_$tag.json" > "$OUT/log_$tag.log" 2>&1 &
  done
done; done; done; done
echo "launched $(pgrep -fc 'scripts/fusion/run_fusion_stats.py')"
