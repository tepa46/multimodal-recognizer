#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
PY="${PYTHON:-.venv/bin/python}"
FOLDS="${FOLDS:-0 1 2 3}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
CACHE="${CACHE:-experiments/fusion/roi_cache_v33/}"
OUT="${OUT:-experiments/fusion/v3_4_rawctl}"
for k in $FOLDS; do
  D=datasets/multimodal/v3.4_folds/fold$k
  CFG=configs/folds/fold$k/calibration_dev.yaml
  COMMON="--tuples-train $D/train/tuples.csv --tuples-val $D/val/tuples.csv --tuples-test $D/test/tuples.csv \
          --cache-dir $CACHE --density-yaml $CFG --seeds $SEEDS --mu-cls 0.0 --lambdas 0 \
          --device cpu --num-workers 0 --threads 1 --input-repr raw"
  run() { tag=$1; shift
    nohup "$PY" scripts/fusion/run_fusion_stats.py $COMMON \
        --output-dir "$OUT/fold$k/$tag/" --stats-json "$OUT/fold$k/stats_$tag.json" "$@" \
        > "$OUT/fold$k/log_$tag.log" 2>&1 & }
  mkdir -p "$OUT/fold$k"
  run proj_raw --no-gate
  run gate_raw --skip-mlp
done
echo "launched: $(pgrep -fc 'scripts/fusion/run_fusion_stats.py') processes"
