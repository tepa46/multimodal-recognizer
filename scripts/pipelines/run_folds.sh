#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
PY="${PYTHON:-.venv/bin/python}"
PART="${PART:-}"
FOLDS="${FOLDS:-0 1 2 3}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
CACHE="${CACHE:-experiments/fusion/roi_cache_v33/}"
OUT="${OUT:-experiments/fusion/v3_4_folds$PART}"
for k in $FOLDS; do
  D=datasets/multimodal/v3.4_folds$PART/fold$k
  CFG=configs/folds$PART/fold$k/calibration_dev.yaml
  COMMON="--tuples-train $D/train/tuples.csv --tuples-val $D/val/tuples.csv --tuples-test $D/test/tuples.csv \
          --cache-dir $CACHE --density-yaml $CFG --seeds $SEEDS --mu-cls 0.0 \
          --device cpu --num-workers 0 --threads 1 --input-repr log"
  run() { tag=$1; shift
    nohup "$PY" scripts/fusion/run_fusion_stats.py $COMMON \
        --output-dir "$OUT/fold$k/$tag/" --stats-json "$OUT/fold$k/stats_$tag.json" "$@" \
        > "$OUT/fold$k/log_$tag.log" 2>&1 & }
  mkdir -p "$OUT/fold$k"
  run proj       --no-gate --lambdas 0 0.5 1 2 5
  run proj_nohat --no-gate --lambdas 0 --no-expected-mass --skip-mlp
  run mlp_nohat  --lambdas 0 --no-expected-mass --mlp-only
  run gate       --lambdas 0 --skip-mlp
  run gate_nohat --lambdas 0 --no-expected-mass --skip-mlp
done
echo "launched: $(pgrep -fc 'scripts/fusion/run_fusion_stats.py') processes"
