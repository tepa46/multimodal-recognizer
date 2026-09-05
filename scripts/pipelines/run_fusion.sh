#!/bin/bash
set -u
cd "$(dirname "$0")/../.."

CKPT=experiments/work_dirs/htc_x101_v33/epoch_17.pth   # best val bbox mAP 0.400
CFG=configs/detectors/htc_x101.py
echo "=== fusion v3.3: detector = $CKPT ==="

TRAIN=datasets/multimodal/v3.3/train/tuples.csv
VAL=datasets/multimodal/v3.3/val/tuples.csv
TEST=datasets/multimodal/v3.3/test/tuples.csv
CACHE=experiments/fusion/roi_cache_v33/
OUT=experiments/fusion/v3_3/
STATS=experiments/fusion/stats_v3_3.json
STATS10=experiments/fusion/stats_v3_3_seeds10.json

BASE="--tuples-train $TRAIN --tuples-val $VAL --tuples-test $TEST \
        --detector-config $CFG \
        --detector-ckpt $CKPT --cache-dir $CACHE \
        --output-dir $OUT"
COMMON="$BASE --stats-json $STATS"

echo "[$(date +%H:%M:%S)] A: lambda sweep (mu=0), 5 lambdas x 10 seeds + mlp + rule"
"${PYTHON:-.venv/bin/python}" scripts/fusion/run_fusion_stats.py $BASE \
    --stats-json "$STATS10" \
    --lambdas 0 0.5 1 2 5 --seeds 42 43 44 45 46 47 48 49 50 51 --mu-cls 0.0

echo "[$(date +%H:%M:%S)] B: anchor battery, 5 lambdas x 3 seeds + mlp + rule"
"${PYTHON:-.venv/bin/python}" scripts/fusion/run_fusion_stats.py $COMMON \
    --lambdas 0 0.5 1 2 5 --seeds 42 43 44 --mu-cls 0.0

echo "[$(date +%H:%M:%S)] CE-anchored run (lambda=1, mu=0.1)"
"${PYTHON:-.venv/bin/python}" scripts/fusion/run_fusion_stats.py $COMMON \
    --lambdas 1 --seeds 42 43 44 --mu-cls 0.1 --skip-mlp --append

echo "[$(date +%H:%M:%S)] FUSION_V33_DONE -> $STATS10 + $STATS"
