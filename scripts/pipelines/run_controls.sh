#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
PY="${PYTHON:-.venv/bin/python}"
SEEDS="42 43 44 45 46 47 48 49 50 51"
CACHE=experiments/fusion/roi_cache_v33/
D=datasets/multimodal/v3.3; O=experiments/fusion/v3_3_gate_pinned; mkdir -p $O
C="--tuples-train $D/train/tuples.csv --tuples-val $D/val/tuples.csv --tuples-test $D/test/tuples.csv --cache-dir $CACHE --density-yaml configs/material_density.yaml --seeds $SEEDS --mu-cls 0.0 --device cpu --num-workers 0 --threads 1 --input-repr raw --lambdas 0 --narrow-rule-grid"
nohup "$PY" scripts/fusion/run_fusion_stats.py $C --output-dir $O/gate/ --stats-json $O/stats_gate.json > $O/log_gate.log 2>&1 &
nohup "$PY" scripts/fusion/run_fusion_stats.py $C --no-gate --skip-mlp --output-dir $O/nogate/ --stats-json $O/stats_nogate.json > $O/log_nogate.log 2>&1 &
D=datasets/multimodal/v3.4_sens/fold0/spread0; O=experiments/fusion/v3_4_gate_sigma0; mkdir -p $O
C="--tuples-train $D/train/tuples.csv --tuples-val $D/val/tuples.csv --tuples-test $D/test/tuples.csv --cache-dir $CACHE --density-yaml configs/folds/fold0/calibration_dev.yaml --seeds $SEEDS --mu-cls 0.0 --device cpu --num-workers 0 --threads 1 --input-repr log --lambdas 0 --skip-mlp"
nohup "$PY" scripts/fusion/run_fusion_stats.py $C --output-dir $O/gate/ --stats-json $O/stats_gate.json > $O/log_gate.log 2>&1 &
nohup "$PY" scripts/fusion/run_fusion_stats.py $C --no-gate --output-dir $O/nogate/ --stats-json $O/stats_nogate.json > $O/log_nogate.log 2>&1 &
echo "launched: $(pgrep -fc 'scripts/fusion/run_fusion_stats.py')"
