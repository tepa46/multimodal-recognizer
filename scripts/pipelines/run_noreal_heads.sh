#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
PY="${PYTHON:-.venv/bin/python}"
FOLDS="${FOLDS:-0 1 2 3}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
CKPT=experiments/work_dirs/htc_x101_v33_noreal/epoch_17.pth
CACHE=experiments/fusion/roi_cache_v33_noreal/
OUT=experiments/fusion/v3_4_noreal_heads
DATA=datasets/multimodal/v3.4_folds_noreal495
STAGE="${STAGE:-train}"
if [ "$STAGE" = train ]; then
  for sp in train val test; do
    [ -f $CACHE/epoch_17_$sp.pt ] || "$PY" scripts/fusion/build_roi_cache.py \
        datasets/multimodal/v3.3/$sp/tuples.csv $CACHE/epoch_17_$sp.pt
  done
  for k in $FOLDS; do
    mkdir -p $DATA/fold$k/train $DATA/fold$k/val $DATA/fold$k/test $OUT/fold$k
    grep -v "/real300_" datasets/multimodal/v3.4_folds/fold$k/train/tuples.csv > $DATA/fold$k/train/tuples.csv
    cp datasets/multimodal/v3.4_folds/fold$k/val/tuples.csv  $DATA/fold$k/val/tuples.csv
    cp datasets/multimodal/v3.4_folds/fold$k/test/tuples.csv $DATA/fold$k/test/tuples.csv
    nohup "$PY" scripts/fusion/run_fusion_stats.py \
        --tuples-train $DATA/fold$k/train/tuples.csv --tuples-val $DATA/fold$k/val/tuples.csv \
        --tuples-test $DATA/fold$k/test/tuples.csv --cache-dir $CACHE --detector-ckpt $CKPT \
        --density-yaml configs/folds/fold$k/calibration_dev.yaml --seeds $SEEDS --mu-cls 0.0 \
        --device cpu --num-workers 0 --threads 1 --input-repr log --no-gate --skip-mlp --lambdas 5 \
        --output-dir $OUT/fold$k/proj/ --stats-json $OUT/fold$k/stats_proj.json \
        > $OUT/fold$k/log_proj.log 2>&1 &
  done
  echo "launched: $(pgrep -fc 'scripts/fusion/run_fusion_stats.py') processes"
else
  R495=experiments/fusion/roi_cache_real495_noreal/epoch_17_real495.pt
  [ -f $R495 ] || "$PY" scripts/fusion/build_roi_cache.py datasets/multimodal/real495/tuples.csv $R495
  "$PY" scripts/fusion/eval_frozen.py --runs-root $OUT --target-tuples datasets/multimodal/real495/tuples.csv \
      --target-cache $R495 \
      --out experiments/fusion/v3_4_ood/real495.json
  "$PY" scripts/analysis/real495_head_rocs.py --runs-root $OUT --out experiments/fusion/v3_4_ood/real495_rocs.json
  "$PY" scripts/analysis/dump_real495_scores.py --runs-root $OUT
fi
