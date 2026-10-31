#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
PY="${PYTHON:-.venv/bin/python}"
FOLDS="${FOLDS:-0 1 2 3}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
LAM="${LAM:-5}"
CACHE=experiments/fusion/roi_cache_v33/
base() { k=$1; D=datasets/multimodal/v3.4_folds/fold$k
  echo "--tuples-train $D/train/tuples.csv --tuples-val $D/val/tuples.csv --cache-dir $CACHE --density-yaml configs/folds/fold$k/calibration_dev.yaml --seeds $SEEDS --mu-cls 0.0 --device cpu --num-workers 0 --threads 1 --input-repr log --no-gate --skip-mlp --lambdas $LAM"; }
for k in $FOLDS; do
  D=datasets/multimodal/v3.4_folds/fold$k; CFG=configs/folds/fold$k/calibration_dev.yaml
  O=experiments/fusion/v3_4_ablation/fold$k; mkdir -p $O
  for mod in visual depth mass; do
    nohup "$PY" scripts/fusion/run_fusion_stats.py $(base $k) --tuples-test $D/test/tuples.csv --mask-modality $mod \
      --output-dir $O/$mod/ --stats-json $O/stats_$mod.json > $O/log_$mod.log 2>&1 &
  done
  for sw in 0.02 0.05 0.10; do for sd in 0.05 0.10 0.20; do
    G=datasets/multimodal/v3.4_noise/fold$k/sw${sw}_sd${sd}
    for p in train:42 val:43; do sp=${p%%:*}; sdd=${p##*:}
      [ -f $G/$sp/tuples.csv ] || "$PY" scripts/fusion/generate_multimodal_tuples.py --coco-json datasets/processed_v3.3/annotations/instances_$sp.json --image-root datasets/processed_v3.3/images/$sp/ --output-dir $G/$sp --seed $sdd --skip-depth --calibration $CFG --weight-noise-sigma $sw --volume-noise $sd >/dev/null
    done
    [ -f $G/test/tuples.csv ] || "$PY" scripts/fusion/generate_multimodal_tuples.py --coco-json datasets/processed_v3.3/annotations/instances_test.json --image-root datasets/processed_v3.3/images/test/ --output-dir $G/test --seed 44 --skip-depth --calibration configs/folds/fold$k/calibration_test.yaml --table-calibration $CFG --weight-noise-sigma $sw --volume-noise $sd >/dev/null
    O=experiments/fusion/v3_4_noise/fold$k; mkdir -p $O
    nohup "$PY" scripts/fusion/run_fusion_stats.py --tuples-train $G/train/tuples.csv --tuples-val $G/val/tuples.csv --tuples-test $G/test/tuples.csv --cache-dir $CACHE --density-yaml $CFG --seeds $SEEDS --mu-cls 0.0 --device cpu --num-workers 0 --threads 1 --input-repr log --no-gate --skip-mlp --lambdas $LAM \
      --output-dir $O/sw${sw}_sd${sd}/ --stats-json $O/stats_sw${sw}_sd${sd}.json > $O/log_sw${sw}_sd${sd}.log 2>&1 &
  done; done
done
echo "launched: $(pgrep -fc 'scripts/fusion/run_fusion_stats.py')"
