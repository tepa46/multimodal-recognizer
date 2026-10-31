#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
PY="${PYTHON:-.venv/bin/python}"
FOLDS="${FOLDS:-0}"
SEEDS="${SEEDS:-42 43 44 45 46 47 48 49 50 51}"
CACHE="${CACHE:-experiments/fusion/roi_cache_v33/}"
OUT="${OUT:-experiments/fusion/v3_4_sens}"
THREADS="${THREADS:-4}"
SIGMAS="${SIGMAS:-0 0.1 0.2 0.3 0.5}"
gen() {
  local out=$1 k=$2; shift 2
  for p in train:42 val:43; do sp=${p%%:*}; sd=${p##*:}
    [ -f "$out/$sp/tuples.csv" ] || "$PY" scripts/fusion/generate_multimodal_tuples.py \
      --coco-json datasets/processed_v3.3/annotations/instances_$sp.json --image-root datasets/processed_v3.3/images/$sp/ \
      --output-dir "$out/$sp" --seed $sd --skip-depth --calibration configs/folds/fold$k/calibration_dev.yaml "$@" >/dev/null
  done
  [ -f "$out/test/tuples.csv" ] || "$PY" scripts/fusion/generate_multimodal_tuples.py \
      --coco-json datasets/processed_v3.3/annotations/instances_test.json --image-root datasets/processed_v3.3/images/test/ \
      --output-dir "$out/test" --seed 44 --skip-depth --calibration configs/folds/fold$k/calibration_test.yaml \
      --table-calibration configs/folds/fold$k/calibration_dev.yaml "$@" >/dev/null
}
run() {
  local d=$1 k=$2 tag=$3
  mkdir -p "$OUT/fold$k"
  OMP_NUM_THREADS=$THREADS nohup "$PY" scripts/fusion/run_fusion_stats.py \
    --tuples-train "$d/train/tuples.csv" --tuples-val "$d/val/tuples.csv" --tuples-test "$d/test/tuples.csv" \
    --cache-dir "$CACHE" --density-yaml configs/folds/fold$k/calibration_dev.yaml \
    --seeds $SEEDS --lambdas 0 --mu-cls 0.0 --device cpu --num-workers 0 --input-repr log --no-gate --skip-mlp \
    --output-dir "$OUT/fold$k/$tag/" --stats-json "$OUT/fold$k/stats_$tag.json" > "$OUT/fold$k/log_$tag.log" 2>&1 &
}
for k in $FOLDS; do
  gen datasets/multimodal/v3.4_sens/fold$k/mult_moderate    $k --heavy-fill-range 1.5 3.0  --foreign-object-range 1.3 2.0
  gen datasets/multimodal/v3.4_sens/fold$k/mult_overlap     $k --heavy-fill-range 1.2 2.5  --foreign-object-range 1.1 1.8
  gen datasets/multimodal/v3.4_sens/fold$k/mult_strong      $k --heavy-fill-range 1.05 1.5 --foreign-object-range 1.05 1.3
  gen datasets/multimodal/v3.4_sens/fold$k/mult_submitted   $k --heavy-fill-range 2.5 6.0  --foreign-object-range 1.8 3.5
  for t in mult_moderate mult_overlap mult_strong mult_submitted; do run datasets/multimodal/v3.4_sens/fold$k/$t $k $t; done
  for s in $SIGMAS; do
    gen datasets/multimodal/v3.4_sens/fold$k/spread$s $k --density-spread $s
    run datasets/multimodal/v3.4_sens/fold$k/spread$s $k spread$s
  done
done
echo "launched: $(pgrep -fc 'scripts/fusion/run_fusion_stats.py') processes"
