#!/bin/bash
set -u
cd "$(dirname "$0")/../.."
PY="${PYTHON:-.venv/bin/python}"
O=experiments/fusion/v3_4_ood; mkdir -p $O
for k in 0 1 2 3; do
  "$PY" scripts/fusion/generate_multimodal_tuples.py --coco-json datasets/processed_ood/beach_litter_v2/annotations/instances_test_verified.json \
    --image-root datasets/processed_ood/beach_litter_v2/images/test/ --output-dir datasets/multimodal/v3.4_beach/fold$k --seed 45 --skip-depth \
    --calibration configs/folds/fold$k/calibration_test.yaml --table-calibration configs/folds/fold$k/calibration_dev.yaml > /dev/null
done
BCACHE=experiments/fusion/roi_cache_beachlitter_v33/epoch_17_test.pt
[ -f $BCACHE ] || "$PY" scripts/fusion/build_roi_cache.py datasets/multimodal/v3.4_beach/fold0/tuples.csv $BCACHE \
    experiments/work_dirs/htc_x101_v33/epoch_17.pth
"$PY" scripts/real495/to_tuples.py --measurements datasets/real495/measurements_specimens.csv --out datasets/multimodal/real495/tuples.csv
"$PY" scripts/fusion/eval_frozen.py --target-tuples datasets/multimodal/v3.4_beach/fold0/tuples.csv --per-fold-target "datasets/multimodal/v3.4_beach/fold{k}/tuples.csv" \
    --target-cache $BCACHE --out $O/beachlitter.json
