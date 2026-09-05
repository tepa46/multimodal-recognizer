#!/usr/bin/env bash
set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
CKPT="experiments/work_dirs/htc_x101_v33/epoch_17.pth"

mkdir -p experiments/fusion/sweep
SUMMARY="experiments/fusion/sweep/summary.tsv"
echo -e "sigma_w\tsigma_d\tval_F1\ttest_F1\ttest_FPR\twrong_class_rec" > "$SUMMARY"

for sw in 0.02 0.05 0.10; do
  for sd in 0.05 0.10 0.20; do
    OUT="datasets/multimodal/v3.3_sweep/sw${sw}_sd${sd}"
    FUSION_DIR="experiments/fusion/sweep/sw${sw}_sd${sd}"
    echo ""
    echo "=== sigma_w=${sw}, sigma_d=${sd} ==="

    # 1. Regenerate 3 splits (skip-depth — fusion only needs CSV + ROI cache)
    for split in train val test; do
      case "$split" in train) seed=42;; val) seed=43;; test) seed=44;; esac
      "$PY" scripts/fusion/generate_multimodal_tuples.py \
          --coco-json "datasets/processed_v3.3/annotations/instances_${split}.json" \
          --image-root "datasets/processed_v3.3/images/${split}/" \
          --output-dir "${OUT}/${split}/" \
          --weight-noise-sigma "$sw" --depth-noise-sigma "$sd" \
          --fraud-ratio 0.25 --seed "$seed" --skip-depth \
          2>&1 | tail -2
    done

    # 2. Train canonical gated λ=2.0, 30 ep
    "$PY" scripts/fusion/train_fusion.py \
        --variant gated --epochs 30 --lambda-phys 2.0 \
        --tuples-train "${OUT}/train/tuples.csv" \
        --tuples-val "${OUT}/val/tuples.csv" \
        --detector-ckpt "$CKPT" \
        --seed 42 \
        --output-dir "${FUSION_DIR}/" 2>&1 | tail -5

    # 3. Eval on test split (gated only — no rule/mlp at off-grid points)
    "$PY" scripts/fusion/eval_fraud.py \
        --variants gated \
        --tuples-val "${OUT}/val/tuples.csv" \
        --tuples-test "${OUT}/test/tuples.csv" \
        --fusion-dir "${FUSION_DIR}/" \
        --detector-ckpt "$CKPT" \
        --bbox-map 0.542 --mask-map 0.528 \
        --output-json "${FUSION_DIR}/eval.json" 2>&1 | tail -10

    # 4. Append to summary TSV
    "$PY" -c "
import json
with open('${FUSION_DIR}/eval.json') as f:
    d = json.load(f)
g = d['results']['gated']
ov = g['test']['overall']
wc = g['test'].get('wrong_class', {}).get('rec', float('nan'))
print(f\"${sw}\t${sd}\t{g['ckpt_best_f1']:.4f}\t{ov['f1']:.4f}\t{ov['fpr']:.4f}\t{wc:.4f}\")
" | tee -a "$SUMMARY"
  done
done

echo ""
echo "=== Sensor-noise sweep complete ==="
column -t "$SUMMARY"
