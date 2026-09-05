#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

PY="${PYTHON:-.venv/bin/python}"
CKPT="experiments/work_dirs/htc_x101_v33/epoch_17.pth"
CFG="configs/detectors/htc_x101.py"
CACHE="experiments/fusion/roi_cache_v33/"
STATS="experiments/fusion/stats_v3_3.json"

CANON_GROUP=$("$PY" -c "import json;print(json.load(open('$STATS'))['canonical']['group'])")
CANON="experiments/fusion/v3_3/${CANON_GROUP}_s42/best.pth"
LAM=$("$PY" -c "print('${CANON_GROUP}'.split('lam')[1].split('_')[0])")
MU=$("$PY" -c "print('${CANON_GROUP}'.split('mu')[1])")

TR=datasets/multimodal/v3.3/train/tuples.csv
VA=datasets/multimodal/v3.3/val/tuples.csv
TE=datasets/multimodal/v3.3/test/tuples.csv

echo "=== anchor: $CANON (lambda=$LAM mu=$MU), detector epoch 17 (v3.3) ==="

echo "=== [1/3] density perturbation (B5) ==="
"$PY" scripts/fusion/eval_density_perturbation.py \
    --gated-ckpt "$CANON" --detector-ckpt "$CKPT" --cache-dir "$CACHE" \
    --tuples-val "$VA" --tuples-test "$TE" \
    --sigma-rho 0.2 --n-draws 10 \
    --output-json experiments/fusion/density_perturbation_v33.json 2>&1 | tail -8

echo "=== [2/3] modality ablation ==="
ABL=experiments/fusion/v3_3_ablation
mkdir -p "$ABL/none/gated"
cp "$CANON" "$ABL/none/gated/best.pth"
"$PY" scripts/fusion/eval_fraud.py --variants gated \
    --fusion-dir "$ABL/none/" --detector-ckpt "$CKPT" --cache-dir "$CACHE" \
    --tuples-val "$VA" --tuples-test "$TE" \
    --output-json "$ABL/none/eval.json" 2>&1 | tail -4

for mod in visual depth mass; do
  echo "--- remove $mod ---"
  "$PY" scripts/fusion/train_fusion.py --variant gated --epochs 30 \
      --lambda-phys "$LAM" --mu-cls "$MU" --seed 42 --mask-modality "$mod" \
      --detector-ckpt "$CKPT" --cache-dir "$CACHE" \
      --tuples-train "$TR" --tuples-val "$VA" \
      --output-dir "$ABL/${mod}/" 2>&1 | tail -2
  "$PY" scripts/fusion/eval_fraud.py --variants gated \
      --fusion-dir "$ABL/${mod}/" --detector-ckpt "$CKPT" --cache-dir "$CACHE" \
      --tuples-val "$VA" --tuples-test "$TE" --mask-modality "$mod" \
      --output-json "$ABL/${mod}/eval.json" 2>&1 | tail -4
done

echo "=== [3/3] sensor-noise sweep 3x3 ==="
SWEEP_DATA=datasets/multimodal/v3.3_sweep
SWEEP_OUT=experiments/fusion/sweep_v33
mkdir -p "$SWEEP_OUT"
SUMMARY="$SWEEP_OUT/summary.tsv"
printf "sigma_w\tsigma_d\tval_F1\ttest_F1\ttest_FPR\twrong_class_rec\n" > "$SUMMARY"

for sw in 0.02 0.05 0.10; do
  for sd in 0.05 0.10 0.20; do
    OUT="$SWEEP_DATA/sw${sw}_sd${sd}"
    FD="$SWEEP_OUT/sw${sw}_sd${sd}"
    echo "--- sigma_w=${sw} sigma_d=${sd} ---"
    for split in train val test; do
      case "$split" in train) seed=42;; val) seed=43;; test) seed=44;; esac
      if [ ! -f "${OUT}/${split}/tuples.csv" ]; then
        "$PY" scripts/fusion/generate_multimodal_tuples.py \
            --coco-json "datasets/processed_v3.3/annotations/instances_${split}.json" \
            --image-root "datasets/processed_v3.3/images/${split}/" \
            --output-dir "${OUT}/${split}/" \
            --weight-noise-sigma "$sw" --depth-noise-sigma "$sd" \
            --fraud-ratio 0.25 --seed "$seed" --skip-depth 2>&1 | tail -1
      fi
    done
    "$PY" scripts/fusion/train_fusion.py --variant gated --epochs 30 \
        --lambda-phys "$LAM" --mu-cls "$MU" --seed 42 \
        --tuples-train "${OUT}/train/tuples.csv" \
        --tuples-val "${OUT}/val/tuples.csv" \
        --detector-ckpt "$CKPT" --cache-dir "$CACHE" \
        --output-dir "${FD}/" 2>&1 | tail -2
    "$PY" scripts/fusion/eval_fraud.py --variants gated \
        --tuples-val "${OUT}/val/tuples.csv" \
        --tuples-test "${OUT}/test/tuples.csv" \
        --fusion-dir "${FD}/" --detector-ckpt "$CKPT" --cache-dir "$CACHE" \
        --output-json "${FD}/eval.json" 2>&1 | tail -3
    "$PY" -c "
import json
d=json.load(open('${FD}/eval.json'))
g=d['results']['gated']; ov=g['test']['overall']
wc=g['test'].get('wrong_class',{}).get('rec',float('nan'))
print(f\"${sw}\t${sd}\t{g['ckpt_best_f1']:.6f}\t{ov['f1']:.6f}\t{ov['fpr']:.6f}\t{wc:.6f}\")
" | tee -a "$SUMMARY"
  done
done

echo "FOLLOWUP_V33_DONE"
column -t "$SUMMARY"
