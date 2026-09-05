#!/bin/bash
# OOD re-evaluation on Hybrid-v3.3.
set -u
cd "$(dirname "$0")/../.."

V2=datasets/processed_ood/beach_litter_v2
OUT=experiments/work_dirs/beachlitter_v2_eval_v33
CKPT="experiments/work_dirs/htc_x101_v33/epoch_17.pth"
CFG="configs/detectors/htc_x101.py"
STATS=experiments/fusion/stats_v3_3.json
BEST=experiments/work_dirs/_v33_logs/best_epochs.csv
mkdir -p "$OUT"

echo "=== [1/2] five detectors on BeachLitter-v2 ==="
while IFS=, read -r cfg ep score; do
  [ "$cfg" = "config" ] && continue
  ck="experiments/work_dirs/${cfg}_v33/epoch_${ep}.pth"
  [ -f "$ck" ] || { echo "SKIP $cfg - $ck missing"; continue; }
  mkdir -p "$OUT/$cfg"
  echo "[$(date +%H:%M:%S)] OOD $cfg (epoch $ep)"
  CUDA_VISIBLE_DEVICES=${GPU:-0} "${MIM:-.venv/bin/mim}" test mmdet "configs/detectors/${cfg}.py" \
    --checkpoint "$ck" --work-dir "$OUT/$cfg" \
    --cfg-options \
      test_dataloader.dataset.data_root="$V2/" \
      test_dataloader.dataset.ann_file=annotations/instances_test.json \
      test_dataloader.dataset.data_prefix.img=images/test/ \
      test_evaluator.ann_file="$V2/annotations/instances_test.json" \
      test_evaluator.classwise=True \
      test_evaluator.outfile_prefix="$OUT/$cfg/preds" \
    > "$OUT/$cfg/eval.log" 2>&1
  echo "   rc=$?"
done < "$BEST"

echo
echo "=== [2/2] OOD fusion eval (frozen in-domain thresholds) ==="
CACHE=experiments/fusion/roi_cache_beachlitter_v33/
OOD_TUPLES=datasets/multimodal/beachlitter_v2/tuples.csv

# rule: tau swept on in-domain val, applied unchanged to OOD
"${PYTHON:-.venv/bin/python}" scripts/fusion/eval_fraud.py --variants rule \
    --tuples-val datasets/multimodal/v3.3/val/tuples.csv \
    --tuples-test "$OOD_TUPLES" \
    --detector-config "$CFG" \
    --detector-ckpt "$CKPT" --cache-dir "$CACHE" \
    --output-json experiments/fusion/eval_fraud_beachlitter_ood_v33_rule.json 2>&1 | tail -5

# canonical gated, per seed, each at ITS OWN frozen in-domain threshold
for seed in 42 43 44; do
  RUN=$("${PYTHON:-.venv/bin/python}" - "$STATS" "$seed" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); g=d["canonical"]["group"]
print(f"{g}_s{sys.argv[2]}")
PY
)
  TAU=$("${PYTHON:-.venv/bin/python}" - "$STATS" "$RUN" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(d["runs"][sys.argv[2]]["val_threshold"])
PY
)
  DIR="experiments/fusion/_ood33_tmp_${RUN}/gated"
  mkdir -p "$DIR"
  cp "experiments/fusion/v3_3/${RUN}/best.pth" "$DIR/best.pth"
  echo "--- $RUN (frozen tau=$TAU) ---"
  "${PYTHON:-.venv/bin/python}" scripts/fusion/eval_fraud.py --variants gated \
      --tuples-val datasets/multimodal/v3.3/val/tuples.csv \
      --tuples-test "$OOD_TUPLES" \
      --fusion-dir "experiments/fusion/_ood33_tmp_${RUN}/" \
      --detector-config "$CFG" \
      --detector-ckpt "$CKPT" --cache-dir "$CACHE" \
      --learned-threshold "$TAU" \
      --output-json "experiments/fusion/eval_fraud_beachlitter_ood_v33_s${seed}.json" 2>&1 | tail -4
done

echo "OOD_EVALS_V33_DONE"
