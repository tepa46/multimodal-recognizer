# Physics-aware multimodal fusion for fraud detection in reverse-vending machines

Code, results and datasets for the paper *"Physics-aware multimodal fusion for
fraud detection in automated reverse-vending machines"*.

## What is here

| |                                                                               |
|---|-------------------------------------------------------------------------------|
| `scripts/` | the full pipeline, from raw source download to the paper's figures            |
| `src/models/multimodal_fusion.py` | the fusion head (rule / MLP / gated variants)                                 |
| `configs/` | class taxonomy, effective-density table, detector configs                     |
| `results/` | every artefact behind the paper's tables and figures                          |
| `datasets/` | see `datasets/README.md` |

## Install

```bash
uv sync
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install openmim && mim install mmengine "mmcv>=2.1" "mmdet>=3.3"
uv pip install git+https://github.com/facebookresearch/segment-anything.git
```

Tested with Python 3.12, PyTorch 2.5.1+cu121, MMDetection 3.3, MMCV ≥ 2.1,
SAM ViT-H. Experiments ran on 2× NVIDIA RTX 4070 Ti SUPER (16 GB each).

## Reproducing

Run every script from the repository root. Shell runners invoke `.venv/bin/python`
by default; override with `PYTHON=/path/to/python bash scripts/...`.

```bash
# 1. sources -> Hybrid-v3.3
python scripts/build_data/download_sources.py
python scripts/build_data/build_hybrid_coco.py \
    --raw-dir datasets/raw/ --output-dir datasets/processed_v3.3/ \
    --exclude beach_litter --zerowaste-cardboard-frac 0.33

# 2. detectors — train (~29 GPU-hours for all five), or download the
#    checkpoints (models/README.md) and run link_checkpoints.sh instead
bash scripts/detectors/train_all.sh
bash scripts/detectors/eval_all.sh

# 3. multimodal tuples — one call per split, seeds 42/43/44
for split in train:42 val:43 test:44; do
  s=${split%%:*}; seed=${split##*:}
  python scripts/fusion/generate_multimodal_tuples.py \
      --coco-json datasets/processed_v3.3/annotations/instances_${s}.json \
      --image-root datasets/processed_v3.3/images/${s}/ \
      --output-dir datasets/multimodal/v3.3/${s}/ \
      --density-config configs/material_density.yaml \
      --fraud-ratio 0.25 --seed $seed
done

# 3b. OOD set and its tuples (reserved seed 45), needed by run_ood_evals.sh
python scripts/beachlitter/build_beachlitter10.py
python scripts/fusion/generate_multimodal_tuples.py \
    --coco-json datasets/processed_ood/beach_litter_v2/annotations/instances_test_verified.json \
    --image-root datasets/processed_ood/beach_litter_v2/images/test/ \
    --output-dir datasets/multimodal/beachlitter_v2/ \
    --density-config configs/material_density.yaml --fraud-ratio 0.25 --seed 45

# 4. everything downstream of the detectors (< 1 GPU-hour, uses a cached ROI tensor)
bash scripts/pipelines/run_all.sh
```

`run_all.sh` chains the fusion sweep, the follow-up studies (modality ablation,
density misspecification, sensor-noise grid) and the OOD evaluations. The two
experiments that need their own detector are separate:

```bash
bash scripts/pipelines/run_noreal_ablation.sh   # retrain without real495 frames
bash scripts/pipelines/run_safety_net.sh        # physics on real detector confusions
```

## Licences

Code is MIT (`LICENSE`). The datasets are not uniformly licensed and the release
reflects that — see `DATA_LICENSES.md` before redistributing anything.

## Citation

See `CITATION.cff`.
