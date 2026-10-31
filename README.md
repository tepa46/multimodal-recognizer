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
    --sources taco zerowaste drinking_waste dwsd real495 \
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

## Specimen-disjoint protocol

All numbers of the paper come from this protocol. Its inputs are in the
repository: the 89 calibration specimens (`datasets/calibration/`), the table
fitted on them (`configs/calibration_packphys.yaml`), four random partitions of
the specimens into four folds (`configs/folds{,_rep1,_rep2,_rep3}/`, partition
seeds 2026, 1, 2, 3) and the fold tuples (`datasets/multimodal/v3.4_folds*/`,
seeds 42/43/44 for train/val/test). `datasets/calibration/README.md` shows how
the table and the folds were built.

Every head is trained for ten seeds (42–51) on the CPU with one thread
(`--threads 1`, set by all runners). Under this setting a run reproduces bit
for bit; with more threads the same seed can move F1 by up to 0.02.

The runners start their jobs in the background; wait for them to finish before
the next step. The heads read ROI features of `htc_x101_v33/epoch_17`, cached in
`experiments/fusion/roi_cache_v33/` on the first run.

```bash
for p in "" _rep1 _rep2 _rep3; do PART=$p bash scripts/pipelines/run_folds.sh; done
bash scripts/pipelines/run_controls.sh
bash scripts/pipelines/run_hparam_gate.sh
bash scripts/pipelines/run_sensitivity.sh
bash scripts/pipelines/run_followup_folds.sh
bash scripts/pipelines/run_raw_control.sh
bash scripts/pipelines/run_frozen_evals.sh
STAGE=train bash scripts/pipelines/run_noreal_heads.sh
STAGE=eval  bash scripts/pipelines/run_noreal_heads.sh
python scripts/analysis/raw_control_gates.py
python scripts/real495/eval_measured.py
bash scripts/pipelines/run_safety_net.sh
for d in htc_x101 htc_r50; do
  python scripts/real495/safety_net_revised.py --label $d \
      --raw-detections datasets/real495/raw_detections_v33_$d.json \
      --out results/real495/safety_net_revised_$d.json
done
python scripts/analysis/collect_revision.py
python scripts/analysis/make_revision_tables.py
```

`collect_revision.py` writes every number of the paper to
`results/revision/numbers.json`; `make_revision_tables.py` turns it into the
LaTeX tables and number macros in `results/revision/tables/`.

The statistics of all these runs are stored in `results/revision/fusion/`, so
the numbers and tables can be regenerated without training anything:

```bash
python scripts/analysis/collect_revision.py results/revision/fusion
python scripts/analysis/make_revision_tables.py
```

The trained heads at the default λ = 5 are in `models/fusion/` (see
`models/README.md`).

## Licences

Code is MIT (`LICENSE`). The datasets are not uniformly licensed and the release
reflects that — see `DATA_LICENSES.md` before redistributing anything.

## Citation

See `CITATION.cff`.
