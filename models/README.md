# Models

Trained weights produced by this work. Third-party pretrained weights are *not*
here — see **Pretrained weights we do not redistribute** below.

## Fusion heads — in this repository

The heads of the specimen-disjoint protocol at the default λ = 5 (projection
head without gates, log inputs), one per fold, partition and seed:

```
fusion/v3_4_folds{,_rep1,_rep2,_rep3}/fold{0..3}/proj/gated_lam5_mu0_s{42..51}/best.pth
fusion/v3_4_noreal_heads/fold{0..3}/proj/gated_lam5_mu0_s{42..51}/best.pth
```

`v3_4_noreal_heads` are trained on the ROI features of `htc_x101_v33_noreal`
without the real495 frames; they are the heads scored on the measured real495
tuples. Each `fold*/stats_proj.json` holds the validation-selected thresholds
that the frozen evaluations apply:

```bash
python scripts/fusion/eval_frozen.py --runs-root models/fusion/v3_4_folds \
    --target-tuples ... --target-cache ... --out ...
```

The flat directories (`fusion/gated_lam*_mu*_s*`, `fusion/mlp_s*`) are the heads
of the earlier v3.3 protocol.

The heads consume cached ROI features from a frozen detector; they are not
standalone image models.

## Detector checkpoints — hosted externally

Five benchmark detectors at their best-validation epoch, plus the two retrained
without real495 frames for the safety-net experiment and the measured-mass evaluation.

| checkpoint | val bbox mAP | size |
|---|---:|---:|
| `htc_x101_v33/epoch_17.pth` | 0.400 | 1.1 GB |
| `htc_r50_v33/epoch_19.pth` | 0.385 | 613 MB |
| `cascade_mask_rcnn_r50_v33/epoch_18.pth` | 0.368 | 610 MB |
| `mask_rcnn_r50_v33/epoch_20.pth` | 0.357 | 350 MB |
| `faster_rcnn_r50_v33/epoch_17.pth` | 0.355 | 327 MB |
| `htc_x101_v33_noreal/epoch_17.pth` | — | 1.1 GB |
| `htc_r50_v33_noreal/epoch_19.pth` | — | 613 MB |


**Download:** https://drive.google.com/drive/folders/1ZUoKk8UKXZszlQFgY0vroZ0sFAV-yAwG

Unpack the archive into `models/detectors/`. That directory already holds
`CHECKSUMS.md5`, which is tracked in this repository — so the expected hashes
come from git, independently of the download, and the extracted weights land
right next to them:

```bash
tar -xzf detectors_v33.tar.gz -C models/
cd models/detectors && md5sum -c CHECKSUMS.md5
```

Every line must print `OK`.

Unpack into `models/detectors/` preserving the directory names above, then:

```bash
bash scripts/detectors/link_checkpoints.sh
```

That symlinks each directory into `experiments/work_dirs/`, which is where
training would have written it and where the pipelines look. After this the
pipelines run unchanged whether the weights were trained or downloaded. The
script leaves any real directory alone, so it is safe to run in a tree that
already has locally trained checkpoints.

Epoch numbers are the argmax of validation bbox mAP, recorded in
`results/detectors/best_epochs.csv`.

## Pretrained weights we do not redistribute

Both are third-party and freely downloadable; the configs expect them under
`checkpoints/`.

**COCO-pretrained detectors** (~1.5 GB), referenced by each config's `load_from`:

```bash
mim download mmdet --config htc_x101_64x4d_fpn_16x1_20e_coco --dest checkpoints/
mim download mmdet --config htc_r50_fpn_1x_coco               --dest checkpoints/
mim download mmdet --config mask_rcnn_r50_fpn_1x_coco         --dest checkpoints/
mim download mmdet --config faster_rcnn_r50_fpn_1x_coco       --dest checkpoints/
mim download mmdet --config cascade_mask_rcnn_r50_fpn_1x_coco --dest checkpoints/
```

**SAM ViT-H** (2.4 GB), used for mask generation on Drinking Waste, real495 and
BeachLitter:

```bash
curl -L -o checkpoints/sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```
