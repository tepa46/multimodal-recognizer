# Data card

This card covers the data released with the paper: **real495**, the
**Hybrid-v3.3** benchmark, the **calibration specimens** and the **BeachLitter
packaging labels**. Licences per source are in `../DATA_LICENSES.md`.

## real495

Photographs of household packaging taken by the authors, with exhaustive
annotation and a physical measurement of every object.

| | |
|---|---|
| Images | 215 JPEG, long side 3024 px, EXIF orientation applied to the pixels, no people in any frame |
| Annotations | 495 objects, 1–6 per frame (median 2), all ten classes |
| Masks | 493 SAM ViT-H masks prompted by the hand-drawn boxes, 2 box polygons where SAM failed the quality check |
| Measurements | every object: height, width, depth (cm), mass (g), empty mass (g), condition, filling material |
| Physical specimens | 25, each with an id (`measurements_specimens.csv`); one specimen contributes up to 41 objects |
| Annotators | two (118 and 97 frames); per-frame provenance in `review_state.json` and in the `annotator` field |
| Licence | CC BY 4.0 |

Objects per class:

| class | objects | specimens |
|---|---:|---:|
| other | 123 | 5 |
| plastic_bag | 70 | 4 |
| plastic_container | 49 | 2 |
| plastic_bottle | 49 | 3 |
| metal_can | 47 | 2 |
| carton | 41 | 3 |
| glass_jar | 41 | 1 |
| tetrapack | 27 | 1 |
| glass_bottle | 25 | 1 |
| paper | 23 | 3 |

`other` in real495 consists of non-packaging items left in the frame
(measuring tools, a power bank, a cloth); this differs from `other` in the
public sources, which is mixed litter.

Condition of the objects:

| condition | objects | filling material |
|---|---:|---|
| normal | 365 | — |
| heavy_fill | 123 | water 70, ballast 34, stone 16, not recorded 3 |
| foreign_object | 7 | ballast 6, not recorded 1 |

All seven `foreign_object` rows are one physical specimen. Dimensions are
measured in the state in which the object was photographed (a crumpled item is
measured crumpled).

**Measurement table.** One row is one object on one frame:

```
filename, pos, object_id, class, height_cm, width_cm, depth_cm,
mass_g, mass_empty_g, condition, fill_content, notes[, specimen_id]
```

`pos` is the left-to-right rank of the box centre in the frame and joins a
row to its COCO annotation; `instances_all.json` carries the same value in
`attributes.pos`, together with `condition`, `origin` (detector proposal or
manual box), `det_class`, `det_score` and the SigLIP-2 class suggestion.

**Split.** `split_train_heldout.json` fixes 128 training and 87 held-out
frames, split by frame. A split by specimen is impossible: the 25 specimens
co-occur on shared frames and form one connected group. All 25 specimens of
the held-out frames also appear in training frames, so held-out detector
metrics measure recognition of known objects in new arrangements. The
leakage figures stored inside the split file (32 objects, 93%) come from an
earlier object signature and are superseded by the specimen ids.

**Overlap with Hybrid-v3.3.** The 128 training frames are part of the
Hybrid-v3.3 training split, where their file names start with `real300_` and
their `source` field reads `real300`, an earlier name of the same set. The
evaluations on real495 in the paper use a detector retrained without these
frames.

## Hybrid-v3.3

A ten-class instance-segmentation benchmark of 10,213 images and 28,249
annotations built from four public datasets and the real495 training frames.

| source | images | annotations | masks |
|---|---:|---:|---|
| TACO | 1,500 | 4,784 | upstream polygons |
| ZeroWaste-f | 4,208 | 14,872 | upstream polygons |
| Drinking Waste | 3,679 | 3,679 | SAM, prompted by the upstream boxes |
| DWSD | 698 | 4,626 | upstream polygons |
| real495 (train frames) | 128 | 288 | SAM, prompted by hand-drawn boxes |
| **total** | **10,213** | **28,249** | |

Split: train 7,187 / val 1,513 / test 1,513 images (19,713 / 4,272 / 4,264
annotations), 70/15/15 stratified by the most frequent class of each image
(`random_state=42`); real495 frames go to train only.

Annotations per class:

| class | train | val | test |
|---|---:|---:|---:|
| plastic_bottle | 2,141 | 457 | 483 |
| glass_bottle | 509 | 111 | 106 |
| glass_jar | 42 | 2 | 11 |
| metal_can | 1,298 | 281 | 304 |
| carton | 4,311 | 935 | 918 |
| tetrapack | 81 | 23 | 33 |
| paper | 402 | 79 | 84 |
| plastic_bag | 5,413 | 1,215 | 1,196 |
| plastic_container | 1,813 | 402 | 374 |
| other | 3,703 | 767 | 755 |

**Construction.** Every source category is mapped to one of the ten classes
by `configs/class_mapping.yaml`; categories it does not list become `other`.
The build keeps every upstream image and annotation except:

- ZeroWaste-f cardboard, sub-sampled to 33% of its annotations (17,751 →
  5,857) so that it does not dominate `carton`; 295 images left without any
  annotation are dropped;
- the 87 held-out real495 frames (207 annotations);
- BeachLitter, held out entirely.

No other filtering is applied. `processed_v3.3/manifest.json` lists, per
source, the upstream files, the image and annotation counts, the mapping of
every source category with its annotation count, and each exclusion; it is
produced by `scripts/build_data/make_manifest.py`, which re-runs the build
logic on the upstream data and confirms that it reproduces the released
annotation files image by image.

**Known limitations.** `glass_jar` (11 test annotations) and `tetrapack`
(33) are too thin for per-class conclusions. `carton` comes mostly from
cluttered conveyor images of ZeroWaste-f. `other` pools whatever the other
classes do not cover. The sources differ strongly in scene, from studio
close-ups (Drinking Waste) to distant street litter (TACO).

**Images.** Only real495 images ship here. The others are rebuilt from the
upstream downloads (`../README.md`); the released annotation files fix the
exact split.

## Calibration specimens

89 further measured specimens, none of them in real495 or in any Hybrid-v3.3
image: dimensions and empty mass for all, filled mass for 61. They give the
density table, its between-specimen spread, the fill ratios and the volume
calibration. Details and the fold partitions: `calibration/README.md`.
Licence: CC BY 4.0.

## BeachLitter packaging labels

Packaging labels for the connected components of the `artificial_litter`
layer of BeachLitter v2022, in the same ten classes: 1,411 verified packaging
objects, 2,160 verified non-packaging objects and 14,282 `iscrowd` ignore
zones over 2,120 images, with SAM masks
(`processed_ood/beach_litter_v2/annotations/`, per-object decisions in
`../results/ood/beachlitter_relabel/verified_labels.csv`). Only the labels are
ours (CC BY 4.0); the imagery stays under its upstream terms.
