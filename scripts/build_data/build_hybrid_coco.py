from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

CocoJson = dict[str, Any]
ClassMap = dict[str, dict[str, str]]  # source → {original_name: unified_name}


# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------

def load_class_mapping(path: Path) -> tuple[list[str], ClassMap]:
    with open(path) as f:
        data = yaml.safe_load(f)
    unified: list[str] = data["unified_classes"]
    mapping: ClassMap = {
        key: val for key, val in data.items() if key != "unified_classes"
    }
    return unified, mapping


def remap_category_id(
    original_name: str,
    source: str,
    class_map: ClassMap,
    unified_classes: list[str],
    fallback: str = "other",
) -> int:
    src_map = class_map.get(source, {})
    lc_src_map = {k.lower(): v for k, v in src_map.items()}
    unified_name = lc_src_map.get(original_name.lower(), fallback)
    if unified_name not in unified_classes:
        unified_name = fallback
    return unified_classes.index(unified_name)


# ---------------------------------------------------------------------------
# COCO parsers
# ---------------------------------------------------------------------------

def load_coco_json(json_path: Path) -> CocoJson:
    with open(json_path) as f:
        return json.load(f)


def parse_coco_source(
    source_name: str,
    coco: CocoJson,
    images_dir: Path,
    class_map: ClassMap,
    unified_classes: list[str],
    global_image_id: int,
    global_ann_id: int,
) -> tuple[list[dict], list[dict], int, int]:
    """
    Extract images + annotations from a COCO JSON, remapping category IDs.
    Returns (images, annotations, next_global_image_id, next_global_ann_id).
    """
    # Build local category_id → original_name lookup
    cat_id_to_name: dict[int, str] = {
        cat["id"]: cat["name"] for cat in coco.get("categories", [])
    }

    # Build local image_id → image record lookup
    local_id_to_image: dict[int, dict] = {img["id"]: img for img in coco.get("images", [])}

    # local image_id → global image_id
    local_to_global: dict[int, int] = {}

    out_images: list[dict] = []
    for img in coco.get("images", []):
        local_to_global[img["id"]] = global_image_id
        out_images.append({
            "id": global_image_id,
            "file_name": img["file_name"],
            "width": img.get("width", 0),
            "height": img.get("height", 0),
            "source": source_name,
            "original_id": img["id"],
        })
        global_image_id += 1

    out_anns: list[dict] = []
    for ann in coco.get("annotations", []):
        original_cat_name = cat_id_to_name.get(ann["category_id"], "")
        new_cat_id = remap_category_id(
            original_cat_name, source_name, class_map, unified_classes
        )
        global_img_id = local_to_global.get(ann["image_id"])
        if global_img_id is None:
            continue  # orphaned annotation

        out_ann: dict = {
            "id": global_ann_id,
            "image_id": global_img_id,
            "category_id": new_cat_id,
            "bbox": ann.get("bbox", []),
            "area": ann.get("area", 0),
            "iscrowd": ann.get("iscrowd", 0),
        }
        if "segmentation" in ann:
            out_ann["segmentation"] = ann["segmentation"]
        out_anns.append(out_ann)
        global_ann_id += 1

    return out_images, out_anns, global_image_id, global_ann_id


# ---------------------------------------------------------------------------
# YOLO bbox parser (for Drinking Waste)
# ---------------------------------------------------------------------------

def parse_yolo_drinking_waste(
    dataset_dir: Path,
    class_map: ClassMap,
    unified_classes: list[str],
    global_image_id: int,
    global_ann_id: int,
) -> tuple[list[dict], list[dict], int, int]:
    class_dirs = ["Aluminium", "Glass", "HDPE", "PET"]
    out_images: list[dict] = []
    out_anns: list[dict] = []

    for class_name in class_dirs:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            print(f"    [warn] Drinking Waste: directory not found: {class_dir}")
            continue

        new_cat_id = remap_category_id(class_name, "drinking_waste", class_map, unified_classes)

        for img_path in sorted(class_dir.glob("*.jpg")):
            from PIL import Image as PILImage
            with PILImage.open(img_path) as im:
                w, h = im.size

            file_name = f"drinking_waste/{class_name}/{img_path.name}"
            out_images.append({
                "id": global_image_id,
                "file_name": file_name,
                "width": w,
                "height": h,
                "source": "drinking_waste",
                "original_id": global_image_id,
            })
            rect_poly = [0.0, 0.0, float(w), 0.0, float(w), float(h), 0.0, float(h)]
            out_anns.append({
                "id": global_ann_id,
                "image_id": global_image_id,
                "category_id": new_cat_id,
                "bbox": [0, 0, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [rect_poly],
            })
            global_image_id += 1
            global_ann_id += 1

    return out_images, out_anns, global_image_id, global_ann_id


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def _fold_rare(labels: list[int], min_count: int = 2, sentinel: int = -2) -> list[int]:
    counts = Counter(labels)
    out = [l if counts[l] >= min_count else sentinel for l in labels]
    counts = Counter(out)
    short = {l for l, c in counts.items() if c < min_count}
    if short:
        biggest, biggest_n = counts.most_common(1)[0]
        if biggest_n >= min_count:
            out = [biggest if l in short else l for l in out]
    return out


def stratified_split(
    images: list[dict],
    annotations: list[dict],
    val_ratio: float,
    test_ratio: float,
) -> tuple[list[int], list[int], list[int]]:
    from sklearn.model_selection import train_test_split

    # Map image_id → dominant category_id
    img_cat_counts: dict[int, Counter] = defaultdict(Counter)
    for ann in annotations:
        img_cat_counts[ann["image_id"]][ann["category_id"]] += 1

    img_ids = [img["id"] for img in images]
    # Fallback label = -1 for images with no annotations
    labels = [
        img_cat_counts[iid].most_common(1)[0][0] if iid in img_cat_counts else -1
        for iid in img_ids
    ]

    # First split off test set
    val_test_ratio = val_ratio + test_ratio
    train_ids, valtest_ids, _, valtest_labels = train_test_split(
        img_ids, labels,
        test_size=val_test_ratio,
        stratify=_fold_rare(labels),
        random_state=42,
    )

    # Split val/test from the remainder
    relative_test = test_ratio / val_test_ratio
    val_ids, test_ids = train_test_split(
        valtest_ids,
        test_size=relative_test,
        stratify=_fold_rare(valtest_labels),
        random_state=42,
    )

    return train_ids, val_ids, test_ids


# ---------------------------------------------------------------------------
# Image copy / resize
# ---------------------------------------------------------------------------

def copy_or_resize_image(
    src: Path,
    dst: Path,
    image_size: int | None,
) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if image_size is None:
        shutil.copy2(src, dst)
        return

    from PIL import Image as PILImage
    with PILImage.open(src) as im:
        im = im.convert("RGB")
        # Letterbox: keep aspect ratio, pad with black
        im.thumbnail((image_size, image_size), PILImage.LANCZOS)
        padded = PILImage.new("RGB", (image_size, image_size), (0, 0, 0))
        offset_x = (image_size - im.width) // 2
        offset_y = (image_size - im.height) // 2
        padded.paste(im, (offset_x, offset_y))
        padded.save(dst, quality=95)


# ---------------------------------------------------------------------------
# Save split
# ---------------------------------------------------------------------------

def save_split(
    split_name: str,
    img_id_set: set[int],
    all_images: list[dict],
    all_annotations: list[dict],
    unified_classes: list[str],
    raw_dir: Path,
    out_dir: Path,
    image_size: int | None,
    source_image_dirs: dict[str, Path],
) -> dict[str, int]:
    """Write images + COCO JSON for one split. Returns class distribution."""
    split_images = [img for img in all_images if img["id"] in img_id_set]
    split_anns = [ann for ann in all_annotations if ann["image_id"] in img_id_set]

    images_out_dir = out_dir / "images" / split_name
    images_out_dir.mkdir(parents=True, exist_ok=True)

    # Copy/resize images
    missing = 0
    for img in split_images:
        src = _resolve_image_path(img, raw_dir, source_image_dirs)
        if src is None or not src.exists():
            missing += 1
            continue
        dst_name = f"{img['source']}_{img['id']:08d}{src.suffix}"
        img["file_name"] = dst_name
        dst = images_out_dir / dst_name
        copy_or_resize_image(src, dst, image_size)

    if missing:
        print(f"    [warn] {missing} image files not found for split '{split_name}'")

    # Build COCO categories
    categories = [
        {"id": i, "name": name, "supercategory": "waste"}
        for i, name in enumerate(unified_classes)
    ]

    coco_out: CocoJson = {
        "info": {
            "description": f"Hybrid Waste COCO — {split_name} split",
            "version": "1.0",
            "year": 2025,
        },
        "categories": categories,
        "images": split_images,
        "annotations": split_anns,
    }

    ann_path = out_dir / "annotations" / f"instances_{split_name}.json"
    ann_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ann_path, "w") as f:
        json.dump(coco_out, f)
    print(f"    Saved {ann_path.name}: {len(split_images)} images, {len(split_anns)} annotations")

    # Class distribution
    dist: dict[str, int] = Counter()
    for ann in split_anns:
        dist[unified_classes[ann["category_id"]]] += 1
    return dist


def _resolve_image_path(
    img: dict,
    raw_dir: Path,
    source_image_dirs: dict[str, Path],
) -> Path | None:
    source = img.get("source", "")
    file_name = img.get("file_name", "")
    base_dir = source_image_dirs.get(source, raw_dir / source)
    # Full relative path first (TACO stores file_name as "batch_1/000006.jpg")
    candidate = base_dir / file_name
    if candidate.exists():
        return candidate
    # Basename fallback (for flat layouts)
    candidate = base_dir / Path(file_name).name
    if candidate.exists():
        return candidate
    # Last resort: resolve relative to raw_dir
    direct = raw_dir / file_name
    if direct.exists():
        return direct
    return None


# ---------------------------------------------------------------------------
# Source discovery helpers
# ---------------------------------------------------------------------------

def find_coco_json(dataset_dir: Path, hints: list[str]) -> Path | None:
    for hint in hints:
        p = dataset_dir / hint
        if p.exists():
            return p
    # Recursive search
    for p in sorted(dataset_dir.rglob("*.json")):
        try:
            with open(p) as f:
                d = json.load(f)
            if "annotations" in d and "images" in d:
                return p
        except Exception:
            continue
    return None


def find_images_dir(dataset_dir: Path) -> Path:
    for candidate in ["images", "train", "val", "."]:
        p = dataset_dir / candidate
        if p.is_dir():
            imgs = (
                list(p.glob("*.jpg")) + list(p.glob("*.JPG"))
                + list(p.glob("*.jpeg")) + list(p.glob("*.JPEG"))
                + list(p.glob("*.png")) + list(p.glob("*.PNG"))
            )
            if imgs:
                return p
    return dataset_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def subsample_zerowaste_cardboard(
    images: list[dict],
    annotations: list[dict],
    unified_classes: list[str],
    cardboard_frac: float,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    if cardboard_frac >= 1.0:
        return images, annotations

    import random
    rng = random.Random(seed)
    try:
        carton_id = unified_classes.index("carton")
    except ValueError:
        return images, annotations

    zw_image_ids = {img["id"] for img in images if img.get("source") == "zerowaste"}

    zw_carton_anns = [
        a for a in annotations
        if a["image_id"] in zw_image_ids and a["category_id"] == carton_id
    ]
    other_anns = [
        a for a in annotations
        if not (a["image_id"] in zw_image_ids and a["category_id"] == carton_id)
    ]

    rng.shuffle(zw_carton_anns)
    keep_n = int(len(zw_carton_anns) * cardboard_frac)
    kept_carton = zw_carton_anns[:keep_n]

    new_anns = other_anns + kept_carton

    # Drop ZeroWaste images that lost ALL their annotations.
    referenced_ids = {a["image_id"] for a in new_anns}
    new_images = [
        img for img in images
        if (img["id"] in referenced_ids) or (img.get("source") != "zerowaste")
    ]

    dropped_anns = len(annotations) - len(new_anns)
    dropped_imgs = len(images) - len(new_images)
    print(
        f"[zerowaste subsample] cardboard: {len(zw_carton_anns)} → {keep_n} "
        f"(dropped {dropped_anns} ann, {dropped_imgs} img)"
    )
    return new_images, new_anns


def build(
    raw_dir: Path,
    output_dir: Path,
    class_mapping_path: Path,
    val_ratio: float,
    test_ratio: float,
    image_size: int | None,
    sources: list[str],
    exclude: list[str] | None = None,
    zerowaste_cardboard_frac: float = 1.0,
) -> None:
    unified_classes, class_map = load_class_mapping(class_mapping_path)
    print(f"Unified classes ({len(unified_classes)}): {unified_classes}\n")

    if exclude:
        sources = [s for s in sources if s not in exclude]
        print(f"Excluded sources: {exclude}")
    print(f"Active sources: {sources}\n")

    all_images: list[dict] = []
    all_annotations: list[dict] = []
    source_image_dirs: dict[str, Path] = {}
    global_image_id = 1
    global_ann_id = 1

    # --- TACO ---
    if "taco" in sources:
        taco_dir = raw_dir / "taco"
        json_path = find_coco_json(taco_dir, [
            "data/annotations.json", "annotations.json",
        ])
        if json_path:
            print(f"[taco] Loading from {json_path}")
            coco = load_coco_json(json_path)
            imgs, anns, global_image_id, global_ann_id = parse_coco_source(
                "taco", coco, json_path.parent, class_map, unified_classes,
                global_image_id, global_ann_id,
            )
            all_images.extend(imgs)
            all_annotations.extend(anns)
            source_image_dirs["taco"] = json_path.parent
            print(f"  → {len(imgs)} images, {len(anns)} annotations")
        else:
            print("[taco] COCO JSON not found — skipping. Run download_datasets.py first.")

    if "zerowaste" in sources:
        zw_dir = raw_dir / "zerowaste"
        zw_splits_root = zw_dir / "splits_final_deblurred"
        if zw_splits_root.is_dir():
            for zw_split in ("train", "val", "test"):
                json_path = zw_splits_root / zw_split / "labels.json"
                if not json_path.exists():
                    print(f"[zerowaste/{zw_split}] labels.json not found — skipping")
                    continue
                print(f"[zerowaste/{zw_split}] Loading from {json_path}")
                coco = load_coco_json(json_path)
                rel_prefix = f"splits_final_deblurred/{zw_split}/data"
                for img in coco.get("images", []):
                    img["file_name"] = f"{rel_prefix}/{Path(img['file_name']).name}"
                imgs, anns, global_image_id, global_ann_id = parse_coco_source(
                    "zerowaste", coco, zw_dir, class_map, unified_classes,
                    global_image_id, global_ann_id,
                )
                for img in imgs:
                    img["zerowaste_orig_split"] = zw_split
                all_images.extend(imgs)
                all_annotations.extend(anns)
                print(f"  → {len(imgs)} images, {len(anns)} annotations")
            source_image_dirs["zerowaste"] = zw_dir
        else:
            json_path = find_coco_json(zw_dir, [
                "zerowaste-f/train_annotations.json",
                "train_annotations.json", "annotations.json",
            ])
            if json_path:
                print(f"[zerowaste] Loading legacy merged JSON: {json_path}")
                coco = load_coco_json(json_path)
                imgs, anns, global_image_id, global_ann_id = parse_coco_source(
                    "zerowaste", coco, json_path.parent, class_map, unified_classes,
                    global_image_id, global_ann_id,
                )
                all_images.extend(imgs)
                all_annotations.extend(anns)
                source_image_dirs["zerowaste"] = find_images_dir(json_path.parent)
                print(f"  → {len(imgs)} images, {len(anns)} annotations")
            else:
                print("[zerowaste] COCO JSON not found — skipping.")

    # --- Drinking Waste ---
    if "drinking_waste" in sources:
        dw_dir = raw_dir / "drinking_waste"
        # Check if SAM-labelled JSON already exists
        sam_json = dw_dir / "annotations" / "instances_sam.json"
        if sam_json.exists():
            print(f"[drinking_waste] Loading SAM-labelled JSON: {sam_json}")
            coco = load_coco_json(sam_json)
            imgs, anns, global_image_id, global_ann_id = parse_coco_source(
                "drinking_waste", coco, dw_dir, class_map, unified_classes,
                global_image_id, global_ann_id,
            )
        elif dw_dir.exists() and any(dw_dir.iterdir()):
            print("[drinking_waste] No SAM JSON found — using whole-image bbox placeholders.")
            print("  Run sam_autolabel.py later to replace with real masks.")
            imgs, anns, global_image_id, global_ann_id = parse_yolo_drinking_waste(
                dw_dir, class_map, unified_classes, global_image_id, global_ann_id,
            )
        else:
            print("[drinking_waste] Directory not found — skipping.")
            imgs, anns = [], []
        all_images.extend(imgs)
        all_annotations.extend(anns)
        source_image_dirs["drinking_waste"] = dw_dir

        if imgs:
            print(f"  → {len(imgs)} images, {len(anns)} annotations")

    # --- DWSD ---
    if "dwsd" in sources:
        dwsd_dir = raw_dir / "dwsd"
        json_path = find_coco_json(dwsd_dir, [
            "annotations.json", "instances.json",
        ])
        if json_path:
            print(f"[dwsd] Loading from {json_path}")
            coco = load_coco_json(json_path)
            imgs, anns, global_image_id, global_ann_id = parse_coco_source(
                "dwsd", coco, json_path.parent, class_map, unified_classes,
                global_image_id, global_ann_id,
            )
            all_images.extend(imgs)
            all_annotations.extend(anns)
            source_image_dirs["dwsd"] = find_images_dir(json_path.parent)
            print(f"  → {len(imgs)} images, {len(anns)} annotations")
        else:
            print("[dwsd] COCO JSON not found — skipping.")

    # --- BeachLitter ---
    if "beach_litter" in sources:
        bl_dir = raw_dir / "beach_litter"
        json_path = find_coco_json(bl_dir, [
            "annotations/instances.json", "instances.json",
        ])
        if json_path:
            print(f"[beach_litter] Loading from {json_path}")
            coco = load_coco_json(json_path)
            imgs, anns, global_image_id, global_ann_id = parse_coco_source(
                "beach_litter", coco, json_path.parent, class_map, unified_classes,
                global_image_id, global_ann_id,
            )
            all_images.extend(imgs)
            all_annotations.extend(anns)
            # beach_litter file_name paths are stored as full paths relative
            # to bl_dir (e.g. beachlitter_dataset_2022/.../000003.jpg), so
            # point source_image_dirs at bl_dir rather than the annotations subdir.
            source_image_dirs["beach_litter"] = bl_dir
            print(f"  → {len(imgs)} images, {len(anns)} annotations")
        else:
            print("[beach_litter] COCO JSON not found — skipping.")

    # --- real495 ---
    if "real495" in sources:
        r300_root = Path("datasets/real495")
        json_path = r300_root / "annotations" / "instances_all.json"
        split_path = r300_root / "split_train_heldout.json"
        if json_path.exists() and split_path.exists():
            print(f"[real495] Loading from {json_path}")
            split = json.loads(split_path.read_text())
            keep = set(split["train"])
            coco = load_coco_json(json_path)
            kept_ids = {im["id"] for im in coco["images"]
                        if im["file_name"] in keep}
            coco = {
                **coco,
                "images": [im for im in coco["images"] if im["id"] in kept_ids],
                "annotations": [a for a in coco["annotations"]
                                if a["image_id"] in kept_ids],
            }
            imgs, anns, global_image_id, global_ann_id = parse_coco_source(
                "real495", coco, r300_root / "images", class_map, unified_classes,
                global_image_id, global_ann_id,
            )
            all_images.extend(imgs)
            all_annotations.extend(anns)
            source_image_dirs["real495"] = r300_root / "images"
            print(f"  → {len(imgs)} images, {len(anns)} annotations "
                  f"(of {len(split['train']) + len(split['heldout'])} total; "
                  f"{len(split['heldout'])} held-out frames withheld)")
        else:
            print("[real495] instances_all.json or split_train_heldout.json "
                  "not found — skipping. Run the real495 pipeline first.")

    print(f"\nTotal before subsample: {len(all_images)} images, {len(all_annotations)} annotations")

    # --- Optional ZeroWaste cardboard sub-sampling ---
    if zerowaste_cardboard_frac < 1.0:
        all_images, all_annotations = subsample_zerowaste_cardboard(
            all_images, all_annotations, unified_classes,
            cardboard_frac=zerowaste_cardboard_frac,
        )
        print(
            f"Total after subsample:  {len(all_images)} images, "
            f"{len(all_annotations)} annotations\n"
        )
    else:
        print()

    if not all_images:
        print("No images loaded. Download datasets first: python scripts/build_data/download_sources.py")
        return

    # --- Split ---
    print("Splitting into train/val/test...")
    real495_forced_train = [img["id"] for img in all_images
                            if img.get("source") == "real495"]
    splittable_images = [img for img in all_images
                         if img.get("source") != "real495"]
    splittable_ids = {img["id"] for img in splittable_images}
    splittable_anns = [a for a in all_annotations
                       if a["image_id"] in splittable_ids]

    train_ids, val_ids, test_ids = stratified_split(
        splittable_images, splittable_anns, val_ratio, test_ratio,
    )
    if real495_forced_train:
        train_ids = list(train_ids) + real495_forced_train
        print(f"  real495: {len(real495_forced_train)} images forced into train "
              f"(held-out frames excluded from the corpus entirely)")
    print(f"  train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}\n")

    # --- Save ---
    distributions: dict[str, dict[str, int]] = {}
    for split_name, id_list in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        distributions[split_name] = save_split(
            split_name=split_name,
            img_id_set=set(id_list),
            all_images=all_images,
            all_annotations=all_annotations,
            unified_classes=unified_classes,
            raw_dir=raw_dir,
            out_dir=output_dir,
            image_size=image_size,
            source_image_dirs=source_image_dirs,
        )

    # --- Distribution table ---
    print("\nClass distribution:")
    header = f"{'Class':<22}" + "".join(f"{'train':>10}{'val':>8}{'test':>8}")
    print(header)
    print("-" * len(header))
    for cls in unified_classes:
        row = f"{cls:<22}"
        for split in ("train", "val", "test"):
            count = distributions[split].get(cls, 0)
            row += f"{count:>10}" if split == "train" else f"{count:>8}"
        print(row)

    print(f"\nDone. Output: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a unified hybrid COCO dataset from raw waste datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--raw-dir", default="datasets/raw/", metavar="DIR",
                        help="Root directory with raw datasets (default: datasets/raw/)")
    parser.add_argument("--output-dir", default="datasets/processed/", metavar="DIR",
                        help="Output directory (default: datasets/processed/)")
    parser.add_argument("--class-mapping", default="configs/class_mapping.yaml", metavar="YAML",
                        help="Path to class_mapping.yaml (default: configs/class_mapping.yaml)")
    parser.add_argument("--val-ratio", type=float, default=0.15, metavar="FLOAT",
                        help="Fraction of images for validation (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, metavar="FLOAT",
                        help="Fraction of images for testing (default: 0.15)")
    parser.add_argument("--image-size", type=int, default=None, metavar="INT",
                        help="Resize images to NxN px with letterbox padding. "
                             "Disabled by default (images kept at original size).")
    parser.add_argument(
        "--sources", nargs="+",
        default=["taco", "zerowaste", "drinking_waste", "dwsd", "beach_litter"],
        metavar="NAME",
        help="Which sources to include (default: the five public ones). "
             "Add 'real495' to fold in the project's own real-sensor train "
             "frames (held-out frames are withheld automatically).",
    )
    parser.add_argument(
        "--exclude", nargs="+", default=[], metavar="NAME",
        help="Sources to exclude from --sources (e.g. 'beach_litter' for hybrid-v3).",
    )
    parser.add_argument(
        "--zerowaste-cardboard-frac", type=float, default=1.0, metavar="FLOAT",
        help="Keep this fraction of ZeroWaste cardboard annotations (default: 1.0). "
             "Use 0.33 for hybrid-v3 to rebalance the dominant cardboard class.",
    )
    args = parser.parse_args()

    build(
        raw_dir=Path(args.raw_dir),
        output_dir=Path(args.output_dir),
        class_mapping_path=Path(args.class_mapping),
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        image_size=args.image_size,
        sources=args.sources,
        exclude=args.exclude,
        zerowaste_cardboard_frac=args.zerowaste_cardboard_frac,
    )


if __name__ == "__main__":
    main()
