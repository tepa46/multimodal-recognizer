from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Unified category for all BeachLitter instances
# ---------------------------------------------------------------------------
BEACH_LITTER_CATEGORY = {
    "id": 9,
    "name": "other",
    "supercategory": "waste",
}

ALL_CATEGORIES = [
    {"id": 0, "name": "plastic_bottle",    "supercategory": "waste"},
    {"id": 1, "name": "glass_bottle",      "supercategory": "waste"},
    {"id": 2, "name": "glass_jar",         "supercategory": "waste"},
    {"id": 3, "name": "metal_can",         "supercategory": "waste"},
    {"id": 4, "name": "carton",            "supercategory": "waste"},
    {"id": 5, "name": "tetrapack",         "supercategory": "waste"},
    {"id": 6, "name": "paper",             "supercategory": "waste"},
    {"id": 7, "name": "plastic_bag",       "supercategory": "waste"},
    {"id": 8, "name": "plastic_container", "supercategory": "waste"},
    {"id": 9, "name": "other",             "supercategory": "waste"},
]


# ---------------------------------------------------------------------------
# Directory structure discovery
# ---------------------------------------------------------------------------

def _find_pairs(root: Path) -> list[tuple[Path, Path]]:
    layout_candidates = [
        ("images", "maskpngs"),      # BeachLitter v2022 actual layout
        ("images", "masks"),
        ("images", "labels"),
        ("img", "mask"),
        ("img", "masks"),
        ("rgb", "segmentation"),
        ("JPEGImages", "SegmentationClass"),
    ]

    for img_sub, mask_sub in layout_candidates:
        img_dir = root / img_sub
        mask_dir = root / mask_sub
        if img_dir.is_dir() and mask_dir.is_dir():
            pairs = _match_pairs(img_dir, mask_dir)
            if pairs:
                print(f"  Detected layout: {img_sub}/ + {mask_sub}/ → {len(pairs)} pairs")
                return pairs

    # Fallback: walk recursively, match by stem
    print("  Standard layout not found — scanning recursively…")
    img_paths = {p.stem: p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg")}
    mask_paths = {p.stem: p for p in root.rglob("*.png")}

    pairs = []
    for stem, img in sorted(img_paths.items()):
        # Exclude PNG images that are in image folders
        if stem in mask_paths and "mask" in str(mask_paths[stem]).lower():
            pairs.append((img, mask_paths[stem]))
        elif stem in mask_paths:
            pairs.append((img, mask_paths[stem]))

    if pairs:
        print(f"  Found {len(pairs)} image/mask pairs via recursive scan.")
    return pairs


def _match_pairs(img_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """Match images to masks by stem, tolerating different extensions."""
    mask_by_stem: dict[str, Path] = {}
    for p in mask_dir.iterdir():
        if p.suffix.lower() in (".png", ".bmp", ".tif", ".tiff"):
            mask_by_stem[p.stem] = p

    pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if img_path.stem in mask_by_stem:
            pairs.append((img_path, mask_by_stem[img_path.stem]))
    return pairs


# ---------------------------------------------------------------------------
# Inspection mode
# ---------------------------------------------------------------------------

def inspect(root: Path, n_sample: int = 50) -> None:
    """Print pixel value distribution from a sample of mask files."""
    mask_files = list(root.rglob("*.png"))
    # Filter out obvious image files (large, coloured)
    mask_files = [p for p in mask_files if "mask" in str(p).lower() or "label" in str(p).lower()]
    if not mask_files:
        mask_files = list(root.rglob("*.png"))

    sample = mask_files[:n_sample]
    print(f"Inspecting {len(sample)} mask files from {root}\n")

    global_counts: Counter = Counter()
    for p in tqdm(sample, desc="  Scanning"):
        try:
            im = PILImage.open(p)
            arr = np.array(im) if im.mode == "P" else np.array(im.convert("L"))
            global_counts.update(arr.flatten().tolist())
        except Exception as exc:
            print(f"  [warn] {p.name}: {exc}")

    total = sum(global_counts.values())
    print("\nPixel value distribution:")
    print(f"  {'Value':>6}  {'Count':>12}  {'%':>7}")
    for val, cnt in sorted(global_counts.items()):
        pct = cnt / total * 100
        print(f"  {val:>6}  {cnt:>12,}  {pct:>6.2f}%")

    print(
        "\nBased on the BeachLitter paper, value 1 = artificial_litter (all waste).\n"
        "Run without --inspect to convert, or use --litter-class-id to override."
    )


# ---------------------------------------------------------------------------
# Mask → COCO instances
# ---------------------------------------------------------------------------

def mask_to_polygons(binary_mask: np.ndarray) -> list[list[float]]:
    """Convert binary mask to COCO polygon list using connected components."""
    try:
        import cv2
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        polygons = []
        for c in contours:
            if len(c) >= 3:
                flat = c.flatten().tolist()
                if len(flat) >= 6:
                    polygons.append(flat)
        return polygons
    except ImportError:
        pass

    try:
        from skimage import measure
        contours = measure.find_contours(binary_mask.astype(float), 0.5)
        polygons = []
        for c in contours:
            if len(c) >= 4:
                flat = []
                for pt in c:
                    flat.extend([float(pt[1]), float(pt[0])])  # (col, row) = (x, y)
                polygons.append(flat)
        return polygons
    except ImportError:
        pass

    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return []
    r_idx = np.where(rows)[0]
    c_idx = np.where(cols)[0]
    r0, r1 = int(r_idx[0]), int(r_idx[-1])
    c0, c1 = int(c_idx[0]), int(c_idx[-1])
    return [[c0, r0, c1, r0, c1, r1, c0, r1]]


def extract_instances(
    mask_arr: np.ndarray,
    litter_class_id: int,
    min_area: int,
) -> list[dict[str, Any]]:
    """
    Extract COCO-style annotation dicts (without id/image_id) from a mask array.
    Uses connected-component labelling to separate touching blobs.
    """
    binary = (mask_arr == litter_class_id).astype(np.uint8)
    if not binary.any():
        return []

    # Connected components
    try:
        import cv2
        n_labels, labels = cv2.connectedComponents(binary)
        label_range = range(1, n_labels)  # 0 = background
    except ImportError:
        try:
            from scipy import ndimage
            labels, n_labels = ndimage.label(binary)
            label_range = range(1, n_labels + 1)
        except ImportError:
            # No cv2/scipy: treat whole litter region as single instance
            warnings.warn(
                "Neither cv2 nor scipy found — treating all litter as one instance. "
                "Install one for proper connected-component splitting.",
                stacklevel=2,
            )
            labels = binary
            label_range = [1]

    instances = []
    for label_id in label_range:
        component = (labels == label_id).astype(np.uint8)
        area = int(component.sum())
        if area < min_area:
            continue

        polygons = mask_to_polygons(component)
        if not polygons:
            continue

        # Tight bbox [x, y, w, h]
        rows = np.where(np.any(component, axis=1))[0]
        cols = np.where(np.any(component, axis=0))[0]
        x0, x1 = int(cols[0]), int(cols[-1])
        y0, y1 = int(rows[0]), int(rows[-1])
        bbox = [x0, y0, x1 - x0 + 1, y1 - y0 + 1]

        instances.append({
            "category_id": BEACH_LITTER_CATEGORY["id"],
            "segmentation": polygons,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0,
        })
    return instances


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(
    root: Path,
    output_json: Path,
    litter_class_id: int,
    min_area: int,
) -> None:
    pairs = _find_pairs(root)
    if not pairs:
        print(f"ERROR: No image/mask pairs found in {root}")
        print("Run with --inspect to see what files are present.")
        return

    output_json.parent.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_anns = []
    image_id = 1
    ann_id = 1
    skipped = 0

    for img_path, mask_path in tqdm(pairs, desc="  Converting", unit="img"):
        try:
            with PILImage.open(img_path) as im:
                w, h = im.size
        except Exception as exc:
            print(f"  [warn] Cannot open image {img_path.name}: {exc}")
            skipped += 1
            continue

        try:
            mask_img = PILImage.open(mask_path)
            if mask_img.mode == "P":
                mask_arr = np.array(mask_img)
            else:
                mask_arr = np.array(mask_img.convert("L"))
        except Exception as exc:
            print(f"  [warn] Cannot open mask {mask_path.name}: {exc}")
            skipped += 1
            continue

        instances = extract_instances(mask_arr, litter_class_id, min_area)
        if not instances:
            continue  # image has no litter → skip entirely

        try:
            rel = img_path.relative_to(root)
        except ValueError:
            rel = img_path.name

        coco_images.append({
            "id": image_id,
            "file_name": str(rel),
            "width": w,
            "height": h,
        })
        for inst in instances:
            inst["id"] = ann_id
            inst["image_id"] = image_id
            coco_anns.append(inst)
            ann_id += 1

        image_id += 1

    coco_out = {
        "info": {
            "description": "BeachLitter v2022 — converted to COCO instance segmentation",
            "version": "1.0",
            "year": 2022,
            "notes": (
                f"Litter class pixel id={litter_class_id}. "
                f"All instances mapped to 'other' (id=9). "
                f"Min area filter: {min_area} px²."
            ),
        },
        "categories": ALL_CATEGORIES,
        "images": coco_images,
        "annotations": coco_anns,
    }

    with open(output_json, "w") as f:
        json.dump(coco_out, f)

    print(f"\nDone.")
    print(f"  Images with litter : {len(coco_images)} / {len(pairs)}")
    print(f"  Total instances    : {len(coco_anns)}")
    print(f"  Skipped (errors)   : {skipped}")
    print(f"  Output             : {output_json}")
    print(
        "\nNext: run build_hybrid_coco.py — it will auto-detect this JSON "
        "at datasets/raw/beach_litter/annotations/instances.json"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert BeachLitter v2022 semantic masks to COCO instance segmentation JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--beach-litter-dir", default="datasets/raw/beach_litter/", metavar="DIR",
        help="Root directory of extracted BeachLitter dataset (default: datasets/raw/beach_litter/)",
    )
    parser.add_argument(
        "--output-json",
        default="datasets/raw/beach_litter/annotations/instances.json",
        metavar="JSON",
        help="Path for the output COCO JSON (default: datasets/raw/beach_litter/annotations/instances.json)",
    )
    parser.add_argument(
        "--litter-class-id", type=int, default=1, metavar="INT",
        help="Pixel value in the mask PNG that corresponds to artificial litter (default: 1). "
             "Run --inspect first if unsure.",
    )
    parser.add_argument(
        "--min-area", type=int, default=200, metavar="INT",
        help="Minimum instance area in pixels to keep (filters noise blobs, default: 200).",
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="Print pixel value distribution from a sample of masks and exit (no conversion).",
    )
    args = parser.parse_args()

    root = Path(args.beach_litter_dir)
    if not root.exists():
        print(f"ERROR: {root} does not exist. Download BeachLitter first.")
        return

    if args.inspect:
        inspect(root)
        return

    print(f"Converting BeachLitter: {root}")
    print(f"  Litter class pixel id : {args.litter_class_id}")
    print(f"  Min instance area     : {args.min_area} px²")
    print()
    convert(
        root=root,
        output_json=Path(args.output_json),
        litter_class_id=args.litter_class_id,
        min_area=args.min_area,
    )


if __name__ == "__main__":
    main()
