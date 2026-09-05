from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from tqdm import tqdm


DW_CLASS_TO_CAT = {
    "Aluminium": "metal_can",
    "Glass": "glass_bottle",
    "HDPE": "plastic_container",
    "PET": "plastic_bottle",
}

UNIFIED_CATEGORIES = [
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

CAT_NAME_TO_ID = {c["name"]: c["id"] for c in UNIFIED_CATEGORIES}


# ---------------------------------------------------------------------------
# Mask encoding
# ---------------------------------------------------------------------------

def mask_to_polygon(mask: np.ndarray) -> list[list[float]]:
    """Convert binary mask to a list of COCO polygon contours."""
    try:
        import cv2
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = []
        for c in contours:
            if c.size >= 6:
                polygons.append(c.flatten().tolist())
        return polygons
    except ImportError:
        pass

    # Fallback: simple border trace without OpenCV
    from skimage import measure
    contours = measure.find_contours(mask, 0.5)
    polygons = []
    for contour in contours:
        if len(contour) >= 4:
            # skimage returns (row, col) = (y, x) — swap for COCO (x, y)
            flat = []
            for pt in contour:
                flat.extend([float(pt[1]), float(pt[0])])
            polygons.append(flat)
    return polygons


def mask_to_rle(mask: np.ndarray) -> dict:
    """Encode binary mask as COCO RLE."""
    from pycocotools import mask as mask_util
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def bbox_from_mask(mask: np.ndarray) -> list[float]:
    """Compute tight COCO bbox [x, y, w, h] from binary mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_dataset(
    dataset_dir: Path,
    predictor: Any | None,  # SamPredictor or None in dry-run mode
    use_rle: bool,
) -> tuple[list[dict], list[dict]]:
    """
    Walk the Drinking Waste class folders, run SAM, return COCO images + annotations.
    """
    coco_images: list[dict] = []
    coco_anns: list[dict] = []
    image_id = 1
    ann_id = 1

    for class_name, unified_name in DW_CLASS_TO_CAT.items():
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            print(f"  [skip] {class_dir} not found")
            continue

        img_paths = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        cat_id = CAT_NAME_TO_ID[unified_name]

        for img_path in tqdm(img_paths, desc=f"  {class_name}", unit="img"):
            with PILImage.open(img_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                w, h = pil_img.size
                np_img = np.array(pil_img)

            file_name = f"{class_name}/{img_path.name}"
            coco_images.append({
                "id": image_id,
                "file_name": file_name,
                "width": w,
                "height": h,
            })

            # Whole-image bbox prompt for SAM
            box = np.array([0, 0, w, h], dtype=float)

            if predictor is not None:
                predictor.set_image(np_img)
                masks, scores, _ = predictor.predict(
                    box=box,
                    multimask_output=True,
                )
                # Pick the mask with highest confidence
                best_idx = int(np.argmax(scores))
                mask = masks[best_idx].astype(bool)
            else:
                # Dry run: full-image mask
                mask = np.ones((h, w), dtype=bool)

            area = float(mask.sum())
            tight_bbox = bbox_from_mask(mask)

            if use_rle:
                segmentation = mask_to_rle(mask)
                is_rle = True
            else:
                segmentation = mask_to_polygon(mask)
                is_rle = False

            coco_anns.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": tight_bbox,
                "area": area,
                "iscrowd": 0,
                "segmentation": segmentation if not is_rle else {"counts": segmentation["counts"], "size": segmentation["size"]},
            })
            image_id += 1
            ann_id += 1

    return coco_images, coco_anns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate SAM instance masks for the Drinking Waste dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--drinking-waste-dir", default="datasets/raw/drinking_waste/", metavar="DIR",
        help="Path to extracted Drinking Waste directory (default: datasets/raw/drinking_waste/)",
    )
    parser.add_argument(
        "--sam-checkpoint", default="checkpoints/sam_vit_h_4b8939.pth", metavar="PTH",
        help="Path to SAM checkpoint .pth file.",
    )
    parser.add_argument(
        "--model-type", default="vit_h", choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type matching the checkpoint (default: vit_h).",
    )
    parser.add_argument(
        "--device", default="cuda", metavar="DEVICE",
        help="Torch device: cuda, cpu, mps (default: cuda).",
    )
    parser.add_argument(
        "--output-json", default=None, metavar="JSON",
        help="Output COCO JSON path. "
             "Default: <drinking-waste-dir>/annotations/instances_sam.json",
    )
    parser.add_argument(
        "--use-rle", action="store_true",
        help="Store segmentation as COCO RLE instead of polygon (smaller files for large masks).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip SAM inference; write JSON with full-image masks. "
             "Useful for testing the pipeline without GPU.",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.drinking_waste_dir)
    if not dataset_dir.exists():
        print(f"ERROR: {dataset_dir} not found. Run download_datasets.py first.")
        return

    out_json = Path(args.output_json) if args.output_json else (
        dataset_dir / "annotations" / "instances_sam.json"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)

    predictor = None
    if not args.dry_run:
        ckpt = Path(args.sam_checkpoint)
        if not ckpt.exists():
            print(f"ERROR: SAM checkpoint not found: {ckpt}")
            print("Download with:")
            print(f"  wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth "
                  f"-P checkpoints/")
            return

        print(f"Loading SAM ({args.model_type}) on {args.device}...")
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError:
            print("segment-anything not installed. Run:")
            print("  pip install git+https://github.com/facebookresearch/segment-anything.git")
            return

        sam = sam_model_registry[args.model_type](checkpoint=str(ckpt))
        sam.to(device=args.device)
        predictor = SamPredictor(sam)
        print("SAM loaded.\n")
    else:
        print("Dry-run mode: SAM inference disabled, writing placeholder masks.\n")

    print(f"Processing: {dataset_dir}")
    coco_images, coco_anns = process_dataset(dataset_dir, predictor, args.use_rle)

    coco_out = {
        "info": {
            "description": "Drinking Waste — SAM auto-annotated masks",
            "version": "1.0",
            "year": 2025,
        },
        "categories": UNIFIED_CATEGORIES,
        "images": coco_images,
        "annotations": coco_anns,
    }

    with open(out_json, "w") as f:
        json.dump(coco_out, f)

    print(f"\nSaved: {out_json}")
    print(f"  Images     : {len(coco_images)}")
    print(f"  Annotations: {len(coco_anns)}")
    print("\nNext step: run build_hybrid_coco.py to merge into the unified dataset.")


if __name__ == "__main__":
    main()
