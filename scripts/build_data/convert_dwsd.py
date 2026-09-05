from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image as PILImage
from tqdm import tqdm


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
NAME_TO_ID = {c["name"]: c["id"] for c in UNIFIED_CATEGORIES}


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    """Return a list of boolean masks, one per connected component of True pixels."""
    from scipy.ndimage import label
    labels, n = label(mask)
    return [(labels == i) for i in range(1, n + 1)]


def mask_to_polygon(component: np.ndarray) -> list[float] | None:
    """Convert a binary component into a single COCO polygon (flattened list)."""
    import cv2
    contours, _ = cv2.findContours(
        component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    # pick the largest external contour
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 3:
        return None
    poly = contour.reshape(-1).astype(float).tolist()
    if len(poly) < 6:
        return None
    return poly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dwsd-dir", type=Path, default=Path("datasets/raw/dwsd"))
    parser.add_argument(
        "--class-mapping", type=Path, default=Path("configs/class_mapping.yaml")
    )
    parser.add_argument("--min-area", type=int, default=100)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    class_cfg = yaml.safe_load(open(args.class_mapping))["dwsd"]
    # Resolve `class_N` keys into {N: unified_id}
    class_id_to_unified: dict[int, int] = {}
    for key, target_name in class_cfg.items():
        if key.startswith("class_"):
            cid = int(key.split("_", 1)[1])
            if target_name in NAME_TO_ID:
                class_id_to_unified[cid] = NAME_TO_ID[target_name]

    # Iterate splits
    images = []
    annotations = []
    next_img_id = 1
    next_ann_id = 1
    skipped = 0

    for split in ["Train", "Test"]:
        img_dir = args.dwsd_dir / "DSWD" / split / "Image"
        mask_dir = args.dwsd_dir / "DSWD" / split / "Mask"
        if not img_dir.is_dir():
            continue
        files = sorted(img_dir.iterdir())
        for img_path in tqdm(files, desc=f"DWSD-{split}"):
            mask_path = mask_dir / img_path.name.replace("img_", "mask_")
            if not mask_path.exists():
                skipped += 1
                continue
            try:
                mask = np.array(PILImage.open(mask_path))
                img = PILImage.open(img_path)
                w, h = img.size
            except Exception:
                skipped += 1
                continue

            image_record = {
                "id": next_img_id,
                "file_name": f"{split}/Image/{img_path.name}",
                "width": w,
                "height": h,
            }
            has_any = False

            # Per-class binary mask → connected components → polygon
            for cid in np.unique(mask):
                cid = int(cid)
                if cid == 0 or cid not in class_id_to_unified:
                    continue
                unified = class_id_to_unified[cid]
                bin_mask = (mask == cid)
                for comp in connected_components(bin_mask):
                    area = int(comp.sum())
                    if area < args.min_area:
                        continue
                    ys, xs = np.where(comp)
                    if ys.size == 0:
                        continue
                    x0, x1 = int(xs.min()), int(xs.max())
                    y0, y1 = int(ys.min()), int(ys.max())
                    bw, bh = x1 - x0 + 1, y1 - y0 + 1
                    poly = mask_to_polygon(comp)
                    if poly is None:
                        continue
                    annotations.append({
                        "id": next_ann_id,
                        "image_id": next_img_id,
                        "category_id": unified,
                        "bbox": [x0, y0, bw, bh],
                        "area": area,
                        "iscrowd": 0,
                        "segmentation": [poly],
                    })
                    next_ann_id += 1
                    has_any = True

            if has_any:
                images.append(image_record)
                next_img_id += 1

    out = args.output_json or (args.dwsd_dir / "annotations" / "instances.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": images,
        "annotations": annotations,
        "categories": UNIFIED_CATEGORIES,
    }
    with open(out, "w") as f:
        json.dump(coco, f)
    print(f"Done. {len(images)} images, {len(annotations)} anns, skipped={skipped}")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
