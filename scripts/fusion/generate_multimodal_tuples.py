from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image


FRAUD_TYPES = ("heavy_fill", "foreign_object", "wrong_class")


def load_density_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def estimate_volume_cm3(bbox_area_px: float, image_area_px: float) -> float:
    if image_area_px <= 0 or bbox_area_px <= 0:
        return 1.0
    rel = bbox_area_px / image_area_px
    # reference: object covering 25% of image → ~1000 cm^3
    volume = (rel ** 1.5) * 8000.0
    return max(volume, 1.0)


def pick_fraud_type(rng: random.Random) -> str:
    return rng.choice(FRAUD_TYPES)


def apply_fraud(
    true_class_id: int,
    true_weight: float,
    fraud_type: str,
    fraud_multipliers: dict[str, list[float]],
    num_classes: int,
    rng: random.Random,
    *,
    volume: float = 1.0,
    densities_by_id: dict[int, float] | None = None,
    legit_sigma: float = 0.05,
) -> tuple[int, float]:
    lo, hi = fraud_multipliers[fraud_type]
    mult = rng.uniform(lo, hi)
    if fraud_type == "wrong_class":
        other = [c for c in range(num_classes) if c != true_class_id]
        reported_class = rng.choice(other)
        if densities_by_id is not None:
            density = densities_by_id.get(reported_class, 0.5)
            plausible = volume * density * float(rng.gauss(1.0, legit_sigma))
            return reported_class, max(plausible, 0.1)
        # Fallback to the legacy multiplicative mass if densities missing.
        return reported_class, true_weight * mult
    # heavy_fill / foreign_object: class is correct, mass is wrong
    return true_class_id, true_weight * mult


def build_depth_map(
    image_size: tuple[int, int],
    instances: list[dict[str, Any]],
) -> np.ndarray:
    w, h = image_size
    depth = np.full((h, w), 2000, dtype=np.uint16)
    total_area = float(w * h)
    for inst in instances:
        x, y, bw, bh = inst["bbox"]
        rel = (bw * bh) / total_area if total_area > 0 else 0.0
        # depth proxy: closer for larger objects, clamped to [300, 1800] mm
        d = int(max(300, min(1800, 1800 - 1500 * math.sqrt(rel))))
        x0 = max(0, int(x)); y0 = max(0, int(y))
        x1 = min(w, int(x + bw)); y1 = min(h, int(y + bh))
        if x1 > x0 and y1 > y0:
            depth[y0:y1, x0:x1] = d
    return depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True,
                        help="Directory with images referenced by the COCO JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--density-config", type=Path,
                        default=Path("configs/material_density.yaml"))
    parser.add_argument("--fraud-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-depth", action="store_true",
                        help="Skip writing depth PNGs (CSV only)")
    parser.add_argument("--weight-noise-sigma", type=float, default=None,
                        help="Override weight_noise_sigma from the YAML "
                             "(used by sensor-noise sweep, paper §6.6)")
    parser.add_argument("--depth-noise-sigma", type=float, default=None,
                        help="Override depth_noise_sigma from the YAML")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    cfg = load_density_config(args.density_config)
    densities: dict[str, float] = cfg["densities"]
    sigma_w = float(cfg.get("weight_noise_sigma", 0.05))
    sigma_d = float(cfg.get("depth_noise_sigma", 0.10))
    if args.weight_noise_sigma is not None:
        sigma_w = float(args.weight_noise_sigma)
    if args.depth_noise_sigma is not None:
        sigma_d = float(args.depth_noise_sigma)
    print(f"sigma_w={sigma_w:.4f}, sigma_d={sigma_d:.4f}")
    fraud_multipliers: dict[str, list[float]] = cfg["fraud_multipliers"]

    with open(args.coco_json) as f:
        coco = json.load(f)

    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    name_to_density = {name: densities.get(name, 0.5) for name in id_to_name.values()}
    densities_by_id = {cid: name_to_density[name] for cid, name in id_to_name.items()}
    num_classes = len(coco["categories"])

    images_by_id: dict[int, dict[str, Any]] = {img["id"]: img for img in coco["images"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    depth_dir = args.output_dir / "depth"
    if not args.skip_depth:
        depth_dir.mkdir(exist_ok=True)

    csv_path = args.output_dir / "tuples.csv"
    fraud_count = 0
    total_count = 0
    per_class_counts = {name: 0 for name in id_to_name.values()}

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path", "depth_map_path", "annotation_id",
            "true_class_id", "true_class_name",
            "reported_class_id", "reported_class_name",
            "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "volume_cm3", "expected_weight_g", "measured_weight_g",
            "is_fraud", "fraud_type",
        ])

        for img_id, img in images_by_id.items():
            anns = anns_by_image.get(img_id, [])
            if not anns:
                continue
            img_w, img_h = img["width"], img["height"]
            image_area = float(img_w * img_h)

            # Write one depth map per image (shared across its annotations).
            if not args.skip_depth:
                depth = build_depth_map((img_w, img_h), anns)
                depth_path = depth_dir / (Path(img["file_name"]).stem + ".png")
                Image.fromarray(depth).save(depth_path)
                depth_rel = str(depth_path.relative_to(args.output_dir))
            else:
                depth_rel = ""

            for ann in anns:
                x, y, bw, bh = ann["bbox"]
                bbox_area = bw * bh
                volume = estimate_volume_cm3(bbox_area, image_area)
                volume *= float(np_rng.normal(1.0, sigma_d))
                volume = max(volume, 0.5)

                true_cid = ann["category_id"]
                true_name = id_to_name[true_cid]
                density = name_to_density[true_name]
                expected_w = volume * density
                measured_w = expected_w * float(np_rng.normal(1.0, sigma_w))
                measured_w = max(measured_w, 1e-3)

                is_fraud = rng.random() < args.fraud_ratio
                if is_fraud:
                    ftype = pick_fraud_type(rng)
                    reported_cid, measured_w = apply_fraud(
                        true_cid, measured_w, ftype, fraud_multipliers,
                        num_classes, rng,
                        volume=volume,
                        densities_by_id=densities_by_id,
                        legit_sigma=sigma_w,
                    )
                    fraud_count += 1
                else:
                    ftype = "normal"
                    reported_cid = true_cid

                reported_name = id_to_name[reported_cid]
                per_class_counts[true_name] += 1
                total_count += 1

                writer.writerow([
                    str((args.image_root / img["file_name"]).as_posix()),
                    depth_rel,
                    ann["id"],
                    true_cid, true_name,
                    reported_cid, reported_name,
                    f"{x:.2f}", f"{y:.2f}", f"{bw:.2f}", f"{bh:.2f}",
                    f"{volume:.3f}", f"{expected_w:.3f}", f"{measured_w:.3f}",
                    int(is_fraud), ftype,
                ])

    print(f"Wrote {total_count} tuples to {csv_path}")
    print(f"  fraud: {fraud_count} ({100*fraud_count/max(total_count,1):.1f}%)")
    print("  per-class counts (true class):")
    for name, c in sorted(per_class_counts.items(), key=lambda x: -x[1]):
        print(f"    {name:20s} {c}")
    if not args.skip_depth:
        print(f"Depth maps: {depth_dir}/")


if __name__ == "__main__":
    main()
