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


def estimate_volume_cm3(bbox_area_px: float, image_area_px: float,
                        clamp: bool = True) -> float:
    if image_area_px <= 0 or bbox_area_px <= 0:
        return 1.0
    rel = bbox_area_px / image_area_px
    # reference: object covering 25% of image → ~1000 cm^3
    volume = (rel ** 1.5) * 8000.0
    return max(volume, 1.0) if clamp else volume


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
    parser.add_argument("--calibration", type=Path, default=None,
                        help="Calibration YAML (configs/calibration_packphys.yaml). "
                             "Switches on the calibrated mode described in the "
                             "module docstring.")
    parser.add_argument("--table-calibration", type=Path, default=None,
                        help="Calibrated mode: take the evaluator side (density "
                             "table, volume scale and noise) from THIS yaml and "
                             "only the specimen draw pool from --calibration. Use "
                             "it to generate masses from specimens that are "
                             "disjoint from the ones behind the table.")
    parser.add_argument("--volume-noise", type=float, default=None,
                        help="Calibrated mode: replace the per-class measured "
                             "proxy scatter by one log-s.d. for all classes "
                             "(0 disables the proxy error).")
    parser.add_argument("--density-spread", type=float, default=None,
                        help="Calibrated mode: instead of a real specimen's "
                             "density, draw rho_c * exp(N(0, s^2)) (sensitivity "
                             "axis; 0 reproduces a noise-free table).")
    parser.add_argument("--min-volume-cm3", type=float, default=1.0,
                        help="Calibrated mode: skip instances whose calibrated "
                             "volume is below this (sub-cm^3 proxies of tiny "
                             "litter items that no RVM accepts).")
    parser.add_argument("--heavy-fill-range", type=float, nargs=2, default=None,
                        help="Override the heavy_fill multiplier with U(lo, hi) "
                             "(both modes).")
    parser.add_argument("--foreign-object-range", type=float, nargs=2, default=None,
                        help="Override the foreign_object multiplier with U(lo, hi).")
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
    fraud_multipliers: dict[str, list[float]] = dict(cfg["fraud_multipliers"])
    if args.heavy_fill_range:
        fraud_multipliers["heavy_fill"] = list(args.heavy_fill_range)
    if args.foreign_object_range:
        fraud_multipliers["foreign_object"] = list(args.foreign_object_range)

    calib: dict[str, Any] | None = None
    calib_eval: dict[str, Any] | None = None
    if args.calibration is not None:
        with open(args.calibration) as f:
            calib = yaml.safe_load(f)
        calib_eval = calib
        if args.table_calibration is not None:
            with open(args.table_calibration) as f:
                calib_eval = yaml.safe_load(f)
        densities = calib_eval["densities"]
        sigma_w = float(calib_eval.get("weight_noise_sigma", sigma_w))
        if args.weight_noise_sigma is not None:
            sigma_w = float(args.weight_noise_sigma)
        fraud_multipliers.setdefault("foreign_object", [1.8, 3.5])
        fraud_multipliers.update({k: v for k, v in calib.get("fraud_multipliers", {}).items()
                                  if k not in ("heavy_fill",) or not args.heavy_fill_range})
        if args.foreign_object_range:
            fraud_multipliers["foreign_object"] = list(args.foreign_object_range)
        print(f"calibrated mode: draw pool {args.calibration} ({calib['n_specimens']} specimens), "
              f"table {args.table_calibration or args.calibration} ({calib_eval['n_specimens']} specimens)")

    with open(args.coco_json) as f:
        coco = json.load(f)

    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    name_to_density = {name: densities.get(name, 0.5) for name in id_to_name.values()}
    densities_by_id = {cid: name_to_density[name] for cid, name in id_to_name.items()}
    num_classes = len(coco["categories"])

    spec_pool: dict[int, list[dict[str, Any]]] = {}
    fill_pool: dict[int, list[float]] = {}
    all_fills: list[float] = []
    if calib is not None:
        for cid, name in id_to_name.items():
            spec_pool[cid] = list(calib["specimens"].get(name, []))
            fill_pool[cid] = [s["fill_ratio"] for s in spec_pool[cid] if s.get("fill_ratio")]
            all_fills.extend(fill_pool[cid])
        missing = [id_to_name[c] for c, p in spec_pool.items() if not p]
        if missing:
            raise SystemExit(f"calibration has no specimens for classes {missing}")

    def draw_density(cid: int) -> tuple[float, str]:
        if args.density_spread is not None:
            kappa = math.exp(rng.gauss(0.0, args.density_spread)) if args.density_spread > 0 else 1.0
            return densities_by_id[cid] * kappa, ""
        s = rng.choice(spec_pool[cid])
        return float(s["density"]), str(s["id"])

    import statistics as _st

    table_pool: dict[int, list[dict[str, Any]]] = {}
    if calib_eval is not None:
        for cid, name in id_to_name.items():
            table_pool[cid] = list(calib_eval["specimens"].get(name, []))

    def table_density(cid: int, exclude: str) -> float:
        if calib is None or not exclude:
            return densities_by_id[cid]
        vals = [float(s["density"]) for s in table_pool[cid] if str(s["id"]) != exclude]
        return _st.median(vals) if vals else densities_by_id[cid]

    def draw_fill(cid: int) -> float:
        pool = fill_pool.get(cid) or all_fills
        return float(rng.choice(pool))

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
    skipped_small = 0
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
        ] + (["specimen_id", "expected_weight_loso_g"] if calib is not None else []))

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

            source = Path(img["file_name"]).name.split("_")[0]
            if calib is not None and source not in calib_eval["volume_scale"]:
                raise SystemExit(f"no volume_scale for source {source!r} in {args.calibration}")
            for ann in anns:
                x, y, bw, bh = ann["bbox"]
                bbox_area = bw * bh
                true_cid = ann["category_id"]
                true_name = id_to_name[true_cid]
                specimen_id = ""

                if calib is None:
                    volume = estimate_volume_cm3(bbox_area, image_area)
                    volume *= float(np_rng.normal(1.0, sigma_d))
                    volume = max(volume, 0.5)
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
                else:
                    scale = float(calib_eval["volume_scale"].get(source, 1.0))
                    v_sig = (args.volume_noise if args.volume_noise is not None
                             else float(calib_eval["volume_sigma"][true_name]))
                    volume = scale * estimate_volume_cm3(bbox_area, image_area, clamp=False)
                    if v_sig > 0:
                        volume *= math.exp(float(np_rng.normal(0.0, v_sig)))
                    if volume < args.min_volume_cm3:
                        skipped_small += 1
                        continue
                    rho_item, specimen_id = draw_density(true_cid)
                    measured_w = volume * rho_item * float(np_rng.normal(1.0, sigma_w))
                    measured_w = max(measured_w, 1e-3)
                    is_fraud = rng.random() < args.fraud_ratio
                    ftype = "normal"
                    reported_cid = true_cid
                    if is_fraud:
                        ftype = pick_fraud_type(rng)
                        fraud_count += 1
                        if ftype == "heavy_fill":
                            if args.heavy_fill_range:
                                lo, hi = fraud_multipliers["heavy_fill"]
                                measured_w *= rng.uniform(lo, hi)
                            else:
                                measured_w *= draw_fill(true_cid)
                        elif ftype == "foreign_object":
                            lo, hi = fraud_multipliers["foreign_object"]
                            measured_w *= rng.uniform(lo, hi)
                        else:
                            reported_cid = rng.choice(
                                [c for c in range(num_classes) if c != true_cid])
                            rho_claim, specimen_id = draw_density(reported_cid)
                            measured_w = max(
                                volume * rho_claim * float(rng.gauss(1.0, sigma_w)), 1e-3)
                    expected_w = volume * densities_by_id[reported_cid]
                    expected_w_loso = volume * table_density(reported_cid, specimen_id)

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
                    *( [f"{volume:.6g}", f"{expected_w:.6g}", f"{measured_w:.6g}"] if calib is not None
                       else [f"{volume:.3f}", f"{expected_w:.3f}", f"{measured_w:.3f}"] ),
                    int(is_fraud), ftype,
                ] + ([specimen_id, f"{expected_w_loso:.6g}"] if calib is not None else []))

    print(f"Wrote {total_count} tuples to {csv_path}")
    if calib is not None:
        print(f"  skipped {skipped_small} instances with volume < {args.min_volume_cm3} cm^3")
    print(f"  fraud: {fraud_count} ({100*fraud_count/max(total_count,1):.1f}%)")
    print("  per-class counts (true class):")
    for name, c in sorted(per_class_counts.items(), key=lambda x: -x[1]):
        print(f"    {name:20s} {c}")
    if not args.skip_depth:
        print(f"Depth maps: {depth_dir}/")


if __name__ == "__main__":
    main()
