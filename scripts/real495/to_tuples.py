#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

CONDITIONS = ("normal", "heavy_fill", "foreign_object")

HEADER = [
    "image_path", "depth_map_path", "annotation_id",
    "true_class_id", "true_class_name",
    "reported_class_id", "reported_class_name",
    "bbox_x", "bbox_y", "bbox_w", "bbox_h",
    "volume_cm3", "expected_weight_g", "measured_weight_g",
    "is_fraud", "fraud_type", "specimen_id", "expected_weight_loso_g",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="datasets/real495/annotations/instances_all.json")
    ap.add_argument("--measurements", default="datasets/real495/measurements.csv")
    ap.add_argument("--image-root", default="datasets/real495/images")
    ap.add_argument("--density-config", default="configs/calibration_packphys.yaml",
                    help="Density table for the expected mass. The revision "
                         "uses the table calibrated on specimens disjoint from "
                         "real495 (configs/calibration_packphys.yaml).")
    ap.add_argument("--out", default="datasets/multimodal/real495/tuples.csv")
    args = ap.parse_args()

    coco = json.loads(Path(args.coco).read_text())
    cfg = yaml.safe_load(Path(args.density_config).read_text())
    densities: dict[str, float] = cfg["densities"]

    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}

    meas_by_key: dict[tuple[str, int], dict] = {}
    with open(args.measurements) as f:
        for row in csv.DictReader(f):
            meas_by_key[(row["filename"], int(row["pos"]))] = row

    image_root = Path(args.image_root)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out: list[list] = []
    missing: list[tuple] = []
    cond_counts = {c: 0 for c in CONDITIONS}
    cls_counts: dict[str, int] = {}

    for ann in coco["annotations"]:
        img = images[ann["image_id"]]
        fn = img["file_name"]
        attrs = ann.get("attributes", {})
        pos = attrs.get("pos")
        m = meas_by_key.get((fn, pos))
        if (m is None or m.get("mass_g") in (None, "")
                or m.get("height_cm") in (None, "")
                or m.get("width_cm") in (None, "")
                or m.get("depth_cm") in (None, "")):
            missing.append((fn, pos, ann["id"]))
            continue

        cid = ann["category_id"]
        cname = id_to_name[cid]
        h, w, d = float(m["height_cm"]), float(m["width_cm"]), float(m["depth_cm"])
        volume = h * w * d
        measured_w = float(m["mass_g"])
        density = densities.get(cname, 0.5)
        expected_w = volume * density

        condition = attrs.get("condition") or "normal"
        if condition not in CONDITIONS:
            condition = "normal"
        is_fraud = int(condition != "normal")
        fraud_type = condition if is_fraud else "normal"
        cond_counts[condition] = cond_counts.get(condition, 0) + 1
        cls_counts[cname] = cls_counts.get(cname, 0) + 1

        x, y, bw, bh = ann["bbox"]
        rows_out.append([
            str((image_root / fn).as_posix()),
            "",  # no synthetic depth map — the real "depth" IS the measured value
            ann["id"],
            cid, cname,
            cid, cname,            # reported == true: no wrong_class by construction
            f"{x:.2f}", f"{y:.2f}", f"{bw:.2f}", f"{bh:.2f}",
            f"{volume:.6g}", f"{expected_w:.6g}", f"{measured_w:.6g}",
            is_fraud, fraud_type,
            m.get("specimen_id", ""),
            f"{expected_w:.6g}",
        ])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows_out)

    print(f"[tuples] wrote {len(rows_out)} tuples -> {out_path}")
    if missing:
        print(f"[tuples] ПРОПУЩЕНО (нет измерений): {len(missing)}")
        for x in missing:
            print("   ", x)
    print(f"[tuples] condition: {cond_counts}")
    print("[tuples] classes: " + ", ".join(
        f"{k}={v}" for k, v in sorted(cls_counts.items(), key=lambda x: -x[1])))
    n_fraud = sum(1 for r in rows_out if r[14] == 1)
    if rows_out:
        print(f"[tuples] fraud: {n_fraud}/{len(rows_out)} ({n_fraud / len(rows_out):.1%})")


if __name__ == "__main__":
    main()
