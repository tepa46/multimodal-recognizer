#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime
import json
import re
from pathlib import Path

CLASSES = ["plastic_bottle", "glass_bottle", "glass_jar", "metal_can",
           "carton", "tetrapack", "paper", "plastic_bag",
           "plastic_container", "other"]


def image_id_of(fn: str) -> int:
    m = re.search(r"(\d+)", Path(fn).stem)
    if not m:
        raise ValueError(f"no digits in {fn}")
    return int(m.group(1))


def clamp_bbox(bbox, W, H):
    x, y, w, h = (float(v) for v in bbox)
    x2, y2 = min(x + w, W), min(y + h, H)
    x, y = max(0.0, x), max(0.0, y)
    return [round(x, 1), round(y, 1), round(x2 - x, 1), round(y2 - y, 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/real495")
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)

    state = json.loads((root / "review_state.json").read_text())
    proposals = json.loads((root / "proposals.json").read_text())
    dims = {im["file_name"]: (im["width"], im["height"])
            for im in proposals["images"]}

    images, annotations, meas_rows, problems = [], [], [], []
    cls_count = {c: 0 for c in CLASSES}
    cond_count: dict[str, int] = {}
    n_unsure = 0
    exported = 0

    for fn in sorted(state["images"]):
        st = state["images"][fn]
        if not st.get("exhaustive") and not args.allow_incomplete:
            continue
        W, H = dims[fn]
        img_id = image_id_of(fn)
        exported += 1
        if not st["boxes"]:
            problems.append(f"{fn}: done, но 0 боксов (пустой кадр?)")
        images.append({"id": img_id, "file_name": fn, "width": W, "height": H,
                       "source": "real495", "original_id": img_id})

        boxes = []
        for b in st["boxes"]:
            if not b.get("class"):
                problems.append(f"{fn}: бокс #{b['bid']} без класса — пропущен")
                continue
            bbox = clamp_bbox(b["bbox"], W, H)
            if bbox[2] < 4 or bbox[3] < 4:
                problems.append(f"{fn}: бокс #{b['bid']} вырожден — пропущен")
                continue
            boxes.append((b, bbox))

        # pos: left-to-right by bbox x-center (tie -> y-center)
        order = sorted(boxes, key=lambda t: (t[1][0] + t[1][2] / 2,
                                             t[1][1] + t[1][3] / 2))
        for pos, (b, bbox) in enumerate(order, start=1):
            cond = b.get("condition", "normal")
            cls_count[b["class"]] += 1
            cond_count[cond] = cond_count.get(cond, 0) + 1
            n_unsure += 1 if b.get("unsure") else 0
            annotations.append({
                "id": img_id * 100 + b["bid"],
                "image_id": img_id,
                "category_id": CLASSES.index(b["class"]),
                "bbox": bbox,
                "area": round(bbox[2] * bbox[3], 1),
                "iscrowd": 0,
                "segmentation": [],
                "attributes": {
                    "pos": pos,
                    "condition": cond,
                    "unsure": bool(b.get("unsure")),
                    "notes": b.get("notes", ""),
                    "origin": b.get("origin", "manual"),
                    "det_class": b.get("det_class"),
                    "det_score": b.get("det_score"),
                    "siglip_sugg": b.get("siglip_sugg"),
                },
            })
            meas_rows.append({
                "filename": fn, "pos": pos, "object_id": "",
                "class": b["class"], "height_cm": "", "width_cm": "",
                "depth_cm": "", "mass_g": "", "mass_empty_g": "",
                "condition": cond, "fill_content": "",
                "notes": b.get("notes", ""),
            })

    coco = {
        "info": {"description": "real495 — real-sensor fraud evaluation set "
                                "(author photos), bbox stage",
                 "version": "1.0", "year": 2026,
                 "date_created": datetime.date.today().isoformat(),
                 "annotator": state.get("annotator", "")},
        "categories": [{"id": i, "name": c, "supercategory": "waste"}
                       for i, c in enumerate(CLASSES)],
        "images": images,
        "annotations": annotations,
    }
    out_dir = root / "annotations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "instances_boxes.json"
    out_json.write_text(json.dumps(coco, ensure_ascii=False))

    csv_path = root / "measurements.csv"
    if csv_path.exists():
        csv_path = root / "measurements.csv.new"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(meas_rows[0].keys())
                            if meas_rows else
                            ["filename", "pos", "object_id", "class",
                             "height_cm", "width_cm", "depth_cm", "mass_g",
                             "mass_empty_g", "condition", "fill_content",
                             "notes"])
        wr.writeheader()
        wr.writerows(meas_rows)

    total = len(dims)
    print(f"[export] images done: {exported}/{total}  "
          f"annotations: {len(annotations)}  unsure: {n_unsure}")
    print(f"[export] classes: " + ", ".join(
        f"{k}={v}" for k, v in sorted(cls_count.items(), key=lambda x: -x[1])
        if v))
    print(f"[export] condition: {cond_count}")
    print(f"[export] wrote {out_json}")
    print(f"[export] wrote {csv_path}")
    if problems:
        print(f"[export] ПРОБЛЕМЫ ({len(problems)}):")
        for p in problems:
            print("   -", p)


if __name__ == "__main__":
    main()
