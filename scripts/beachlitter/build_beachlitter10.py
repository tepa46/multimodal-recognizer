#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

PACKAGING = [
    "plastic_bottle", "glass_bottle", "glass_jar", "metal_can", "carton",
    "tetrapack", "paper", "plastic_bag", "plastic_container",
]


def load_verified(path: Path) -> dict[int, str]:
    with open(path) as f:
        return {int(r["ann_id"]): r["final_class"] for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--verified", required=True)
    ap.add_argument("--refined", required=True)
    ap.add_argument("--refined-fallback", default=None,
                    help="Phase-3 refined_masks.json (rescue-group masks)")
    ap.add_argument("--rescue-csv", default=None,
                    help="rescue verdicts; its `other` rows use SAM geometry")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from pycocotools import mask as mu

    coco = json.load(open(args.coco))
    verified = load_verified(Path(args.verified))
    cat_id = {c["name"]: c["id"] for c in coco["categories"]}
    other_id = cat_id["other"]

    refined: dict[int, dict] = {}
    if args.refined_fallback:
        refined.update({int(k): v for k, v in
                        json.load(open(args.refined_fallback)).items()})
    refined.update({int(k): v for k, v in json.load(open(args.refined)).items()})

    rescue_ids: set[int] = set()
    if args.rescue_csv:
        with open(args.rescue_csv) as f:
            rescue_ids = {int(r["ann_id"]) for r in csv.DictReader(f)}

    def sam_geometry(aid: int, ann: dict) -> bool:
        """Swap in SAM mask + tight bbox; keep original on any failure."""
        r = refined.get(aid)
        if r is None:
            return False
        rle = {"size": r["rle"]["size"], "counts": r["rle"]["counts"]}
        ann["segmentation"] = rle
        ann["bbox"] = [float(v) for v in r["bbox"]]
        ann["area"] = float(mu.area({"size": rle["size"],
                                     "counts": rle["counts"].encode("ascii")}))
        ann["sam_score"] = round(r["sam_score"], 4)
        return True

    stats = Counter()
    sam_used = sam_missing_pack = 0
    has_packaging: set[int] = set()
    for ann in coco["annotations"]:
        aid = ann["id"]
        cls = verified.get(aid)
        if cls in PACKAGING:
            ann["category_id"] = cat_id[cls]
            ann["iscrowd"] = 0
            if sam_geometry(aid, ann):
                sam_used += 1
            else:
                sam_missing_pack += 1
            has_packaging.add(ann["image_id"])
            stats[cls] += 1
        elif cls == "other":
            ann["category_id"] = other_id
            ann["iscrowd"] = 0
            if aid in rescue_ids:
                sam_geometry(aid, ann)   # CC blob was the problem — use SAM
            stats["other_reviewed"] += 1
        elif cls == "ignore_merged":
            ann["category_id"] = other_id
            ann["iscrowd"] = 1
            stats["ignore_merged"] += 1
        else:
            ann["category_id"] = other_id
            ann["iscrowd"] = 1
            stats["ignore_unreviewed"] += 1

    for im in coco["images"]:
        im["has_packaging"] = im["id"] in has_packaging

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco, open(out_p, "w"))

    print(f"wrote {out_p}")
    print(f"images: {len(coco['images'])} "
          f"(has_packaging: {len(has_packaging)})")
    print(f"annotations: {len(coco['annotations'])}")
    npack = sum(stats[c] for c in PACKAGING)
    print(f"\npackaging total: {npack} (SAM geometry: {sam_used}, "
          f"original CC geometry: {sam_missing_pack})")
    for c in PACKAGING:
        if stats[c]:
            print(f"  {c:<18} {stats[c]}")
    print(f"other (reviewed, iscrowd=0): {stats['other_reviewed']}")
    print(f"iscrowd=1: ignore_merged {stats['ignore_merged']} + "
          f"unreviewed {stats['ignore_unreviewed']}")

    # --- validation ---
    from pycocotools.coco import COCO
    c = COCO(str(out_p))
    n_eval = sum(1 for a in c.anns.values() if not a["iscrowd"])
    print(f"\npycocotools OK: {len(c.anns)} anns loaded, "
          f"{n_eval} evaluable (iscrowd=0)")
    bad_area = sum(1 for a in c.anns.values()
                   if not a["iscrowd"] and a["area"] <= 0)
    bad_bbox = sum(1 for a in c.anns.values()
                   if a["bbox"][2] <= 0 or a["bbox"][3] <= 0)
    print(f"zero-area evaluable anns: {bad_area}, degenerate bboxes: {bad_bbox}")


if __name__ == "__main__":
    main()
