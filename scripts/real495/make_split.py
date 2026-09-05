#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path,
                    default=Path("datasets/real495/annotations/instances_all.json"))
    ap.add_argument("--measurements", type=Path,
                    default=Path("datasets/real495/measurements.csv"))
    ap.add_argument("--train-frac", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path,
                    default=Path("datasets/real495/split_train_heldout.json"))
    args = ap.parse_args()

    coco = json.loads(args.coco.read_text())
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {im["id"]: im for im in coco["images"]}

    # per-frame signature for stratification
    per_frame_classes: dict[str, list[str]] = defaultdict(list)
    per_frame_fraud: dict[str, bool] = defaultdict(bool)
    for a in coco["annotations"]:
        fn = imgs[a["image_id"]]["file_name"]
        per_frame_classes[fn].append(id2name[a["category_id"]])
        if a.get("attributes", {}).get("condition", "normal") != "normal":
            per_frame_fraud[fn] = True

    # physical-object identity, to measure (not prevent) leakage
    meas = list(csv.DictReader(open(args.measurements)))
    frame_objects: dict[str, set] = defaultdict(set)
    for r in meas:
        sig = (r["class"], r["height_cm"], r["width_cm"],
               r["depth_cm"], r.get("mass_empty_g", ""))
        frame_objects[r["filename"]].add(sig)

    frames = sorted(imgs[i]["file_name"] for i in imgs)
    strata: dict[tuple, list[str]] = defaultdict(list)
    for fn in frames:
        sig = (tuple(sorted(set(per_frame_classes[fn]))), per_frame_fraud[fn])
        strata[sig].append(fn)

    rng = random.Random(args.seed)
    train, heldout = [], []
    for sig, group in sorted(strata.items()):
        g = group[:]
        rng.shuffle(g)
        if len(g) == 1:
            (train if rng.random() < args.train_frac else heldout).extend(g)
            continue
        k = round(len(g) * args.train_frac)
        k = min(max(k, 1), len(g) - 1)   # both sides get at least one frame
        train.extend(g[:k])
        heldout.extend(g[k:])
    train.sort(); heldout.sort()

    tr_objs = {o for fn in train for o in frame_objects.get(fn, ())}
    ho_objs = {o for fn in heldout for o in frame_objects.get(fn, ())}
    shared = tr_objs & ho_objs

    def ann_count(fnames):
        s = set(fnames)
        return sum(1 for a in coco["annotations"]
                   if imgs[a["image_id"]]["file_name"] in s)

    out = {
        "meta": {
            "seed": args.seed, "train_frac": args.train_frac,
            "split_level": "frame",
            "rationale": "object-level split impossible: all 32 objects form "
                         "one connected component across 215 frames",
        },
        "train": train,
        "heldout": heldout,
        "leakage": {
            "objects_train": len(tr_objs),
            "objects_heldout": len(ho_objs),
            "objects_shared": len(shared),
            "share_of_heldout_objects_seen_in_train":
                round(len(shared) / max(len(ho_objs), 1), 3),
            "note": "Same physical objects appear on both sides (different "
                    "scenes/poses). Held-out detector metrics are optimistic "
                    "w.r.t. unseen objects — disclose in the paper.",
        },
    }
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"[split] train frames  : {len(train):>4}  ({ann_count(train)} annotations)")
    print(f"[split] heldout frames: {len(heldout):>4}  ({ann_count(heldout)} annotations)")
    print(f"[split] fraud frames  : train "
          f"{sum(1 for f in train if per_frame_fraud[f])}, "
          f"heldout {sum(1 for f in heldout if per_frame_fraud[f])}")
    print(f"[leakage] distinct physical objects: train {len(tr_objs)}, "
          f"heldout {len(ho_objs)}, shared {len(shared)} "
          f"({len(shared)/max(len(ho_objs),1):.0%} of held-out objects were seen in train)")
    print(f"[split] wrote {args.out}")


if __name__ == "__main__":
    main()
