#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

def _load_density():
    import pathlib as _p
    txt = _p.Path("configs/material_density.yaml").read_text()
    out, in_block = {}, False
    for line in txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if not line.startswith((" ", "	")):
            in_block = s.startswith("densities")
            continue
        if in_block and ":" in s:
            k, v = s.split(":", 1)
            try:
                out[k.strip()] = float(v.split("#")[0].strip())
            except ValueError:
                pass
    if len(out) < 10:
        raise SystemExit("could not parse 10 densities from configs/material_density.yaml")
    return out


DENSITY = _load_density()
TAU = 0.25  # canonical rule threshold (val-swept on v3.3)


def iou_xywh(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2 = min(ax1 + aw, bx1 + bw)
    iy2 = min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter) if inter > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-detections", default="datasets/real495/raw_detections.json")
    ap.add_argument("--coco", default="datasets/real495/annotations/instances_all.json")
    ap.add_argument("--measurements", default="datasets/real495/measurements.csv")
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=TAU)
    ap.add_argument("--split-file", default=None,
                    help="datasets/real495/split_train_heldout.json — when the "
                         "detector was trained WITH real495 frames, routing must "
                         "only be measured on the held-out ones, otherwise it is "
                         "scored partly on its own training data.")
    ap.add_argument("--split", choices=["heldout", "train", "all"],
                    default="all")
    args = ap.parse_args()

    keep_frames = None
    if args.split_file and args.split != "all":
        split = json.loads(Path(args.split_file).read_text())
        keep_frames = set(split[args.split])
        print(f"[split] restricted to {args.split}: {len(keep_frames)} frames\n")

    raw = json.loads(Path(args.raw_detections).read_text())
    coco = json.loads(Path(args.coco).read_text())
    meas = {(r["filename"], int(r["pos"])): r
            for r in csv.DictReader(open(args.measurements))}
    id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    images = {im["id"]: im for im in coco["images"]}

    n_missed = n_correct = n_confused = 0
    confusion: Counter = Counter()
    flagged = silent = 0
    silent_rows = []

    for ann in coco["annotations"]:
        attrs = ann.get("attributes", {})
        if attrs.get("condition") != "normal":
            continue
        img = images[ann["image_id"]]
        fn = img["file_name"]
        if keep_frames is not None and fn not in keep_frames:
            continue
        true_name = id_to_name[ann["category_id"]]
        dets = [d for d in raw["detections"].get(fn, []) if d["score"] >= args.score_thr]
        best_iou, best_det = 0.0, None
        for d in dets:
            v = iou_xywh(ann["bbox"], d["bbox"])
            if v > best_iou:
                best_iou, best_det = v, d
        if best_det is None or best_iou < args.iou_thr:
            n_missed += 1
            continue
        rep_name = best_det["det_class"]
        if rep_name == true_name:
            n_correct += 1
            continue
        n_confused += 1
        confusion[(true_name, rep_name)] += 1

        m = meas.get((fn, attrs.get("pos")))
        if not m or not m.get("mass_g") or not m.get("height_cm"):
            continue
        h, w, d = float(m["height_cm"]), float(m["width_cm"]), float(m["depth_cm"])
        vol = h * w * d
        mass = float(m["mass_g"])
        exp_reported = vol * DENSITY[rep_name]
        rel_err = abs(mass - exp_reported) / max(exp_reported, 1e-3)
        if rel_err > args.tau:
            flagged += 1
        else:
            silent += 1
            silent_rows.append((fn, true_name, rep_name, round(vol, 1), mass,
                               round(exp_reported, 1), round(rel_err, 3)))

    n_normal = n_missed + n_correct + n_confused
    print(f"GT (condition=normal): {n_normal}")
    print(f"  не локализовано (IoU<{args.iou_thr} @ score>={args.score_thr}): {n_missed} "
          f"({n_missed/n_normal:.1%})")
    print(f"  локализовано: {n_missed + n_correct + n_confused - n_missed} "
          f"({(n_correct+n_confused)/n_normal:.1%})")
    loc = n_correct + n_confused
    print(f"    верный класс: {n_correct} ({n_correct/max(loc,1):.1%} of localized)")
    print(f"    РЕАЛЬНАЯ путаница класса: {n_confused} ({n_confused/max(loc,1):.1%} of localized)")
    print()
    print("=== confusion matrix (true -> reported) ===")
    for (t, r), c in sorted(confusion.items(), key=lambda x: -x[1]):
        print(f"  {t:20s} -> {r:20s}  x{c}")
    print()
    n_scored = flagged + silent
    print(f"=== physics-as-safety-net (confused instances с измерениями, n={n_scored}) ===")
    print(f"  поймано физикой (rel_err > {args.tau}): {flagged} ({flagged/max(n_scored,1):.1%})")
    print(f"  пропущено физикой: {silent} ({silent/max(n_scored,1):.1%})")
    print()
    print("=== пропущенные случаи (детально) ===")
    for x in silent_rows:
        print(f"  {x[0]:16s} {x[1]:18s} -> {x[2]:18s}  vol={x[3]:>7.1f}  "
              f"mass={x[4]:>6.1f}  exp={x[5]:>6.1f}  rel_err={x[6]:.3f}")


if __name__ == "__main__":
    main()
