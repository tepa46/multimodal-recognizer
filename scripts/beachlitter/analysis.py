#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PACKAGING_IDS = set(range(9))          # category ids 0..8
OTHER_ID = 9


def class_agnostic_eval(coco_gt, pred_file: str) -> dict:
    from pycocotools.cocoeval import COCOeval
    dt = coco_gt.loadRes(pred_file)
    ev = COCOeval(coco_gt, dt, iouType="bbox")
    ev.params.useCats = 0
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {"AP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2],
            "AR100": ev.stats[8]}


def iou_matrix(dets: np.ndarray, gts: np.ndarray, iscrowd: list[int]):
    from pycocotools import mask as mu
    return mu.iou(dets.tolist(), gts.tolist(), iscrowd)


def routing_audit(coco_gt, pred_file: str, score_thr: float) -> dict:
    dets_by_img = defaultdict(list)
    for d in json.load(open(pred_file)):
        if d["score"] >= score_thr:
            dets_by_img[d["image_id"]].append(d)

    gt_by_img = defaultdict(lambda: {"eval": [], "crowd": []})
    for a in coco_gt.anns.values():
        gt_by_img[a["image_id"]]["crowd" if a["iscrowd"] else "eval"].append(a)

    per_class = defaultdict(lambda: {"n": 0, "matched": 0, "correct": 0})
    other_total = other_false_accept = 0
    det_bins = {"same_class": 0, "cross_class": 0, "other_gt": 0,
                "ignore_zone": 0, "background": 0}

    for img_id, groups in gt_by_img.items():
        dets = dets_by_img.get(img_id, [])
        evals, crowds = groups["eval"], groups["crowd"]
        dboxes = np.array([d["bbox"] for d in dets], dtype=np.float64)
        # --- GT side: best det for each evaluable GT ---
        if evals and dets:
            gboxes = np.array([a["bbox"] for a in evals], dtype=np.float64)
            ious = iou_matrix(dboxes, gboxes, [0] * len(evals))
        else:
            ious = np.zeros((len(dets), len(evals)))
        for gi, a in enumerate(evals):
            cid = a["category_id"]
            col = ious[:, gi] if len(dets) else np.array([])
            best = int(col.argmax()) if col.size and col.max() >= 0.5 else None
            if cid in PACKAGING_IDS:
                pc = per_class[cid]
                pc["n"] += 1
                if best is not None:
                    pc["matched"] += 1
                    if dets[best]["category_id"] == cid:
                        pc["correct"] += 1
            else:
                other_total += 1
                if best is not None and \
                        dets[best]["category_id"] in PACKAGING_IDS:
                    other_false_accept += 1
        # --- detection side (packaging dets only) ---
        pack_idx = [i for i, d in enumerate(dets)
                    if d["category_id"] in PACKAGING_IDS]
        if not pack_idx:
            continue
        crowd_ious = None
        if crowds:
            cboxes = np.array([a["bbox"] for a in crowds], dtype=np.float64)
            crowd_ious = iou_matrix(dboxes, cboxes, [1] * len(crowds))
        for i in pack_idx:
            col = ious[i] if len(evals) else np.array([])
            gi = int(col.argmax()) if col.size and col.max() >= 0.5 else None
            if gi is not None:
                g = evals[gi]
                if g["category_id"] == dets[i]["category_id"]:
                    det_bins["same_class"] += 1
                elif g["category_id"] in PACKAGING_IDS:
                    det_bins["cross_class"] += 1
                else:
                    det_bins["other_gt"] += 1
            elif crowd_ious is not None and crowd_ious[i].max() >= 0.5:
                det_bins["ignore_zone"] += 1
            else:
                det_bins["background"] += 1

    return {"per_class": {k: dict(v) for k, v in per_class.items()},
            "other_total": other_total,
            "other_false_accept": other_false_accept,
            "det_bins": det_bins}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--score-thr", type=float, default=0.3)
    ap.add_argument("--out", default=None, help="json summary path")
    args = ap.parse_args()

    from pycocotools.coco import COCO
    coco_gt = COCO(args.gt)
    cat_names = {c["id"]: c["name"] for c in coco_gt.dataset["categories"]}

    summary = {}
    for cfg in args.configs:
        pred = Path(args.pred_root) / cfg / "preds.bbox.json"
        if not pred.exists():
            print(f"[skip] {cfg}: {pred} missing")
            continue
        print(f"\n########## {cfg} ##########")
        print("--- class-agnostic (useCats=0, bbox) ---")
        agn = class_agnostic_eval(coco_gt, str(pred))
        aud = routing_audit(coco_gt, str(pred), args.score_thr)
        print(f"\n--- routing audit (IoU>=0.5, score>={args.score_thr}) ---")
        tot_n = tot_m = tot_c = 0
        for cid in sorted(aud["per_class"]):
            v = aud["per_class"][cid]
            tot_n += v["n"]; tot_m += v["matched"]; tot_c += v["correct"]
            print(f"  {cat_names[cid]:<18} n={v['n']:<5} "
                  f"recall={v['matched']/v['n']:.3f} "
                  f"routing_acc={v['correct']/max(v['matched'],1):.3f}")
        print(f"  {'ALL packaging':<18} n={tot_n:<5} "
              f"recall={tot_m/tot_n:.3f} "
              f"routing_acc={tot_c/max(tot_m,1):.3f}")
        far = aud["other_false_accept"] / max(aud["other_total"], 1)
        print(f"  verified other: n={aud['other_total']}, "
              f"false-accept={aud['other_false_accept']} ({far:.3f})")
        print(f"  det bins: {aud['det_bins']}")
        summary[cfg] = {"class_agnostic": agn, "audit": aud}

    if args.out:
        json.dump(summary, open(args.out, "w"), indent=1)
        print(f"\nsummary -> {args.out}")


if __name__ == "__main__":
    main()
