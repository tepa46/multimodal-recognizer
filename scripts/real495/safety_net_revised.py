"""Physics as a safety net on real detector confusions (revised protocol)."""
from __future__ import annotations

import argparse, csv, json, statistics as st
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import yaml


def iou(a, b):
    ix = max(0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0])); iy = max(0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1])); I = ix*iy
    return I / (a[2]*a[3] + b[2]*b[3] - I) if I > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-detections", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--coco", default="datasets/real495/annotations/instances_all.json")
    ap.add_argument("--measurements", default="datasets/real495/measurements_specimens.csv")
    ap.add_argument("--calibration", default="configs/calibration_packphys.yaml")
    ap.add_argument("--score-thr", type=float, default=0.3); ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rho = yaml.safe_load(open(a.calibration))["densities"]
    raw = json.load(open(a.raw_detections)); coco = json.load(open(a.coco))
    meas = {(r["filename"], int(r["pos"])): r for r in csv.DictReader(open(a.measurements))}
    id2n = {c["id"]: c["name"] for c in coco["categories"]}; ims = {i["id"]: i for i in coco["images"]}
    n_normal = n_loc = 0; conf = Counter(); items = []
    for ann in coco["annotations"]:
        at = ann["attributes"]
        if at["condition"] != "normal": continue
        n_normal += 1; fn = ims[ann["image_id"]]["file_name"]; t = id2n[ann["category_id"]]
        dets = [d for d in raw["detections"].get(fn, []) if d["score"] >= a.score_thr]
        best = max(dets, key=lambda d: iou(ann["bbox"], d["bbox"]), default=None)
        if best is None or iou(ann["bbox"], best["bbox"]) < a.iou_thr: continue
        n_loc += 1; rep = best["det_class"]; m = meas[(fn, at["pos"])]
        V = float(m["height_cm"]) * float(m["width_cm"]) * float(m["depth_cm"]); r = abs(float(m["mass_g"]) / V - rho[rep]) / rho[rep]
        items.append((rep != t, r, m["specimen_id"], t, rep))
        if rep != t: conf[(t, rep)] += 1
    pos = [r for c, r, *_ in items if c]; neg = [r for c, r, *_ in items if not c]
    auc = sum((p > n) + 0.5 * (p == n) for p, n in product(pos, neg)) / (len(pos) * len(neg))
    res = {"label": a.label, "n_normal": n_normal, "n_localised": n_loc, "n_correct": len(neg), "n_confused": len(pos),
           "auc_confused_vs_correct": auc,
           "at_tau_0.25": {"confusions_flagged": sum(p > 0.25 for p in pos), "correct_rejected": sum(n > 0.25 for n in neg)}}
    ns = sorted(neg)
    for f in (0.05, 0.10, 0.20):
        thr = ns[min(int((1 - f) * len(ns)), len(ns) - 1)]
        res[f"at_false_reject_{int(f*100)}"] = {"tau": thr, "confusions_flagged": sum(p > thr for p in pos), "share": sum(p > thr for p in pos) / len(pos)}
    spec_conf = defaultdict(list)
    for c, r, s, t, rep in items:
        if c: spec_conf[s].append(r)
    res["confused_specimens"] = len(spec_conf)
    res["specimens_flagged_at_0.25"] = sum(all(r > 0.25 for r in v) for v in spec_conf.values())
    res["specimens_never_flagged_at_0.25"] = sum(all(r <= 0.25 for r in v) for v in spec_conf.values())
    res["confusion_pairs_top"] = [(f"{t}->{rep}", n) for (t, rep), n in conf.most_common(6)]
    print(f"{a.label}: normal {n_normal}, localised {n_loc}, correct {len(neg)}, confused {len(pos)} ({len(spec_conf)} specimens)")
    print(f"  AUC(confused vs correct) {auc:.3f}; at tau 0.25: {res['at_tau_0.25']['confusions_flagged']}/{len(pos)} confusions flagged, {res['at_tau_0.25']['correct_rejected']}/{len(neg)} correct items rejected")
    for f in (5, 10, 20): print(f"  at false-reject {f}%: tau {res[f'at_false_reject_{f}']['tau']:.2f}, confusions flagged {res[f'at_false_reject_{f}']['confusions_flagged']}/{len(pos)} = {res[f'at_false_reject_{f}']['share']:.3f}")
    if a.out: Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
