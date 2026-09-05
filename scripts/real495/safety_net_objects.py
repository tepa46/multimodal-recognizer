#!/usr/bin/env python
import argparse
import csv
import json
from collections import Counter, defaultdict
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


def iou_xywh(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter) if inter > 0 else 0.0


ap = argparse.ArgumentParser()
ap.add_argument("--raw-detections", required=True)
ap.add_argument("--coco", default="datasets/real495/annotations/instances_all.json")
ap.add_argument("--measurements", default="datasets/real495/measurements.csv")
ap.add_argument("--score-thr", type=float, default=0.3)
ap.add_argument("--iou-thr", type=float, default=0.5)
ap.add_argument("--tau", type=float, default=0.25)
ap.add_argument("--label", default="")
args = ap.parse_args()

raw = json.loads(Path(args.raw_detections).read_text())
coco = json.loads(Path(args.coco).read_text())
meas = {(r["filename"], int(r["pos"])): r
        for r in csv.DictReader(open(args.measurements))}
id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
images = {im["id"]: im for im in coco["images"]}


def objkey(m, true_name):
    return (true_name, m["height_cm"], m["width_cm"], m["depth_cm"], m["mass_g"])


inst_flag, inst_silent = 0, 0
obj_flag, obj_silent = defaultdict(int), defaultdict(int)
obj_pairs = defaultdict(Counter)
all_normal_objs = set()

for ann in coco["annotations"]:
    attrs = ann.get("attributes", {})
    if attrs.get("condition") != "normal":
        continue
    fn = images[ann["image_id"]]["file_name"]
    true_name = id_to_name[ann["category_id"]]
    m0 = meas.get((fn, attrs.get("pos")))
    if m0 and m0.get("mass_g") and m0.get("height_cm"):
        all_normal_objs.add(objkey(m0, true_name))
    dets = [d for d in raw["detections"].get(fn, []) if d["score"] >= args.score_thr]
    best_iou, best_det = 0.0, None
    for d in dets:
        v = iou_xywh(ann["bbox"], d["bbox"])
        if v > best_iou:
            best_iou, best_det = v, d
    if best_det is None or best_iou < args.iou_thr:
        continue
    rep = best_det["det_class"]
    if rep == true_name:
        continue
    m = meas.get((fn, attrs.get("pos")))
    if not m or not m.get("mass_g") or not m.get("height_cm"):
        continue
    vol = float(m["height_cm"]) * float(m["width_cm"]) * float(m["depth_cm"])
    exp = vol * DENSITY[rep]
    rel = abs(float(m["mass_g"]) - exp) / max(exp, 1e-3)
    k = objkey(m, true_name)
    obj_pairs[k][(true_name, rep)] += 1
    if rel > args.tau:
        inst_flag += 1
        obj_flag[k] += 1
    else:
        inst_silent += 1
        obj_silent[k] += 1

n_inst = inst_flag + inst_silent
objs = set(obj_flag) | set(obj_silent)
# an object counts as "caught" if physics flags it in every photo it is confused in
obj_always = {k for k in objs if obj_silent.get(k, 0) == 0}
obj_never = {k for k in objs if obj_flag.get(k, 0) == 0}
obj_mixed = objs - obj_always - obj_never

print("### %s" % (args.label or args.raw_detections))
print("  instances:  n=%d  caught=%d (%.1f%%)  missed=%d (%.1f%%)"
      % (n_inst, inst_flag, 100 * inst_flag / max(n_inst, 1),
         inst_silent, 100 * inst_silent / max(n_inst, 1)))
print("  DISTINCT physical objects behind those instances: %d" % len(objs))
print("    caught in every photo : %d (%.1f%%)" % (len(obj_always), 100 * len(obj_always) / max(len(objs), 1)))
print("    missed in every photo : %d (%.1f%%)" % (len(obj_never), 100 * len(obj_never) / max(len(objs), 1)))
print("    mixed                 : %d" % len(obj_mixed))
print("  (distinct normal-condition objects in the whole set: %d)" % len(all_normal_objs))
print("  repeats per confused object: %s"
      % dict(sorted(Counter(obj_flag.get(k, 0) + obj_silent.get(k, 0) for k in objs).items())))
print("  --- objects never caught ---")
for k in sorted(obj_never):
    pair = obj_pairs[k].most_common(1)[0]
    print("    %-16s h=%-6s w=%-6s d=%-6s mass=%-7s  %s->%s  x%d photos"
          % (k[0], k[1], k[2], k[3], k[4], pair[0][0], pair[0][1],
             obj_flag.get(k, 0) + obj_silent.get(k, 0)))
print()
