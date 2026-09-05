#!/usr/bin/env python3
"""Extract per-class bbox/segm AP tables from the v3.3 detector test logs."""
import re
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CFGS = ["htc_r50", "htc_x101", "mask_rcnn_r50",
        "faster_rcnn_r50", "cascade_mask_rcnn_r50"]
CLASSES = ["plastic_bottle", "glass_bottle", "glass_jar", "metal_can", "carton",
           "tetrapack", "paper", "plastic_bag", "plastic_container", "other"]

ROW = re.compile(r"^\|\s*([a-z_]+)\s*\|\s*([0-9.]+|nan)\s*\|")

out = {}
for cfg in CFGS:
    path = "experiments/work_dirs/_v33_logs/%s_test.log" % cfg
    if not os.path.exists(path):
        print("MISSING", path)
        continue
    tables, cur = [], {}
    for line in open(path):
        m = ROW.match(line.strip())
        if m and m.group(1) in CLASSES:
            name, val = m.group(1), m.group(2)
            if name in cur:            # new table started
                tables.append(cur)
                cur = {}
            cur[name] = None if val == "nan" else float(val)
        elif cur and not line.strip().startswith("|") and len(cur) >= len(CLASSES):
            tables.append(cur)
            cur = {}
    if cur:
        tables.append(cur)
    out[cfg] = {"bbox": tables[0] if len(tables) > 0 else {},
                "segm": tables[1] if len(tables) > 1 else {}}

dst = ROOT / "experiments/work_dirs/_v33_logs/perclass_v33.json"
dst.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, dst.open("w"), indent=1)

for kind in ("bbox", "segm"):
    print("\n=== per-class %s AP (v3.3, test split) ===" % kind)
    print("%-19s" % "class" + "".join("%16s" % c for c in CFGS))
    print("-" * (19 + 16 * len(CFGS)))
    for cls in CLASSES:
        cells = []
        for cfg in CFGS:
            v = out.get(cfg, {}).get(kind, {}).get(cls)
            cells.append("%16s" % ("%.3f" % v if v is not None else "--"))
        print("%-19s" % cls + "".join(cells))
