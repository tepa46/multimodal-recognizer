#!/usr/bin/env python3
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CLASSES = ["plastic_bottle", "glass_bottle", "glass_jar", "metal_can", "carton",
           "tetrapack", "paper", "plastic_bag", "plastic_container", "other"]
CONFIGS = [("htc_x101", "htc_x101"), ("htc_r50", "htc_r50"),
           ("mask_rcnn_r50", "mask_rcnn_r50"),
           ("cascade_mask_rcnn_r50", "cascade_mask_rcnn_r50"),
           ("faster_rcnn_r50", "faster_rcnn_r50")]

def parse(log: Path):
    """Return {'bbox': {...}, 'segm': {...}} of overall + per-class mAP."""
    if not log.exists():
        return None
    txt = log.read_text(errors="replace")
    out = {}
    m = re.findall(r"coco/(\w+): ([0-9.]+)", txt)
    if not m:
        return None
    scal = {k: float(v) for k, v in m}
    out["bbox_mAP"] = scal.get("bbox_mAP")
    out["segm_mAP"] = scal.get("segm_mAP")
    tables = re.findall(r"\| category\s+\|.*?\n((?:\|\s*\w+.*?\n)+)", txt, re.S)
    for name, tbl in zip(("bbox", "segm"), tables):
        d = {}
        for line in tbl.strip().splitlines():
            parts = [p.strip() for p in line.strip("|").split("|")]
            if len(parts) >= 2 and parts[0] in CLASSES:
                try:
                    d[parts[0]] = float(parts[1])
                except ValueError:
                    d[parts[0]] = float("nan")
        out[f"{name}_per_class"] = d
    return out

res = {}
for cfg33, cfg32 in CONFIGS:
    res[cfg33] = {
        "v3.3": parse(ROOT / f"experiments/work_dirs/_v33_logs/{cfg33}_test.log"),
        "v3.2": parse(ROOT / f"experiments/work_dirs/_v32_logs/{cfg32}_test.log"),
    }

print("## Overall (test split)\n")
print("| config | v3.2 bbox | v3.3 bbox | Δ | v3.2 segm | v3.3 segm | Δ |")
print("|---|---:|---:|---:|---:|---:|---:|")
for cfg, _ in CONFIGS:
    a, b = res[cfg]["v3.2"], res[cfg]["v3.3"]
    if not b:
        print(f"| {cfg} | — | (pending) | | | | |"); continue
    def f(x): return "—" if x is None else f"{x:.3f}"
    def d(x, y): return "—" if (x is None or y is None) else f"{y-x:+.3f}"
    print(f"| {cfg} | {f(a and a['bbox_mAP'])} | {f(b['bbox_mAP'])} | "
          f"{d(a and a['bbox_mAP'], b['bbox_mAP'])} | {f(a and a['segm_mAP'])} | "
          f"{f(b['segm_mAP'])} | {d(a and a['segm_mAP'], b['segm_mAP'])} |")

print("\n## Per-class bbox mAP, htc_x101\n")
a = res["htc_x101"]["v3.2"]; b = res["htc_x101"]["v3.3"]
if a and b:
    print("| class | v3.2 | v3.3 | Δ |")
    print("|---|---:|---:|---:|")
    for c in CLASSES:
        x = a["bbox_per_class"].get(c); y = b["bbox_per_class"].get(c)
        print(f"| {c} | {x:.3f} | {y:.3f} | {y-x:+.3f} |")

out = ROOT / "experiments/work_dirs/_v33_logs/bench_v33_vs_v32.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(res, indent=2))
print(f"\nwrote {out}")
