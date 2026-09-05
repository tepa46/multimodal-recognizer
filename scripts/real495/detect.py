#!/usr/bin/env python
from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path

CLASS_NAMES = ["plastic_bottle", "glass_bottle", "glass_jar", "metal_can",
               "carton", "tetrapack", "paper", "plastic_bag",
               "plastic_container", "other"]


def iou_xywh(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / (aw * ah + bw * bh - inter)


def dedup(dets: list[dict], iou_thr: float = 0.85) -> list[dict]:
    kept: list[dict] = []
    for d in sorted(dets, key=lambda x: -x["score"]):
        merged = False
        for k in kept:
            if iou_xywh(d["bbox"], k["bbox"]) >= iou_thr:
                alt = k.setdefault("alt_classes", [])
                if d["det_class"] != k["det_class"] and \
                        d["det_class"] not in [a[0] for a in alt]:
                    alt.append([d["det_class"], round(d["score"], 4)])
                merged = True
                break
        if not merged:
            kept.append(dict(d))
    return kept


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/detectors/htc_x101.py")
    ap.add_argument("--ckpt", default="experiments/work_dirs/htc_x101_v33/epoch_17.pth")
    ap.add_argument("--image-dir", default="datasets/real495/images")
    ap.add_argument("--score-floor", type=float, default=0.05)
    ap.add_argument("--out", default="datasets/real495/raw_detections.json")
    ap.add_argument("--limit", type=int, default=0, help="smoke run: first N images")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    from PIL import Image
    from mmdet.apis import init_detector, inference_detector

    image_dir = Path(args.image_dir)
    files = sorted(image_dir.glob("*.jpg"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no .jpg files in {image_dir}")

    print(f"[detect] {len(files)} images, floor={args.score_floor}")
    model = init_detector(args.config, args.ckpt, device=args.device)

    images_meta = []
    detections: dict[str, list[dict]] = {}
    t0 = time.time()
    n_total = 0
    for i, fp in enumerate(files):
        with Image.open(fp) as im:
            W, H = im.size
        images_meta.append({"file_name": fp.name, "width": W, "height": H})

        result = inference_detector(model, str(fp))
        inst = result.pred_instances
        keep = inst.scores >= args.score_floor
        bboxes = inst.bboxes[keep].cpu().numpy()   # xyxy
        scores = inst.scores[keep].cpu().numpy()
        labels = inst.labels[keep].cpu().numpy()

        dets = []
        for j in range(len(scores)):
            x1, y1, x2, y2 = (float(v) for v in bboxes[j])
            dets.append({
                "det_id": j,
                "bbox": [round(x1, 1), round(y1, 1),
                         round(x2 - x1, 1), round(y2 - y1, 1)],  # xywh
                "score": round(float(scores[j]), 4),
                "det_class": CLASS_NAMES[int(labels[j])],
            })
        detections[fp.name] = dets
        n_total += len(dets)
        if (i + 1) % 25 == 0 or i == len(files) - 1:
            dt = time.time() - t0
            print(f"  {i + 1}/{len(files)}  dets={n_total}  ({dt:.0f}s)")

    out = {
        "meta": {
            "ckpt": args.ckpt,
            "score_floor": args.score_floor,
            "classes": CLASS_NAMES,
            "date": datetime.date.today().isoformat(),
            "n_images": len(files),
        },
        "images": images_meta,
        "detections": detections,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(f"[detect] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # quick per-image summary at the UI-default threshold
    thr = 0.3
    per_img = [sum(1 for d in dedup(v) if d["score"] >= thr)
               for v in detections.values()]
    zero = sum(1 for n in per_img if n == 0)
    print(f"[detect] deduped dets/image @score>={thr}: "
          f"min={min(per_img)} med={sorted(per_img)[len(per_img)//2]} "
          f"max={max(per_img)}; images with 0 dets: {zero}/{len(per_img)}")


if __name__ == "__main__":
    main()
