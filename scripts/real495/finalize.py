#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["plastic_bottle", "glass_bottle", "glass_jar", "metal_can",
           "carton", "tetrapack", "paper", "plastic_bag",
           "plastic_container", "other"]
COLORS = {"plastic_bottle": (77, 163, 255), "glass_bottle": (57, 194, 109),
          "glass_jar": (32, 197, 197), "metal_can": (232, 185, 62),
          "carton": (192, 120, 64), "tetrapack": (179, 77, 255),
          "paper": (224, 224, 224), "plastic_bag": (255, 110, 169),
          "plastic_container": (125, 214, 73), "other": (154, 154, 154)}


def iou_xywh(a, b) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2 = min(ax1 + aw, bx1 + bw); iy2 = min(ay1 + ah, by1 + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter) if inter > 0 else 0.0


def rect_polygon(bbox):
    x, y, w, h = bbox
    return [[x, y, x + w, y, x + w, y + h, x, y + h]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="datasets/real495/annotations/instances_boxes.json")
    ap.add_argument("--masks", default="datasets/real495/sam_masks.json")
    ap.add_argument("--image-root", default="datasets/real495/images")
    ap.add_argument("--out", default="datasets/real495/annotations/instances_all.json")
    ap.add_argument("--iou-guard", type=float, default=0.5)
    ap.add_argument("--score-guard", type=float, default=0.7)
    ap.add_argument("--sheets", action="store_true", default=True)
    args = ap.parse_args()

    from pycocotools import mask as mu
    from pycocotools.coco import COCO

    coco = json.loads(Path(args.coco).read_text())
    masks = json.loads(Path(args.masks).read_text())
    root = Path(args.coco).parent.parent
    qa = root / "qa"
    qa.mkdir(exist_ok=True)

    flags = []
    n_rle = n_rect = 0
    for a in coco["annotations"]:
        e = masks.get(str(a["id"]))
        ok = bool(e and e.get("rle"))
        if ok:
            iou = iou_xywh(e["bbox"], a["bbox"])
            ok = iou >= args.iou_guard and e["sam_score"] >= args.score_guard
        if ok:
            a["segmentation"] = e["rle"]
            a["bbox"] = [round(v, 1) for v in e["bbox"]]
            a["area"] = float(mu.area(dict(e["rle"])))
            a["sam_score"] = e["sam_score"]
            n_rle += 1
        else:
            a["segmentation"] = rect_polygon(a["bbox"])
            n_rect += 1
            flags.append({"ann_id": a["id"],
                          "file_name": next(im["file_name"] for im in
                                            coco["images"]
                                            if im["id"] == a["image_id"]),
                          "reason": ("no_mask" if not (e and e.get("rle"))
                                     else f"iou={iou:.2f} "
                                          f"score={e['sam_score']:.2f}")})

    coco["info"]["description"] = ("real495 — real-sensor fraud evaluation "
                                   "set (author photos), final")
    out = Path(args.out)
    out.write_text(json.dumps(coco, ensure_ascii=False))
    (qa / "sam_flags.json").write_text(json.dumps(flags, indent=1,
                                                  ensure_ascii=False))
    print(f"[finalize] wrote {out}: {n_rle} SAM RLE + {n_rect} rect-polygon "
          f"fallback; flags: {len(flags)} -> qa/sam_flags.json")

    # ---- validation ------------------------------------------------------ #
    c = COCO(str(out))
    bad = 0
    for aid, a in c.anns.items():
        im = c.imgs[a["image_id"]]
        x, y, w, h = a["bbox"]
        if not (0 <= x and 0 <= y and x + w <= im["width"] + 1
                and y + h <= im["height"] + 1 and w > 0 and h > 0):
            print("  BAD bbox:", aid, a["bbox"]); bad += 1
        if a["area"] <= 0:
            print("  BAD area:", aid); bad += 1
        if not (0 <= a["category_id"] <= 9):
            print("  BAD category:", aid); bad += 1
    per_cls = {c.cats[k]["name"]: len(c.getAnnIds(catIds=[k]))
               for k in sorted(c.cats)}
    print(f"[validate] images={len(c.imgs)} anns={len(c.anns)} bad={bad}")
    print("[validate] per-class:", {k: v for k, v in per_cls.items() if v})
    conds = {}
    for a in c.anns.values():
        cd = a.get("attributes", {}).get("condition", "?")
        conds[cd] = conds.get(cd, 0) + 1
    print("[validate] condition:", conds)

    # ---- contact sheets --------------------------------------------------- #
    if not args.sheets:
        return
    image_root = Path(args.image_root)
    img_ids = sorted(c.imgs)
    flagged_imgs = {f["file_name"] for f in flags}
    img_ids.sort(key=lambda i: (c.imgs[i]["file_name"] not in flagged_imgs,
                                c.imgs[i]["file_name"]))
    cols, rows_pp, thumb_h = 3, 4, 760
    per_page = cols * rows_pp
    try:
        from PIL import ImageFont
        font = ImageFont.load_default()
    except Exception:
        font = None
    for p in range(0, len(img_ids), per_page):
        page_ids = img_ids[p:p + per_page]
        cells = []
        for iid in page_ids:
            meta = c.imgs[iid]
            with Image.open(image_root / meta["file_name"]) as im:
                im = im.convert("RGB")
                sc = thumb_h / im.size[1]
                thumb = im.resize((int(im.size[0] * sc), thumb_h))
            arr = np.array(thumb, dtype=np.float32)
            dr_im = None
            for a in c.imgToAnns[iid]:
                col = np.array(COLORS[c.cats[a["category_id"]]["name"]],
                               dtype=np.float32)
                seg = a["segmentation"]
                if isinstance(seg, dict):
                    m = mu.decode(seg).astype(bool)
                    m = np.array(Image.fromarray(m).resize(thumb.size,
                                                           Image.NEAREST))
                    arr[m.astype(bool)] = 0.6 * arr[m.astype(bool)] + 0.4 * col
            thumb = Image.fromarray(arr.astype(np.uint8))
            dr_im = ImageDraw.Draw(thumb)
            for a in c.imgToAnns[iid]:
                name = c.cats[a["category_id"]]["name"]
                col = COLORS[name]
                x, y, w, h = [v * sc for v in a["bbox"]]
                dr_im.rectangle([x, y, x + w, y + h], outline=col, width=3)
                t = name
                at = a.get("attributes", {})
                if at.get("unsure"):
                    t += " ?"
                if at.get("condition", "normal") != "normal":
                    t += f" [{at['condition']}]"
                dr_im.rectangle([x, y - 14, x + 7 * len(t), y], fill=col)
                dr_im.text((x + 2, y - 13), t, fill=(0, 0, 0), font=font)
            tag = meta["file_name"] + (
                "  ⚑" if meta["file_name"] in flagged_imgs else "")
            dr_im.rectangle([0, 0, 7 * len(tag) + 6, 14], fill=(0, 0, 0))
            dr_im.text((3, 1), tag, fill=(255, 255, 80), font=font)
            cells.append(thumb)
        cw = max(t.size[0] for t in cells) + 8
        ch = thumb_h + 8
        sheet = Image.new("RGB", (cols * cw, rows_pp * ch), (18, 20, 24))
        for j, t in enumerate(cells):
            sheet.paste(t, ((j % cols) * cw + 4, (j // cols) * ch + 4))
        fp = qa / f"contact_{p // per_page + 1:03d}.jpg"
        sheet.save(fp, quality=85)
    print(f"[finalize] contact sheets -> {qa}/contact_*.jpg "
          f"({(len(img_ids) + per_page - 1) // per_page} pages, "
          f"flagged first)")


if __name__ == "__main__":
    main()
