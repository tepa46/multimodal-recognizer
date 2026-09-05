#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def rle_encode(mask: np.ndarray) -> dict:
    from pycocotools import mask as mu
    rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return {"size": [int(s) for s in rle["size"]], "counts": rle["counts"]}


def tight_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", default="datasets/real495/annotations/instances_boxes.json")
    ap.add_argument("--image-root", default="datasets/real495/images")
    ap.add_argument("--sam-checkpoint", default="checkpoints/sam_vit_h_4b8939.pth")
    ap.add_argument("--model-type", default="vit_h")
    ap.add_argument("--out", default="datasets/real495/sam_masks.json")
    ap.add_argument("--only-missing", action="store_true")
    ap.add_argument("--spotcheck", type=int, default=50)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    coco = json.loads(Path(args.coco).read_text())
    by_image = defaultdict(list)
    for a in coco["annotations"]:
        by_image[a["image_id"]].append(a)
    img_meta = {im["id"]: im for im in coco["images"]}

    out_path = Path(args.out)
    masks: dict = {}
    if args.only_missing and out_path.exists():
        masks = json.loads(out_path.read_text())
        print(f"[sam] resuming: {len(masks)} masks already present")

    todo = {iid: [a for a in anns if str(a["id"]) not in masks]
            for iid, anns in by_image.items()}
    todo = {k: v for k, v in todo.items() if v}
    n_ann = sum(len(v) for v in todo.values())
    print(f"[sam] {n_ann} annotations on {len(todo)} images to process")
    if not n_ann:
        print("[sam] nothing to do")
        return

    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to(args.device)
    predictor = SamPredictor(sam)

    image_root = Path(args.image_root)
    t0 = time.time()
    done = 0
    for i, (iid, anns) in enumerate(sorted(todo.items())):
        fn = img_meta[iid]["file_name"]
        with Image.open(image_root / fn) as im:
            arr = np.array(im.convert("RGB"))
        predictor.set_image(arr)
        for a in anns:
            x, y, w, h = a["bbox"]
            box = np.array([x, y, x + w, y + h])
            m, scores, _ = predictor.predict(box=box, multimask_output=True)
            k = int(np.argmax(scores))
            mask = m[k]
            tb = tight_bbox(mask)
            if tb is None:            # SAM produced an empty mask
                masks[str(a["id"])] = {"rle": None, "bbox": a["bbox"],
                                       "sam_score": 0.0}
                continue
            masks[str(a["id"])] = {"rle": rle_encode(mask), "bbox": tb,
                                   "sam_score": round(float(scores[k]), 4)}
            done += 1
        if (i + 1) % 25 == 0 or i == len(todo) - 1:
            print(f"  {i + 1}/{len(todo)} images, {done} masks "
                  f"({time.time() - t0:.0f}s)")

    out_path.write_text(json.dumps(masks))
    print(f"[sam] wrote {out_path} ({len(masks)} masks)")

    # ---- spot-check montage --------------------------------------------- #
    if args.spotcheck:
        from pycocotools import mask as mu
        anns_all = [a for a in coco["annotations"]
                    if masks.get(str(a["id"]), {}).get("rle")]
        random.seed(5)
        sample = random.sample(anns_all, min(args.spotcheck, len(anns_all)))
        cell, cols = 170, 10
        rows = (len(sample) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * cell, rows * cell), (20, 22, 26))
        dr = ImageDraw.Draw(sheet)
        for j, a in enumerate(sample):
            e = masks[str(a["id"])]
            fn = img_meta[a["image_id"]]["file_name"]
            with Image.open(image_root / fn) as im:
                im = im.convert("RGB")
                dm = mu.decode(dict(e["rle"])).astype(bool)
                arr = np.array(im, dtype=np.float32)
                arr[dm] = 0.55 * arr[dm] + 0.45 * np.array([60, 220, 90])
                x, y, w, h = e["bbox"]
                pad = 0.15
                x1 = int(max(0, x - w * pad)); y1 = int(max(0, y - h * pad))
                x2 = int(min(im.size[0], x + w * (1 + pad)))
                y2 = int(min(im.size[1], y + h * (1 + pad)))
                crop = Image.fromarray(arr.astype(np.uint8)).crop((x1, y1, x2, y2))
                crop.thumbnail((cell - 6, cell - 18))
            cx, cy = (j % cols) * cell, (j // cols) * cell
            sheet.paste(crop, (cx + 3, cy + 3))
            dr.text((cx + 4, cy + cell - 14),
                    f"{a['id']} s={e['sam_score']:.2f}", fill=(230, 230, 230))
        qa = Path(args.out).parent / "qa"
        qa.mkdir(exist_ok=True)
        sheet.save(qa / "spotcheck_sam.jpg", quality=88)
        print(f"[sam] spot-check -> {qa / 'spotcheck_sam.jpg'}")


if __name__ == "__main__":
    main()
