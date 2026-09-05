#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from candidates import PACKAGING, write_pages  # noqa: E402

CROP_PAD = 0.15
CROP_MAX = 256
CTX_MAX = 640
OVERLAY = (60, 220, 90)      # green mask fill for galleries / spot-check


def load_groups(verified: Path) -> tuple[dict[int, str], list[int]]:
    """Return ({ann_id: class} for accepted packaging, [ann_id] for rescue)."""
    accepted, rescue = {}, []
    with open(verified) as f:
        for row in csv.DictReader(f):
            aid = int(row["ann_id"])
            if row["final_class"] in PACKAGING:
                accepted[aid] = row["final_class"]
            elif row["final_class"] == "ignore_merged":
                rescue.append(aid)
    return accepted, rescue


def rle_encode(mask: np.ndarray) -> dict:
    from pycocotools import mask as mu
    r = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode("ascii")
    return r


def tight_bbox(mask: np.ndarray) -> list[float] | None:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if not len(rows) or not len(cols):
        return None
    x0, x1 = int(cols[0]), int(cols[-1])
    y0, y1 = int(rows[0]), int(rows[-1])
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def overlay_crop(im: Image.Image, mask: np.ndarray, bbox, out_p: Path,
                 max_side: int = CROP_MAX) -> None:
    x, y, w, h = bbox
    dx, dy = w * CROP_PAD, h * CROP_PAD
    box = (max(0, int(x - dx)), max(0, int(y - dy)),
           min(im.width, int(x + w + dx)), min(im.height, int(y + h + dy)))
    arr = np.array(im).copy()
    m = mask.astype(bool)
    arr[m] = (0.55 * arr[m] + 0.45 * np.array(OVERLAY)).astype(np.uint8)
    crop = Image.fromarray(arr).crop(box)
    crop.thumbnail((max_side, max_side))
    crop.save(out_p, quality=88)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verified", required=True)
    ap.add_argument("--coco", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--sam-checkpoint", required=True)
    ap.add_argument("--model-type", default="vit_h")
    ap.add_argument("--zeroshot",
                    default="experiments/beachlitter_relabel/beach_zeroshot40.json",
                    help="for SigLIP suggestions in the rescue gallery")
    ap.add_argument("--out", default="experiments/beachlitter_relabel/")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out)
    accepted, rescue = load_groups(Path(args.verified))
    print(f"accepted packaging: {len(accepted)}, rescue candidates: {len(rescue)}")

    coco = json.load(open(args.coco))
    imgs = {i["id"]: i for i in coco["images"]}
    anns = {a["id"]: a for a in coco["annotations"]}
    zs = {d["ann_id"]: d for d in json.load(open(args.zeroshot))}

    todo_ids = list(accepted) + rescue
    by_image: dict[int, list[int]] = defaultdict(list)
    for aid in todo_ids:
        by_image[anns[aid]["image_id"]].append(aid)
    print(f"images to process: {len(by_image)}")

    # --- SAM ---
    import torch
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam = sam.to(args.device)
    predictor = SamPredictor(sam)

    image_root = Path(args.image_root)
    rescue_dir = out / "qa_html_rescue"
    (rescue_dir / "crops").mkdir(parents=True, exist_ok=True)

    results: dict[int, dict] = {}
    spot_pool: list[tuple[int, Path]] = []
    n_img = 0
    for image_id, aids in by_image.items():
        info = imgs[image_id]
        try:
            im = Image.open(image_root / info["file_name"]).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] {info['file_name']}: {exc}")
            continue
        predictor.set_image(np.array(im))
        for aid in aids:
            a = anns[aid]
            x, y, w, h = a["bbox"]
            box = np.array([x, y, x + w, y + h])
            with torch.no_grad():
                masks, scores, _ = predictor.predict(
                    box=box, multimask_output=True)
            k = int(np.argmax(scores))
            mask, score = masks[k], float(scores[k])
            bb = tight_bbox(mask)
            if bb is None:
                continue
            group = "refine" if aid in accepted else "rescue"
            results[aid] = {"rle": rle_encode(mask), "bbox": bb,
                            "sam_score": score, "group": group}
            # gallery / spot-check assets
            if group == "rescue":
                overlay_crop(im, mask, a["bbox"],
                             rescue_dir / "crops" / f"{aid}.jpg")
                ctx = im.copy()
                dr = ImageDraw.Draw(ctx)
                lw = max(3, int(max(ctx.size) / 300))
                dr.rectangle((x, y, x + w, y + h),
                             outline=(255, 40, 40), width=lw)
                ctx.thumbnail((CTX_MAX, CTX_MAX))
                ctx.save(rescue_dir / "crops" / f"ctx_{aid}.jpg", quality=82)
            else:
                spot_pool.append((aid, image_root / info["file_name"]))
        n_img += 1
        if n_img % 50 == 0:
            print(f"  {n_img}/{len(by_image)} images")

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "refined_masks.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f)
    n_ref = sum(1 for r in results.values() if r["group"] == "refine")
    n_res = len(results) - n_ref
    print(f"refined: {n_ref}, rescue-masked: {n_res} "
          f"-> {out / 'refined_masks.json'}")

    # --- rescue gallery (reuse main gallery UI) ---
    from pycocotools import mask as mu
    queue = []
    for aid in rescue:
        if aid not in results:
            continue
        d = zs.get(aid)
        a = anns[aid]
        side = max(a["bbox"][2], a["bbox"][3])
        psim = max(d["sims"][c] for c in PACKAGING) if d else 0.0
        sugg = (max(PACKAGING, key=lambda c: d["sims"][c]) if d
                else "plastic_bottle")
        queue.append({"ann_id": aid, "fn": imgs[a["image_id"]]["file_name"],
                      "bbox": a["bbox"], "side": side, "psim": psim,
                      "pack_sugg": sugg, "pred": d["pred"] if d else "?"})
    queue.sort(key=lambda q: -q["psim"])
    write_pages(queue, rescue_dir, "rescue_", "SAM-спасение ignore_merged")
    print(f"rescue gallery: {rescue_dir}/rescue_index.html "
          f"(зелёная заливка = маска SAM; принимай класс, только если SAM "
          f"выделил доминирующий объект)")

    # --- spot-check montage of 50 accepted refinements ---
    random.seed(5)
    sample = random.sample(spot_pool, min(50, len(spot_pool)))
    CELL, COLS = 170, 10
    rows_n = (len(sample) + COLS - 1) // COLS
    sheet = Image.new("RGB", (COLS * CELL, rows_n * CELL), (25, 25, 25))
    dr = ImageDraw.Draw(sheet)
    for i, (aid, img_p) in enumerate(sample):
        im = Image.open(img_p).convert("RGB")
        r = results[aid]
        mask = mu.decode({"size": r["rle"]["size"],
                          "counts": r["rle"]["counts"].encode("ascii")})
        tmp = rescue_dir / "crops" / f"_spot_{aid}.jpg"
        overlay_crop(im, mask, anns[aid]["bbox"], tmp, max_side=CELL - 4)
        crop = Image.open(tmp)
        tmp.unlink()
        cx, cy = (i % COLS) * CELL, (i // COLS) * CELL
        sheet.paste(crop, (cx + 2, cy + 2))
        dr.text((cx + 4, cy + 2), f"{results[aid]['sam_score']:.2f}",
                fill=(255, 255, 0))
    sheet.save(out / "spotcheck_refine.jpg", quality=88)
    print(f"spot-check -> {out / 'spotcheck_refine.jpg'}")


if __name__ == "__main__":
    main()
