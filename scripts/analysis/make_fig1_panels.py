"""Raster panels for Fig. 1 of the manuscript from one real495 frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pycocotools import mask as mask_utils


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path, default=Path("datasets/real495/annotations/instances_all.json"))
    ap.add_argument("--images", type=Path, default=Path("datasets/real495/images"))
    ap.add_argument("--image", default="IMG_2118.jpg")
    ap.add_argument("--out", type=Path, default=Path("../manuscript/figures"))
    ap.add_argument("--width", type=int, default=480, help="output width in px")
    ap.add_argument("--aspect", type=float, default=0.75, help="crop width / height")
    ap.add_argument("--margin", type=float, default=0.12, help="margin around the box, fraction of box height")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cut-height", type=int, default=600, help="height of the upright cut-out in px")
    args = ap.parse_args()

    coco = json.load(open(args.coco))
    img = next(i for i in coco["images"] if i["file_name"] == args.image)
    anns = [a for a in coco["annotations"] if a["image_id"] == img["id"]]
    ann = max(anns, key=lambda a: a["area"])
    cat = next(c["name"] for c in coco["categories"] if c["id"] == ann["category_id"])

    im = Image.open(args.images / args.image)
    if im.size != (img["width"], img["height"]):
        im = ImageOps.exif_transpose(im)
    assert im.size == (img["width"], img["height"]), (im.size, img["width"], img["height"])
    im = im.convert("RGB")
    m = mask_utils.decode(ann["segmentation"]).astype(bool)
    assert m.shape == (img["height"], img["width"])

    x, y, w, h = ann["bbox"]
    pad = args.margin * h
    ch = h + 2 * pad
    cw = max(w + 2 * pad, args.aspect * ch)
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
    x1, y1 = int(round(cx + cw / 2)), int(round(cy + ch / 2))
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(im.width, x1), min(im.height, y1)
    crop = im.crop((x0, y0, x1, y1))
    mcrop = m[y0:y1, x0:x1]
    scale = args.width / crop.width
    size = (args.width, int(round(crop.height * scale)))
    crop = crop.resize(size, Image.LANCZOS)
    mres = np.array(Image.fromarray(mcrop.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127

    args.out.mkdir(parents=True, exist_ok=True)
    crop.save(args.out / "fig1_photo.png", optimize=True)

    overlay = np.array(crop).astype(np.float32)
    tint = np.array([40, 170, 90], dtype=np.float32)
    overlay[mres] = 0.45 * overlay[mres] + 0.55 * tint
    eroded = np.array(Image.fromarray(mres.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(5))) > 127
    edge = mres & ~eroded
    overlay[edge] = np.array([255, 255, 255], dtype=np.float32)
    Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(args.out / "fig1_mask.png", optimize=True)

    rng = np.random.default_rng(args.seed)
    cloud = Image.new("RGB", size, (255, 255, 255))
    dr = ImageDraw.Draw(cloud)
    ys, xs = np.nonzero(mres)
    n = min(len(xs), 2600)
    idx = rng.choice(len(xs), size=n, replace=False)
    xmin, xmax = xs.min(), xs.max()
    for px, py in zip(xs[idx], ys[idx]):
        t = (px - xmin) / max(1, xmax - xmin)
        g = int(60 + 120 * t)
        dr.ellipse((px - 1.6, py - 1.6, px + 1.6, py + 1.6), fill=(40, 70 + g // 2, 130 + g // 3))
    cloud.save(args.out / "fig1_cloud.png", optimize=True)

    ys_f, xs_f = np.nonzero(m)
    pts = np.stack([xs_f - xs_f.mean(), ys_f - ys_f.mean()], axis=1).astype(np.float64)
    evals, evecs = np.linalg.eigh(pts.T @ pts / len(pts))
    axis = evecs[:, int(np.argmax(evals))]
    if axis[1] < 0:
        axis = -axis
    tilt = float(np.degrees(np.arctan2(axis[0], axis[1])))
    rgba = im.convert("RGBA")
    alpha = Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.GaussianBlur(1.2))
    rgba.putalpha(alpha)
    rotated = rgba.rotate(-tilt, resample=Image.BICUBIC, expand=True)
    box = rotated.getchannel("A").point(lambda a: 255 if a > 24 else 0).getbbox()
    cut = rotated.crop(box)
    cut = cut.resize((int(round(cut.width * args.cut_height / cut.height)), args.cut_height), Image.LANCZOS)
    cut.save(args.out / "fig1_bottle.png", optimize=True)
    print(f"cut-out rotated by {tilt:.1f} deg -> {cut.width}x{cut.height} px, aspect {cut.width / cut.height:.3f}")

    bx0, by0 = (x - x0) * scale / size[0], (y - y0) * scale / size[1]
    bx1, by1 = (x + w - x0) * scale / size[0], (y + h - y0) * scale / size[1]
    print(f"image {args.image}  class {cat}  annotations on frame {len(anns)}")
    print(f"crop {x0},{y0}-{x1},{y1} -> {size[0]}x{size[1]} px, aspect {size[0]/size[1]:.3f}")
    print(f"bbox in crop (fractions, origin top-left): x {bx0:.3f}-{bx1:.3f}, y {by0:.3f}-{by1:.3f}")


if __name__ == "__main__":
    main()
