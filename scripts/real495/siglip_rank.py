#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from detect import dedup  # noqa: E402

CLASS_NAMES = ["plastic_bottle", "glass_bottle", "glass_jar", "metal_can",
               "carton", "tetrapack", "paper", "plastic_bag",
               "plastic_container", "other"]

CLASS_PROMPTS = {
    "plastic_bottle":    ["a plastic bottle", "a clear PET drink bottle",
                          "an empty plastic water bottle"],
    "glass_bottle":      ["a glass bottle", "an empty glass beer or wine bottle"],
    "glass_jar":         ["a glass jar", "a glass food jar with a lid"],
    "metal_can":         ["a metal drink can", "an aluminium beverage can",
                          "a tin food can"],
    "carton":            ["a cardboard box", "a corrugated cardboard carton"],
    "tetrapack":         ["a tetra pak drink carton", "a juice or milk carton"],
    "paper":             ["a sheet of paper", "crumpled paper waste", "a paper cup"],
    "plastic_bag":       ["a plastic bag", "a soft plastic wrapper or film"],
    "plastic_container": ["a plastic container", "an HDPE plastic container",
                          "a plastic tub or cup"],
    "other":             ["other miscellaneous waste", "an unrecognised object"],
}


def _pooled(o):
    """transformers 5.x get_*_features returns BaseModelOutputWithPooling."""
    return o if torch.is_tensor(o) else o.pooler_output

DISTRACTORS = {
    "_hand":      ["a human hand", "a person's hand holding an object"],
    "_table":     ["an empty grey table surface",
                   "a plain grey tabletop with nothing on it"],
    "_shadow":    ["a shadow on a grey table"],
    "_equipment": ["a digital kitchen scale", "a ruler or measuring tape"],
    "_fragment":  ["a blurry partial object at the edge of the frame"],
}

CROP_PAD = 0.10


class Bank:
    """SigLIP-2 text bank: 10 classes (two prompt templates each) + distractors."""

    def __init__(self, model_id: str, device: str):
        from transformers import AutoModel, AutoProcessor
        self.device = device
        self.model = AutoModel.from_pretrained(model_id).to(device).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)

        self.names: list[str] = []
        feats = []
        with torch.no_grad():
            for name in CLASS_NAMES:
                prompts = [f"This is a photo of {p}." for p in CLASS_PROMPTS[name]]
                prompts += [f"a photo of {p} on a grey table, studio product photo."
                            for p in CLASS_PROMPTS[name]]
                feats.append(self._embed_text(prompts))
                self.names.append(name)
            for name, plist in DISTRACTORS.items():
                prompts = [f"This is a photo of {p}." for p in plist]
                feats.append(self._embed_text(prompts))
                self.names.append(name)
        bank = torch.stack(feats)
        self.bank = (bank / bank.norm(dim=-1, keepdim=True)).cpu().numpy()
        self.n_real = len(CLASS_NAMES)

    def _embed_text(self, prompts: list[str]) -> torch.Tensor:
        t = self.proc(text=prompts, return_tensors="pt", padding="max_length",
                      truncation=True).to(self.device)
        e = _pooled(self.model.get_text_features(**t))
        e = e / e.norm(dim=-1, keepdim=True)
        return e.mean(0)

    @torch.no_grad()
    def encode_images(self, pil_list: list[Image.Image]) -> np.ndarray:
        t = self.proc(images=pil_list, return_tensors="pt").to(self.device)
        e = _pooled(self.model.get_image_features(**t))
        e = e / e.norm(dim=-1, keepdim=True)
        return e.cpu().numpy()


def crop_box(im: Image.Image, bbox, pad: float = CROP_PAD):
    W, H = im.size
    x, y, w, h = bbox
    dx, dy = w * pad, h * pad
    x1 = int(max(0, x - dx)); y1 = int(max(0, y - dy))
    x2 = int(min(W, x + w + dx)); y2 = int(min(H, y + h + dy))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return im.crop((x1, y1, x2, y2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datasets/real495/raw_detections.json")
    ap.add_argument("--image-root", default="datasets/real495/images")
    ap.add_argument("--model", default="google/siglip2-large-patch16-384")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--dedup-iou", type=float, default=0.85)
    ap.add_argument("--out", default="datasets/real495/proposals.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    raw = json.loads(Path(args.raw).read_text())
    image_root = Path(args.image_root)

    print(f"[siglip] loading {args.model}")
    bank = Bank(args.model, args.device)
    n_real = bank.n_real

    out_images = []
    batch_pil: list[Image.Image] = []
    batch_ref: list[dict] = []          # detection dicts awaiting sims
    t0 = time.time()

    def flush():
        if not batch_pil:
            return
        embs = bank.encode_images(batch_pil)          # (B, d)
        sims = embs @ bank.bank.T                     # (B, n_names)
        for ref, s in zip(batch_ref, sims):
            full_arg = int(np.argmax(s))
            real = s[:n_real]
            order = np.argsort(-real)
            ref["siglip_pred"] = bank.names[full_arg]
            ref["siglip_sugg"] = bank.names[int(order[0])]
            ref["siglip_top3"] = [[bank.names[int(k)], round(float(real[k]), 4)]
                                  for k in order[:3]]
        batch_pil.clear()
        batch_ref.clear()

    n_boxes = 0
    for i, im_meta in enumerate(raw["images"]):
        fn = im_meta["file_name"]
        dets = dedup(raw["detections"].get(fn, []), args.dedup_iou)
        entry = {"file_name": fn, "width": im_meta["width"],
                 "height": im_meta["height"], "detections": dets}
        out_images.append(entry)
        if dets:
            with Image.open(image_root / fn) as im:
                im = im.convert("RGB")
                for d in dets:
                    crop = crop_box(im, d["bbox"])
                    if crop is None:
                        d["siglip_pred"] = d["siglip_sugg"] = None
                        d["siglip_top3"] = []
                        continue
                    batch_pil.append(crop)
                    batch_ref.append(d)
                    n_boxes += 1
                    if len(batch_pil) >= args.batch_size:
                        flush()
        if (i + 1) % 25 == 0 or i == len(raw["images"]) - 1:
            flush()
            print(f"  {i + 1}/{len(raw['images'])}  boxes={n_boxes} "
                  f"({time.time() - t0:.0f}s)")
    flush()

    out = {
        "meta": {**raw["meta"], "siglip_model": args.model,
                 "crop_pad": CROP_PAD, "dedup_iou": args.dedup_iou,
                 "distractors": list(DISTRACTORS)},
        "images": out_images,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(f"[siglip] wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")

    # summary: how often the full-bank argmax is a distractor
    n_d = sum(1 for e in out_images for d in e["detections"]
              if d.get("siglip_pred", "").startswith("_"))
    print(f"[siglip] boxes scored: {n_boxes}; distractor-argmax: {n_d}")


if __name__ == "__main__":
    main()
