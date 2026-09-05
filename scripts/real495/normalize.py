#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageOps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", default="datasets/real495/images")
    ap.add_argument("--quality", type=int, default=92)
    args = ap.parse_args()

    files = sorted(Path(args.image_dir).glob("*.jpg"))
    n_rot, n_ok = 0, 0
    portrait = 0
    for fp in files:
        with Image.open(fp) as im:
            before = im.size
            im = ImageOps.exif_transpose(im)
            after = im.size
            if after[1] > after[0]:
                portrait += 1
            if before != after or "exif" in im.info:
                im.convert("RGB").save(fp, "JPEG", quality=args.quality)
                n_rot += 1
            else:
                n_ok += 1
    print(f"[normalize] {len(files)} files: rewritten={n_rot}, untouched={n_ok}, "
          f"portrait after={portrait}")


if __name__ == "__main__":
    main()
