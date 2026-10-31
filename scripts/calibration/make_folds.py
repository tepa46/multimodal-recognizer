"""Specimen-disjoint folds for the physics channel."""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specimens", type=Path, default=Path("datasets/calibration/specimens_calib.csv"))
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--photos-coco", type=Path, required=True)
    ap.add_argument("--extra-coco", type=str, nargs="*", default=[])
    ap.add_argument("--out-data", type=Path, default=Path("datasets/calibration/folds"))
    ap.add_argument("--out-cfg", type=Path, default=Path("configs/folds"))
    ap.add_argument("--data-root", type=Path, default=Path("datasets/multimodal/v3.4_folds"),
                    help="Where to generate the fold tuples (train/val from the dev "
                         "pool, test from the test pool with the dev table).")
    ap.add_argument("--no-tuples", action="store_true", help="Only write specimen groups and yamls.")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.specimens)))
    rng = random.Random(args.seed)
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cls[r["class_name"]].append(r)
    groups: list[list[dict]] = [[] for _ in range(args.k)]
    for c, L in sorted(by_cls.items()):
        if len(L) < args.k:
            raise SystemExit(f"class {c} has {len(L)} specimens < K={args.k}")
        L = L[:]
        rng.shuffle(L)
        start = rng.randrange(args.k)
        for i, r in enumerate(L):
            groups[(start + i) % args.k].append(r)

    manifest = {"k": args.k, "seed": args.seed, "source": str(args.specimens), "folds": []}
    fieldnames = list(rows[0].keys())
    for k in range(args.k):
        test = groups[k]
        dev = [r for j, G in enumerate(groups) if j != k for r in G]
        assert not ({r["specimen_id"] for r in test} & {r["specimen_id"] for r in dev})
        dd = args.out_data / f"fold{k}"; dd.mkdir(parents=True, exist_ok=True)
        cd = args.out_cfg / f"fold{k}"; cd.mkdir(parents=True, exist_ok=True)
        for name, G in (("dev", dev), ("test", test)):
            with open(dd / f"specimens_{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(G)
            cmd = [args.python, "scripts/calibration/build_calibration.py",
                   "--specimens", str(dd / f"specimens_{name}.csv"),
                   "--photos-coco", str(args.photos_coco),
                   "--out", str(cd / f"calibration_{name}.yaml")]
            if args.extra_coco:
                cmd += ["--extra-coco", *args.extra_coco]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        manifest["folds"].append({
            "fold": k,
            "test_specimens": [r["specimen_id"] for r in test],
            "dev_specimens": [r["specimen_id"] for r in dev],
            "test_per_class": {c: sum(1 for r in test if r["class_name"] == c) for c in by_cls},
        })
        print(f"fold {k}: dev {len(dev)}  test {len(test)}  "
              f"test per class {manifest['folds'][-1]['test_per_class']}")
    (args.out_data / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if not args.no_tuples:
        for k in range(args.k):
            cd = args.out_cfg / f"fold{k}"
            for sp, seed, pool in (("train", 42, "dev"), ("val", 43, "dev"), ("test", 44, "test")):
                cmd = [args.python, "scripts/fusion/generate_multimodal_tuples.py",
                       "--coco-json", f"datasets/processed_v3.3/annotations/instances_{sp}.json",
                       "--image-root", f"datasets/processed_v3.3/images/{sp}/",
                       "--output-dir", str(args.data_root / f"fold{k}" / sp), "--seed", str(seed),
                       "--skip-depth", "--calibration", str(cd / f"calibration_{pool}.yaml")]
                if pool == "test":
                    cmd += ["--table-calibration", str(cd / "calibration_dev.yaml")]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
            print(f"fold {k}: tuples written to {args.data_root / f'fold{k}'}")
    seen = [s for f in manifest["folds"] for s in f["test_specimens"]]
    assert len(seen) == len(rows) == len(set(seen))
    print("wrote", args.out_data / "manifest.json")


if __name__ == "__main__":
    main()
