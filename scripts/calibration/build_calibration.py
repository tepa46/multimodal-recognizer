"""Build the physical-channel calibration from measured specimens."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml

CLASSES = ("plastic_bottle", "glass_bottle", "glass_jar", "metal_can", "carton",
           "tetrapack", "paper", "plastic_bag", "plastic_container", "other")


def proxy_volume(bbox_area: float, image_area: float) -> float:
    if image_area <= 0 or bbox_area <= 0:
        return 1.0
    return max((bbox_area / image_area) ** 1.5 * 8000.0, 1.0)


def source_of(image_path: str) -> str:
    return Path(image_path).name.split("_")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--specimens", type=Path,
                    default=Path("datasets/calibration/specimens_calib.csv"))
    ap.add_argument("--hybrid-coco", type=Path, nargs="+",
                    default=[Path(f"datasets/processed_v3.3/annotations/instances_{s}.json")
                             for s in ("train", "val", "test")],
                    help="Hybrid COCO files: the noise-free, unclamped box proxy "
                         "of their annotations is matched to the measured class "
                         "volumes to fit volume_scale per source.")
    ap.add_argument("--extra-coco", type=str, nargs="*", default=[],
                    help="Extra sources given as name=path/to/instances.json; "
                         "their volume_scale is fitted from the unclamped proxy "
                         "of their annotations (used for the BeachLitter set).")
    ap.add_argument("--photos-coco", type=Path, default=None,
                    help="Optional COCO file whose annotations carry "
                         "attributes.object_id and the specimen dimensions; "
                         "used to measure volume_sigma on the calibration "
                         "specimens. Images themselves are not read.")
    ap.add_argument("--min-annotations", type=int, default=20)
    ap.add_argument("--out", type=Path,
                    default=Path("configs/calibration_packphys.yaml"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.specimens)))
    by_cls: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_cls[r["class_name"]].append(r)

    densities, density_sigma, specimens = {}, {}, {}
    for c in CLASSES:
        L = by_cls[c]
        dens = [float(r["mass_empty_g"]) / float(r["volume_cm3"]) for r in L]
        densities[c] = round(st.median(dens), 5)
        density_sigma[c] = round(st.stdev(math.log(x) for x in dens), 3) if len(dens) > 1 else None
        specimens[c] = [
            {"id": r["specimen_id"], "density": round(d, 5),
             "fill_ratio": (round(float(r["mass_loaded_g"]) / float(r["mass_empty_g"]), 3)
                            if r["mass_loaded_g"] else None)}
            for r, d in zip(L, dens)
        ]
    pooled = [s for s in density_sigma.values() if s is not None]
    for c in CLASSES:
        if density_sigma[c] is None:
            density_sigma[c] = round(st.median(pooled), 3)

    v_meas = {c: st.median(float(r["volume_cm3"]) for r in by_cls[c]) for c in CLASSES}
    proxy: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def add_coco(name: str | None, path: Path) -> None:
        cc = json.load(open(path))
        cats = {c["id"]: c["name"] for c in cc["categories"]}
        ims = {im["id"]: im for im in cc["images"]}
        for a in cc["annotations"]:
            im = ims[a["image_id"]]
            rel = a["bbox"][2] * a["bbox"][3] / (im["width"] * im["height"])
            src = name or source_of(im["file_name"])
            proxy[src][cats[a["category_id"]]].append(rel ** 1.5 * 8000.0)

    for p in args.hybrid_coco:
        add_coco(None, p)
    for spec in args.extra_coco:
        name, path = spec.split("=", 1)
        add_coco(name, Path(path))
    volume_scale = {}
    for src, cls in proxy.items():
        logs = [(math.log(v_meas[c] / st.median(v)), len(v)) for c, v in cls.items()
                if c != "other" and len(v) >= args.min_annotations]
        volume_scale[src] = round(math.exp(sum(x * n for x, n in logs) / sum(n for _, n in logs)), 4)

    volume_sigma = None
    if args.photos_coco:
        coco = json.load(open(args.photos_coco))
        cats = {c["id"]: c["name"] for c in coco["categories"]}
        ims = {im["id"]: im for im in coco["images"]}
        ids = {r["specimen_id"] for r in rows}
        per_session: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for a in coco["annotations"]:
            at = a["attributes"]
            if at.get("object_id") not in ids:
                continue
            im = ims[a["image_id"]]
            pv = proxy_volume(a["bbox"][2] * a["bbox"][3], im["width"] * im["height"])
            vt = at["height_cm"] * at["width_cm"] * at["depth_cm"]
            per_session[im.get("camera_id", "all")].append((cats[a["category_id"]], math.log(vt / pv)))
        res: dict[str, list[float]] = defaultdict(list)
        for L in per_session.values():
            k = st.median(x for _, x in L)
            for c, x in L:
                res[c].append(x - k)
        volume_sigma = {c: round(st.pstdev(res[c]), 3) for c in CLASSES if c in res}

    out = {
        "source": str(args.specimens),
        "n_specimens": len(rows),
        "densities": densities,
        "density_sigma": density_sigma,
        "volume_scale": volume_scale,
        "volume_sigma": volume_sigma,
        "weight_noise_sigma": 0.05,
        "fraud_multipliers": {"foreign_object": [1.8, 3.5]},
        "specimens": specimens,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Generated by scripts/calibration/build_calibration.py. Do not edit.\n")
        yaml.safe_dump(out, f, sort_keys=False)
    print(f"wrote {args.out}")
    print(f"{'class':18s} n   rho      sigma_rho  n_fill  V_sigma")
    for c in CLASSES:
        nf = sum(1 for s in specimens[c] if s["fill_ratio"])
        vs = volume_sigma.get(c) if volume_sigma else None
        print(f"{c:18s} {len(specimens[c]):2d}  {densities[c]:.4f}   {density_sigma[c]:.2f}     {nf:2d}     "
              f"{vs if vs is not None else '-'}")
    print("volume_scale:", volume_scale)


if __name__ == "__main__":
    main()
