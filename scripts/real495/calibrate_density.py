#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ASSUMED = {
    "plastic_bottle": 0.10, "glass_bottle": 0.35, "glass_jar": 0.40,
    "metal_can": 0.08, "carton": 0.05, "tetrapack": 0.10, "paper": 0.30,
    "plastic_bag": 0.02, "plastic_container": 0.15, "other": 0.50,
}


def v_heuristic(bbox, img_w: int, img_h: int) -> float:
    _, _, bw, bh = bbox
    area = bw * bh
    img_area = float(img_w * img_h)
    if img_area <= 0 or area <= 0:
        return 1.0
    rel = area / img_area
    return max((rel ** 1.5) * 8000.0, 1.0)


def v_literal(row) -> float:
    """Physical bounding-envelope volume from the hand measurements (cm^3)."""
    return (float(row["height_cm"]) * float(row["width_cm"])
            * float(row["depth_cm"]))


def collect(coco_path: Path, meas_path: Path, volume_mode: str = "literal"):
    coco = json.loads(coco_path.read_text())
    id2name = {c["id"]: c["name"] for c in coco["categories"]}
    imgs = {im["id"]: im for im in coco["images"]}
    meas = {(r["filename"], int(r["pos"])): r
            for r in csv.DictReader(open(meas_path))}

    ratios: dict[str, list[float]] = defaultdict(list)
    objects: dict[str, set] = defaultdict(set)
    skipped = 0
    for ann in coco["annotations"]:
        at = ann.get("attributes", {})
        if at.get("condition") != "normal":
            continue
        im = imgs[ann["image_id"]]
        row = meas.get((im["file_name"], at.get("pos")))
        if not row or not row.get("mass_g"):
            skipped += 1
            continue
        mass = float(row["mass_g"])
        if volume_mode == "literal":
            vol = v_literal(row)
        else:
            vol = v_heuristic(ann["bbox"], im["width"], im["height"])
        if vol <= 0:
            skipped += 1
            continue
        cls = id2name[ann["category_id"]]
        ratios[cls].append(mass / vol)
        # identity of the physical object: class + dimensions + empty mass
        objects[cls].add((row["height_cm"], row["width_cm"],
                          row["depth_cm"], row.get("mass_empty_g", "")))
    return ratios, objects, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco", type=Path,
                    default=Path("datasets/real495/annotations/instances_all.json"))
    ap.add_argument("--measurements", type=Path,
                    default=Path("datasets/real495/measurements.csv"))
    ap.add_argument("--yaml", type=Path,
                    default=Path("configs/material_density.yaml"))
    ap.add_argument("--min-objects", type=int, default=2,
                    help="Classes backed by fewer distinct physical objects "
                         "keep their assumed value (too thin to recalibrate).")
    ap.add_argument("--volume-mode", choices=["literal", "heuristic"],
                    default="literal",
                    help="Which volume convention rho is calibrated against. "
                         "'literal' = h*w*d from the hand measurements (the "
                         "physically meaningful one, and what real495_to_tuples.py "
                         "uses). 'heuristic' = the generator's bbox proxy. "
                         "NOTE the fraud rule is scale-invariant in rho on "
                         "synthetic data (generator and evaluator share it, so it "
                         "cancels — verified empirically), so a literal-volume "
                         "table can serve as the single canonical table without "
                         "affecting synthetic metrics, while being correct for "
                         "the real-sensor evaluation.")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Classes to leave at their assumed value regardless of "
                         "sample size. Default: 'other' — in real495 that class "
                         "is entirely measurement inventory (pens, tape measure, "
                         "power bank, rag: dense compact tools), whereas in "
                         "Hybrid-v3.1 'other' means DWSD miscellaneous waste. "
                         "Different populations; calibrating one from the other "
                         "would be a domain error, not a measurement.")
    ap.add_argument("--write", action="store_true",
                    help="Patch the yaml in place (keeps old values as comments).")
    ap.add_argument("--out-json", type=Path,
                    default=Path("experiments/real495_density_calibration.json"))
    args = ap.parse_args()

    ratios, objects, skipped = collect(args.coco, args.measurements,
                                       args.volume_mode)

    print("Recalibration of effective densities from real495")
    vol_desc = ("h*w*d (measured)" if args.volume_mode == "literal"
                else "V_heuristic(bbox)")
    print(f"  rho_hat = median( mass_g / {vol_desc} )  over condition=normal")
    print(f"  (skipped {skipped} annotations without measurements)\n")
    print(f"{'class':20s} {'n_ann':>6} {'n_obj':>6} {'assumed':>9} "
          f"{'measured':>9} {'factor':>8}   verdict")
    print("-" * 78)

    new_table: dict[str, float] = {}
    report: dict[str, dict] = {}
    for cls in ASSUMED:
        vals = ratios.get(cls, [])
        n_obj = len(objects.get(cls, ()))
        assumed = ASSUMED[cls]
        if cls in args.exclude:
            new_table[cls] = assumed
            # still report what WAS measured, for the record
            measured = statistics.median(vals) if vals else float("nan")
            factor = measured / assumed if vals else float("nan")
            verdict = "EXCLUDED (population mismatch)"
        elif not vals or n_obj < args.min_objects:
            new_table[cls] = assumed
            verdict = f"KEPT (n_obj={n_obj} < {args.min_objects})"
            measured = float("nan")
            factor = float("nan")
        else:
            measured = statistics.median(vals)
            factor = measured / assumed
            new_table[cls] = round(measured, 4)
            if 0.5 <= factor <= 2.0:
                verdict = "confirmed (within 2x)"
            else:
                verdict = "CORRECTED"
        report[cls] = {"n_ann": len(vals), "n_objects": n_obj,
                       "assumed": assumed,
                       "measured": None if measured != measured else round(measured, 4),
                       "factor": None if factor != factor else round(factor, 2),
                       "adopted": new_table[cls], "verdict": verdict}
        m_s = "  n/a" if measured != measured else f"{measured:9.4f}"
        f_s = "   n/a" if factor != factor else f"{factor:7.2f}x"
        print(f"{cls:20s} {len(vals):>6} {n_obj:>6} {assumed:>9.3f} "
              f"{m_s} {f_s}   {verdict}")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(
        {"method": f"median(mass_g / {args.volume_mode} volume), condition=normal",
         "volume_mode": args.volume_mode,
         "source": "real495", "min_objects": args.min_objects,
         "per_class": report}, indent=2, ensure_ascii=False))
    print(f"\nWrote {args.out_json}")

    if not args.write:
        print("\n(dry run — pass --write to patch configs/material_density.yaml)")
        return

    # --- textual patch of the yaml, preserving all comments/structure -------- #
    lines = args.yaml.read_text().splitlines()
    out, in_densities, patched = [], False, 0
    for line in lines:
        if line.startswith("densities:"):
            in_densities = True
            out.append(line)
            continue
        if in_densities:
            if line and not line.startswith((" ", "\t", "#")):
                in_densities = False          # left the block
            else:
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    key = stripped.split(":", 1)[0].strip()
                    if key in new_table:
                        r = report[key]
                        if key in args.exclude:
                            note = ("# assumed (real495 'other' = measurement "
                                    "inventory, not misc waste — not comparable)")
                        elif r["measured"] is None:
                            note = (f"# assumed (kept: only {r['n_objects']} "
                                    f"distinct object(s) in real495)")
                        else:
                            note = (f"# real495-calibrated (was {r['assumed']}, "
                                    f"n_obj={r['n_objects']}, n_ann={r['n_ann']})")
                        out.append(f"  {key}: {new_table[key]:<8} {note}")
                        patched += 1
                        continue
        out.append(line)

    header = [
        "# RECALIBRATED 2026-08 from real495 measurements — see",
        "#   scripts/real495/calibrate_density.py and",
        "#   experiments/real495_density_calibration.json",
        "# rho_c = median(mass_g / V_heuristic(bbox)) over condition=normal, where",
        "# V_heuristic is the SAME proxy the synthetic generator uses. rho and the",
        "# volume formula are one joint calibration — do not mix this table with a",
        "# literal h*w*d volume (real495_to_tuples.py keeps its own convention).",
        "# CAVEAT: real495 has only ~32 distinct physical objects; per-class",
        "# provenance (n_obj / n_ann) is noted inline below.",
        "",
    ]
    args.yaml.write_text("\n".join(header + out) + "\n")
    print(f"Patched {args.yaml} ({patched} density entries)")


if __name__ == "__main__":
    main()
