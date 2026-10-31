"""Density check on measured masses (real495), with an independent table."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

import yaml

CLASSES = ("plastic_bottle", "glass_bottle", "glass_jar", "metal_can", "carton",
           "tetrapack", "paper", "plastic_bag", "plastic_container", "other")


def auc(pos: list[float], neg: list[float]) -> float:
    n = 0.0
    for p in pos:
        for q in neg:
            n += 1.0 if p > q else 0.5 if p == q else 0.0
    return n / (len(pos) * len(neg))


def recall_at_fpr(pos: list[float], neg: list[float], fpr: float) -> tuple[float, float]:
    thr = sorted(neg)[min(int((1 - fpr) * len(neg)), len(neg) - 1)]
    return sum(p > thr for p in pos) / len(pos), thr


def residual(row: dict, rho: float) -> float:
    v = float(row["height_cm"]) * float(row["width_cm"]) * float(row["depth_cm"])
    return abs(float(row["mass_g"]) / v - rho) / rho


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measurements", type=Path,
                    default=Path("datasets/real495/measurements_specimens.csv"))
    ap.add_argument("--calibration", type=Path,
                    default=Path("configs/calibration_packphys.yaml"))
    ap.add_argument("--submitted-table", type=Path,
                    default=Path("configs/material_density.yaml"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=Path("results/real495/measured_eval.json"))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.measurements)))
    calib = yaml.safe_load(open(args.calibration))
    rho_calib = calib["densities"]
    rho_sub = yaml.safe_load(open(args.submitted_table))["densities"]

    by_cls_spec: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["condition"] == "normal":
            v = float(r["height_cm"]) * float(r["width_cm"]) * float(r["depth_cm"])
            by_cls_spec[r["class"]][r["specimen_id"]].append(float(r["mass_g"]) / v)

    def rho_loso(r: dict) -> float | None:
        vals = [x for sid, xs in by_cls_spec[r["class"]].items() if sid != r["specimen_id"] for x in xs]
        return st.median(vals) if vals else None

    protocols = {
        "submitted_in_sample": lambda r: rho_sub[r["class"]],
        "real495_leave_one_specimen_out": rho_loso,
        "calibration_89_specimens": lambda r: rho_calib[r["class"]],
    }
    result: dict = {"n_rows": len(rows), "n_specimens": len({r["specimen_id"] for r in rows}),
                    "protocols": {}}
    rng = random.Random(0)
    for name, fn in protocols.items():
        scored = []
        for r in rows:
            rho = fn(r)
            if rho is None:
                continue
            scored.append((r["specimen_id"], r["condition"], residual(r, rho)))
        pos = [s for _, c, s in scored if c != "normal"]
        neg = [s for _, c, s in scored if c == "normal"]
        rec = {"n_scored": len(scored), "n_tampered": len(pos), "n_intact": len(neg),
               "auc": auc(pos, neg),
               "fpr_at_tau_0.25": sum(n > 0.25 for n in neg) / len(neg)}
        for f in (0.05, 0.10, 0.20):
            rc, thr = recall_at_fpr(pos, neg, f)
            rec[f"recall_at_fpr_{int(f*100)}"] = rc
            rec[f"tau_at_fpr_{int(f*100)}"] = thr
        by_spec: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for sid, c, s in scored:
            by_spec[sid].append((c, s))
        ids = list(by_spec)
        aucs, recs = [], []
        for _ in range(args.n_boot):
            sample = [rng.choice(ids) for _ in ids]
            p = [s for sid in sample for c, s in by_spec[sid] if c != "normal"]
            n = [s for sid in sample for c, s in by_spec[sid] if c == "normal"]
            if not p or not n:
                continue
            aucs.append(auc(p, n)); recs.append(recall_at_fpr(p, n, 0.10)[0])
        rec["auc_ci95"] = [sorted(aucs)[int(0.025 * len(aucs))], sorted(aucs)[int(0.975 * len(aucs))]]
        rec["recall_at_fpr_10_ci95"] = [sorted(recs)[int(0.025 * len(recs))], sorted(recs)[int(0.975 * len(recs))]]
        thr10 = rec["tau_at_fpr_10"]
        per_cls = {}
        for c in CLASSES:
            nn = [s for r, (sid, cond, s) in zip(rows, scored) if cond == "normal" and r["class"] == c] if len(scored) == len(rows) else None
            if nn:
                per_cls[c] = {"n": len(nn), "fpr_tau_0.25": sum(x > 0.25 for x in nn) / len(nn),
                              "fpr_tau_fpr10": sum(x > thr10 for x in nn) / len(nn)}
        rec["per_class_intact"] = per_cls
        result["protocols"][name] = rec
        print(f"{name:32s} n={len(scored)}  AUC {rec['auc']:.3f} [{rec['auc_ci95'][0]:.3f}, {rec['auc_ci95'][1]:.3f}]  "
              f"FPR@0.25 {rec['fpr_at_tau_0.25']:.2f}  recall@FPR5/10/20 "
              f"{rec['recall_at_fpr_5']:.3f}/{rec['recall_at_fpr_10']:.3f}/{rec['recall_at_fpr_20']:.3f}  "
              f"(tau {rec['tau_at_fpr_5']:.2f}/{rec['tau_at_fpr_10']:.2f}/{rec['tau_at_fpr_20']:.2f})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=1)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
