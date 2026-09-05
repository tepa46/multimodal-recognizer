from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import build_fusion_model                       # noqa: E402
from scripts.fusion.train_fusion import (TuplesDataset, collate_fn,    # noqa: E402
                                  CLASS_NAMES)

FRAUD_TYPES = ("normal", "heavy_fill", "foreign_object", "wrong_class")


@torch.no_grad()
def collect(model, loader, device):
    score, gt, ftype, tcls, rcls = [], [], [], [], []
    for b in loader:
        out = model(b["roi_features"].to(device), b["volume"].to(device),
                    b["measured_weight"].to(device),
                    b["reported_class"].to(device).long())
        score.append(torch.sigmoid(out["fraud_logit"]).cpu())
        gt.append(b["is_fraud"].long())
        ftype.extend(b["fraud_type"])
        tcls.append(b["true_class"]); rcls.append(b["reported_class"])
    return (torch.cat(score), torch.cat(gt), np.array(ftype),
            torch.cat(tcls).numpy(), torch.cat(rcls).numpy())


def recall(pred, gt):
    tp = int(((pred == 1) & (gt == 1)).sum()); fn = int(((pred == 0) & (gt == 1)).sum())
    return tp / max(tp + fn, 1), tp + fn


def fpr(pred, gt):
    fp = int(((pred == 1) & (gt == 0)).sum()); tn = int(((pred == 0) & (gt == 0)).sum())
    return fp / max(fp + tn, 1), fp + tn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tuples-test", type=Path,
                    default=Path("datasets/multimodal/v3.3/test/tuples.csv"))
    ap.add_argument("--cache", type=Path,
                    default=Path("experiments/fusion/roi_cache/epoch_30_test.pt"))
    ap.add_argument("--ckpt", type=Path,
                    default=Path("experiments/fusion/gated/best.pth"))
    ap.add_argument("--variant", default="gated")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--output-json", type=Path,
                    default=Path("experiments/fusion/eval_fraud_perclass.json"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache = torch.load(args.cache, map_location="cpu", weights_only=True)
    ds = TuplesDataset(args.tuples_test, cache)
    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_fn)

    model = build_fusion_model(args.variant).to(device)
    state = torch.load(args.ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"]); model.eval()

    score, gt, ftype, tcls, rcls = collect(model, loader, device)
    pred = (score > args.threshold).long()

    # ---- sanity: overall + per-subtype (should match paper Table 4) ----
    ov_r, _ = recall(pred, gt); ov_fpr, _ = fpr(pred, gt)
    tp = int(((pred == 1) & (gt == 1)).sum()); fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    prec = tp / max(tp + fp, 1); f1 = 2 * prec * ov_r / max(prec + ov_r, 1e-9)
    print(f"[sanity] n={len(gt)} overall F1={f1:.3f} recall={ov_r:.3f} FPR={ov_fpr:.3f}")
    sub = {}
    for t in FRAUD_TYPES:
        m = ftype == t
        if t == "normal":
            v, n = fpr(pred[m], gt[m]); sub[t] = {"fpr": round(v, 3), "n": int(n)}
        else:
            v, n = recall(pred[m], gt[m]); sub[t] = {"recall": round(v, 3), "n": int(n)}
    print(f"[sanity] per-subtype: {sub}")

    # ---- per TRUE class breakdown ----
    rows = []
    for cid, cname in enumerate(CLASS_NAMES):
        cm = tcls == cid
        if cm.sum() == 0:
            continue
        row = {"class": cname, "n_total": int(cm.sum())}
        # normal FPR
        nm = cm & (ftype == "normal")
        v, n = fpr(pred[nm], gt[nm]); row["normal_fpr"] = round(v, 3); row["n_normal"] = int(n)
        # per-subtype recall (grouped by true class)
        for t in ("heavy_fill", "foreign_object", "wrong_class"):
            sm = cm & (ftype == t)
            if sm.sum() == 0:
                row[f"{t}_rec"] = None; row[f"n_{t}"] = 0
            else:
                v, n = recall(pred[sm], gt[sm])
                row[f"{t}_rec"] = round(v, 3); row[f"n_{t}"] = int(n)
        rows.append(row)

    out = {"sanity": {"overall_f1": round(f1, 3), "overall_recall": round(ov_r, 3),
                      "overall_fpr": round(ov_fpr, 3), "per_subtype": sub},
           "threshold": args.threshold, "group_by": "true_class",
           "per_class": rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))

    # ---- print table ----
    print(f"\n=== Per TRUE class (gated, thr={args.threshold}) ===")
    print(f"{'class':<18}{'n':>5}{'normFPR':>9}{'heavy':>8}{'foreign':>9}{'wrong':>8}"
          f"{'(n_wc)':>8}")
    for r in rows:
        def s(x): return "  -  " if x is None else f"{x:.3f}"
        print(f"{r['class']:<18}{r['n_total']:>5}{r['normal_fpr']:>9.3f}"
              f"{s(r['heavy_fill_rec']):>8}{s(r['foreign_object_rec']):>9}"
              f"{s(r['wrong_class_rec']):>8}{r['n_wrong_class']:>8}")
    print(f"\n-> {args.output_json}")


if __name__ == "__main__":
    main()
