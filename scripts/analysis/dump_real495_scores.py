"""Per-object scores of the frozen heads and of the density check on the measured real495 tuples."""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models import build_fusion_model
from scripts.fusion.train_fusion import TuplesDataset, collate_fn, model_kwargs, CLASS_NAMES
from scripts.fusion.eval_fraud import collect_predictions
from src.models.multimodal_fusion import load_density_config, density_lookup_tensor
from sklearn.metrics import roc_auc_score


def recall_at_fpr(pos, neg, fpr):
    neg = np.sort(np.asarray(neg)); pos = np.asarray(pos)
    thr = neg[min(int((1 - fpr) * len(neg)), len(neg) - 1)]
    return float((pos > thr).mean()), float(thr)


def rocs(scores, y):
    pos, neg = scores[y == 1], scores[y == 0]
    out = {"auc": float(roc_auc_score(y, scores))}
    for f in (0.05, 0.10, 0.20):
        out[f"r{int(f*100)}"], out[f"thr{int(f*100)}"] = recall_at_fpr(pos, neg, f)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tuples", type=Path, default=Path("datasets/multimodal/real495/tuples.csv"))
    ap.add_argument("--target-cache", type=Path, default=Path("experiments/fusion/roi_cache_real495_noreal/epoch_17_real495.pt"))
    ap.add_argument("--runs-root", type=Path, default=Path("experiments/fusion/v3_4_folds"))
    ap.add_argument("--stats-tag", default="proj")
    ap.add_argument("--group", default="gated_lam5_mu0")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--input-repr", default="log"); ap.add_argument("--no-gate", action="store_true", default=True)
    ap.add_argument("--no-expected-mass", action="store_true"); ap.add_argument("--gate-input", default="vdm"); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--stored-rocs", type=Path, default=Path("experiments/fusion/v3_4_ood/real495_rocs.json"))
    ap.add_argument("--out", type=Path, default=Path("results/real495/per_object_scores.csv"))
    ap.add_argument("--out-rocs", type=Path, default=Path("results/real495/head_rocs_recomputed.json"))
    args = ap.parse_args()
    cache = torch.load(args.target_cache, weights_only=True)
    loader = DataLoader(TuplesDataset(args.target_tuples, cache), batch_size=4096, shuffle=False, collate_fn=collate_fn)
    rows = list(csv.DictReader(open(args.target_tuples)))
    rho = density_lookup_tensor(load_density_config("configs/folds/fold0/calibration_dev.yaml")["densities"], list(CLASS_NAMES))
    rs, ry, rt = collect_predictions(None, rho, loader, "cpu")
    rs = rs.numpy(); y = ry.numpy().astype(int)
    assert len(rs) == len(rows), (len(rs), len(rows))
    stored = json.load(open(args.stored_rocs)) if args.stored_rocs.exists() else {}
    print("density check:", {k: round(v, 3) for k, v in rocs(rs, y).items()})
    head, report = {}, {}
    for k in args.folds:
        yaml = f"configs/folds/fold{k}/calibration_dev.yaml"
        per_seed_scores, per_seed_metrics = [], []
        for s in args.seeds:
            key = f"{args.group}_s{s}"
            m = build_fusion_model("gated", density_yaml=yaml, **model_kwargs(args))
            m.load_state_dict(torch.load(args.runs_root / f"fold{k}" / args.stats_tag / key / "best.pth", weights_only=True)["state_dict"]); m.eval()
            sc, yy, t = collect_predictions(m, None, loader, "cpu")
            sc = sc.numpy(); per_seed_scores.append(sc); per_seed_metrics.append(rocs(sc, y))
        head[k] = np.mean(per_seed_scores, axis=0)
        on_mean = rocs(head[k], y)
        mean_of = {kk: float(np.mean([pm[kk] for pm in per_seed_metrics])) for kk in per_seed_metrics[0]}
        report[str(k)] = {"on_seed_mean_scores": on_mean, "mean_of_per_seed_metrics": mean_of, "stored": stored.get(str(k))}
        print(f"fold{k}: on seed-mean scores auc {on_mean['auc']:.3f} r5 {on_mean['r5']:.3f} r10 {on_mean['r10']:.3f} r20 {on_mean['r20']:.3f} | "
              f"mean of per-seed auc {mean_of['auc']:.3f} r5 {mean_of['r5']:.3f} r10 {mean_of['r10']:.3f} r20 {mean_of['r20']:.3f} | "
              f"stored {stored.get(str(k))}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(args.out_rocs, "w"), indent=1)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["annotation_id", "image", "class", "fraud_type", "is_fraud", "specimen_id", "volume_cm3", "measured_g", "expected_g",
                    "rule_residual"] + [f"head_fold{k}" for k in args.folds] + ["head_mean"])
        grand = np.mean([head[k] for k in args.folds], axis=0)
        for i, r in enumerate(rows):
            w.writerow([r["annotation_id"], Path(r["image_path"]).name, r["true_class_name"], r["fraud_type"], r["is_fraud"], r["specimen_id"],
                        r["volume_cm3"], r["measured_weight_g"], r["expected_weight_loso_g"], f"{float(rs[i]):.4f}"]
                       + [f"{float(head[k][i]):.5f}" for k in args.folds] + [f"{float(grand[i]):.5f}"])
    print("wrote", args.out, len(rows), "rows;", args.out_rocs)


if __name__ == "__main__":
    main()
