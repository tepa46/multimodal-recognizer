"""Evaluate frozen heads of the fold protocol on an external tuple file."""
from __future__ import annotations

import argparse, json, statistics as st, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models import build_fusion_model
from scripts.fusion.train_fusion import TuplesDataset, collate_fn, model_kwargs
from scripts.fusion.eval_fraud import collect_predictions, per_type_breakdown, _binary_metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tuples", type=Path, required=True)
    ap.add_argument("--target-cache", type=Path, required=True, help="ROI cache .pt for the target tuples")
    ap.add_argument("--runs-root", type=Path, default=Path("experiments/fusion/v3_4_folds"))
    ap.add_argument("--stats-tag", default="proj")
    ap.add_argument("--group", default="gated_lam5_mu0")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(42, 52)))
    ap.add_argument("--per-fold-target", default=None,
                    help="Format string with {k} for a per-fold target tuple file "
                         "(when the target's expected mass depends on the fold table).")
    ap.add_argument("--input-repr", default="log"); ap.add_argument("--no-gate", action="store_true", default=True)
    ap.add_argument("--no-expected-mass", action="store_true"); ap.add_argument("--gate-input", default="vdm"); ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    cache = torch.load(args.target_cache, weights_only=True)
    out = {"folds": {}, "meta": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}}
    for k in args.folds:
        tp = Path(args.per_fold_target.format(k=k)) if args.per_fold_target else args.target_tuples
        loader = DataLoader(TuplesDataset(tp, cache), batch_size=4096, shuffle=False, collate_fn=collate_fn)
        stats = json.load(open(args.runs_root / f"fold{k}" / f"stats_{args.stats_tag}.json"))
        yaml = f"configs/folds/fold{k}/calibration_dev.yaml"
        rule_tau = stats["runs"]["rule"]["val_threshold"]
        from src.models.multimodal_fusion import load_density_config, density_lookup_tensor
        from scripts.fusion.train_fusion import CLASS_NAMES
        rho = density_lookup_tensor(load_density_config(yaml)["densities"], list(CLASS_NAMES))
        rs, ry, rt = collect_predictions(None, rho, loader, "cpu")
        rec = {"rule": per_type_breakdown(rs, ry, rt, rule_tau) | {"tau": rule_tau}, "seeds": {}}
        for s in args.seeds:
            key = f"{args.group}_s{s}"; run = stats["runs"][key]
            m = build_fusion_model("gated", density_yaml=yaml, **model_kwargs(args))
            m.load_state_dict(torch.load(args.runs_root / f"fold{k}" / args.stats_tag / key / "best.pth", weights_only=True)["state_dict"]); m.eval()
            sc, y, t = collect_predictions(m, None, loader, "cpu")
            from sklearn.metrics import roc_auc_score
            rec["seeds"][str(s)] = per_type_breakdown(sc, y, t, run["val_threshold"]) | {"threshold": run["val_threshold"], "roc_auc": float(roc_auc_score(y.numpy(), sc.numpy()))}
        out["folds"][str(k)] = rec
        f1s = [v["overall"]["f1"] for v in rec["seeds"].values()]
        print(f"fold{k}: rule F1 {rec['rule']['overall']['f1']:.3f} FPR {rec['rule']['normal']['fpr']:.3f} | head F1 {st.mean(f1s):.3f}±{st.stdev(f1s):.3f} FPR {st.mean(v['normal']['fpr'] for v in rec['seeds'].values()):.3f} AUC {st.mean(v['roc_auc'] for v in rec['seeds'].values()):.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(args.out, "w"), indent=1); print("wrote", args.out)


if __name__ == "__main__":
    main()
