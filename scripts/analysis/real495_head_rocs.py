"""Threshold-free scores of the frozen fold heads on the measured real495 tuples."""
from __future__ import annotations

import argparse, json, statistics as st, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sklearn.metrics import roc_auc_score
from src.models import build_fusion_model
from scripts.fusion.train_fusion import TuplesDataset, collate_fn
from scripts.fusion.eval_fraud import collect_predictions


def rec_at_fpr(s: np.ndarray, y: np.ndarray, f: float) -> float:
    neg = np.sort(s[y == 0]); thr = neg[min(int((1 - f) * len(neg)), len(neg) - 1)]
    return float((s[y == 1] > thr).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("experiments/fusion/v3_4_folds"))
    ap.add_argument("--target-tuples", type=Path, default=Path("datasets/multimodal/real495/tuples.csv"))
    ap.add_argument("--target-cache", type=Path, default=Path("experiments/fusion/roi_cache_real495_noreal/epoch_17_real495.pt"))
    ap.add_argument("--group", default="gated_lam5_mu0")
    ap.add_argument("--out", type=Path, default=Path("experiments/fusion/v3_4_ood/real495_rocs.json"))
    a = ap.parse_args()
    cache = torch.load(a.target_cache, weights_only=True)
    loader = DataLoader(TuplesDataset(a.target_tuples, cache), batch_size=4096, collate_fn=collate_fn)
    out = {}
    for k in range(4):
        yaml = f"configs/folds/fold{k}/calibration_dev.yaml"; m5 = []
        for s in range(42, 52):
            m = build_fusion_model("gated", density_yaml=yaml, input_repr="log", use_gate=False)
            m.load_state_dict(torch.load(a.runs_root / f"fold{k}" / "proj" / f"{a.group}_s{s}" / "best.pth", weights_only=True)["state_dict"]); m.eval()
            sc, y, _ = collect_predictions(m, None, loader, "cpu"); sc, y = sc.numpy(), y.numpy()
            m5.append((roc_auc_score(y, sc), rec_at_fpr(sc, y, .05), rec_at_fpr(sc, y, .10), rec_at_fpr(sc, y, .20)))
        out[k] = dict(zip(("auc", "r5", "r10", "r20"), (st.mean(v[i] for v in m5) for i in range(4))))
        print(f"fold{k} head on real495: AUC {out[k]['auc']:.3f}  recall@FPR 5/10/20% = {out[k]['r5']:.3f}/{out[k]['r10']:.3f}/{out[k]['r20']:.3f}")
    a.out.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
