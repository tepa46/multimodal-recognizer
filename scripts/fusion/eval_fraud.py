from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import build_fusion_model  # noqa: E402

# Re-use TuplesDataset / collate / ROI cache loader from train_fusion.
from scripts.fusion.train_fusion import (  # noqa: E402
    TuplesDataset, collate_fn, load_or_build_roi_cache, evaluate,
)


# Synthetic fraud types from generate_multimodal_tuples.py.
FRAUD_TYPES = ("normal", "heavy_fill", "foreign_object", "wrong_class")


# ---------------------------------------------------------------------------
# Per-fraud-type breakdown
# ---------------------------------------------------------------------------

def _binary_metrics(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    """Compute F1 / precision / recall / FPR / counts on a binary slice."""
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    fpr = fp / max(fp + tn, 1)
    return {"f1": f1, "prec": prec, "rec": rec, "fpr": fpr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "n": tp + fp + fn + tn}


@torch.no_grad()
def collect_predictions(
    model: Any | None,
    densities: torch.Tensor | None,
    loader: DataLoader,
    device: str,
    mask_modality: str = "none",
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    from scripts.fusion.train_fusion import _apply_modality_mask
    all_score: list[torch.Tensor] = []
    all_gt: list[torch.Tensor] = []
    all_type: list[str] = []
    for batch in loader:
        if model is None:
            assert densities is not None
            cls = batch["reported_class"].long()
            vol = batch["volume"]
            meas = batch["measured_weight"]
            expected = vol * densities[cls]
            score = (meas - expected).abs() / torch.clamp(expected, min=1e-3)
        else:
            roi = batch["roi_features"].to(device)
            volume = batch["volume"].to(device)
            measured_weight = batch["measured_weight"].to(device)
            roi, volume, measured_weight = _apply_modality_mask(
                roi, volume, measured_weight, mask_modality,
            )
            out = model(
                roi,
                volume,
                measured_weight,
                batch["reported_class"].to(device).long(),
            )
            # Use sigmoid(logit) as score so threshold lives in [0,1].
            score = torch.sigmoid(out["fraud_logit"]).cpu()
        all_score.append(score)
        all_gt.append(batch["is_fraud"].long())
        all_type.extend(batch["fraud_type"])
    return torch.cat(all_score), torch.cat(all_gt), all_type


def per_type_breakdown(
    score: torch.Tensor,
    gt: torch.Tensor,
    fraud_type: list[str],
    threshold: float,
) -> dict[str, dict[str, float]]:
    pred = (score > threshold).long()
    types = np.array(fraud_type)
    out: dict[str, dict[str, float]] = {}
    out["overall"] = _binary_metrics(pred, gt)
    out["overall"]["threshold"] = float(threshold)
    for t in FRAUD_TYPES:
        mask = torch.from_numpy(types == t)
        if not mask.any():
            out[t] = {"f1": float("nan"), "rec": float("nan"),
                      "fpr": float("nan"), "n": 0}
            continue
        sub_pred = pred[mask]
        sub_gt = gt[mask]
        m = _binary_metrics(sub_pred, sub_gt)
        # For 'normal' (all gt=0) F1 is meaningless — report only FPR.
        out[t] = m
    return out


# ---------------------------------------------------------------------------
# Threshold sweep for rule variant
# ---------------------------------------------------------------------------

@torch.no_grad()
def sweep_threshold(
    csv_loader: DataLoader,
    densities: torch.Tensor,
    num_classes: int,
    grid: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Sweep τ for the rule-based detector and return the best τ on F1."""
    # Pre-compute per-sample relative error once.
    all_rel_err: list[torch.Tensor] = []
    all_y: list[torch.Tensor] = []
    for batch in csv_loader:
        cls = batch["reported_class"].long()
        vol = batch["volume"]
        meas = batch["measured_weight"]
        density = densities[cls]
        expected = vol * density
        rel_err = (meas - expected).abs() / torch.clamp(expected, min=1e-3)
        all_rel_err.append(rel_err)
        all_y.append(batch["is_fraud"])
    rel_err = torch.cat(all_rel_err)
    y = torch.cat(all_y).long()
    pos = (y == 1)
    neg = (y == 0)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())

    best_f1 = -1.0
    best_tau = float(grid[0])
    best_metrics: dict[str, float] = {}
    for tau in grid:
        pred = (rel_err > tau).long()
        tp = int(((pred == 1) & pos).sum())
        fp = int(((pred == 1) & neg).sum())
        fn = int(((pred == 0) & pos).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        fpr = fp / max(n_neg, 1)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)
            best_metrics = {
                "f1": f1, "prec": prec, "rec": rec, "fpr": fpr,
                "tp": tp, "fp": fp, "fn": fn, "tn": n_neg - fp,
            }
    return best_tau, best_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+",
                        default=["rule", "mlp", "gated"])
    parser.add_argument("--tuples-val", type=Path,
                        default=Path("datasets/multimodal/v3.3/val/tuples.csv"))
    parser.add_argument("--tuples-test", type=Path,
                        default=Path("datasets/multimodal/v3.3/test/tuples.csv"))
    parser.add_argument("--detector-config", type=Path,
                        default=Path("configs/detectors/htc_x101.py"))
    parser.add_argument("--detector-ckpt", type=Path,
                        default=Path("experiments/work_dirs/htc_x101_v33/epoch_17.pth"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("experiments/fusion/roi_cache/"))
    parser.add_argument("--fusion-dir", type=Path,
                        default=Path("experiments/fusion/"))
    parser.add_argument("--bbox-map", type=float, default=None,
                        help="bbox mAP from MMDet test (shared across variants)")
    parser.add_argument("--mask-map", type=float, default=None,
                        help="mask mAP from MMDet test (shared across variants)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-roi", action="store_true",
                        help="Skip ROI extraction (debug only)")
    parser.add_argument("--mask-modality",
                        choices=["none", "visual", "depth", "mass"],
                        default="none",
                        help="Zero-out one modality before fusion (paper "
                             "§6.4 modality ablation). Must match the value "
                             "used during train_fusion.py for that ckpt.")
    parser.add_argument("--output-json", type=Path,
                        default=Path("experiments/fusion/eval_fraud.json"))
    parser.add_argument("--learned-threshold", type=float, default=0.5,
                        help="Operating threshold for learned variants "
                             "(paper protocol: the val-swept τ of the "
                             "checkpoint, frozen — e.g. for OOD eval).")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    needs_roi = (any(v in ("mlp", "gated") for v in args.variants)
                 and not args.no_roi)
    cache_val: dict[int, torch.Tensor] = {}
    cache_test: dict[int, torch.Tensor] = {}
    if needs_roi:
        ckpt_tag = args.detector_ckpt.stem
        cache_val = load_or_build_roi_cache(
            args.tuples_val, args.cache_dir / f"{ckpt_tag}_val.pt",
            args.detector_config, args.detector_ckpt, device,
        )
        cache_test = load_or_build_roi_cache(
            args.tuples_test, args.cache_dir / f"{ckpt_tag}_test.pt",
            args.detector_config, args.detector_ckpt, device,
        )

    val_ds = TuplesDataset(args.tuples_val, cache_val)
    test_ds = TuplesDataset(args.tuples_test, cache_test)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, collate_fn=collate_fn)

    results: dict[str, dict[str, Any]] = {}

    # Densities are needed by the rule variant and reused for logging.
    from src.models.multimodal_fusion import (
        load_density_config, density_lookup_tensor,
    )
    from scripts.fusion.train_fusion import CLASS_NAMES
    cfg = load_density_config("configs/material_density.yaml")
    densities = density_lookup_tensor(cfg["densities"], list(CLASS_NAMES))

    for variant in args.variants:
        print(f"\n=== {variant} ===")
        if variant == "rule":
            # Sweep τ on val using collect_predictions (returns rel_err
            # as score), then evaluate at best τ on test with full
            # per-fraud-type breakdown.
            val_score, val_gt, val_types = collect_predictions(
                None, densities, val_loader, device,
            )
            grid = np.linspace(0.05, 0.5, 19)
            best_tau = float(grid[0]); best_f1 = -1.0
            for tau in grid:
                m = _binary_metrics((val_score > tau).long(), val_gt)
                if m["f1"] > best_f1:
                    best_f1 = m["f1"]; best_tau = float(tau)
            print(f"  best τ on val: {best_tau:.3f} → val F1={best_f1:.4f}")
            val_breakdown = per_type_breakdown(val_score, val_gt, val_types, best_tau)
            test_score, test_gt, test_types = collect_predictions(
                None, densities, test_loader, device,
            )
            test_breakdown = per_type_breakdown(test_score, test_gt, test_types, best_tau)
            print(f"  test (τ={best_tau:.3f}): "
                  f"F1={test_breakdown['overall']['f1']:.4f} "
                  f"FPR={test_breakdown['overall']['fpr']:.4f}")
            results[variant] = {"threshold": best_tau,
                                "val": val_breakdown, "test": test_breakdown}
            continue

        ckpt = args.fusion_dir / variant / "best.pth"
        if not ckpt.exists():
            print(f"  SKIP — no checkpoint at {ckpt}")
            results[variant] = {"error": f"no ckpt at {ckpt}"}
            continue
        model = build_fusion_model(variant).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["state_dict"])
        model.eval()
        val_score, val_gt, val_types = collect_predictions(
            model, None, val_loader, device,
            mask_modality=args.mask_modality,
        )
        test_score, test_gt, test_types = collect_predictions(
            model, None, test_loader, device,
            mask_modality=args.mask_modality,
        )
        # Learned variants: sigmoid-logit threshold (default 0.5; pass
        # the checkpoint's val-swept τ via --learned-threshold to eval
        # at the paper's frozen operating point).
        threshold = args.learned_threshold
        val_breakdown = per_type_breakdown(val_score, val_gt, val_types, threshold)
        test_breakdown = per_type_breakdown(test_score, test_gt, test_types, threshold)
        print(f"  val:  F1={val_breakdown['overall']['f1']:.4f}  "
              f"FPR={val_breakdown['overall']['fpr']:.4f}")
        print(f"  test: F1={test_breakdown['overall']['f1']:.4f}  "
              f"FPR={test_breakdown['overall']['fpr']:.4f}")
        results[variant] = {"val": val_breakdown, "test": test_breakdown,
                            "ckpt_best_f1": float(state.get("best_f1", -1)),
                            "threshold": threshold}

    # ------------------- main ablation table -------------------
    print("\n=== Ablation summary (test split) ===")
    bbox_str = f"{args.bbox_map:.3f}" if args.bbox_map is not None else "—"
    mask_str = f"{args.mask_map:.3f}" if args.mask_map is not None else "—"
    print(f"{'Variant':<22} | {'Fraud F1':>9} | {'Fraud FPR':>10} | "
          f"{'bbox mAP':>9} | {'mask mAP':>9}")
    print("-" * 72)
    for variant in args.variants:
        r = results.get(variant, {})
        if "test" in r:
            ov = r["test"]["overall"]
            f1 = ov["f1"]; fpr = ov["fpr"]
            print(f"{variant:<22} | {f1:9.4f} | {fpr:10.4f} | "
                  f"{bbox_str:>9} | {mask_str:>9}")
        else:
            print(f"{variant:<22} | {'—':>9} | {'—':>10} | "
                  f"{bbox_str:>9} | {mask_str:>9}")

    # ------------------- per-fraud-type breakdown -------------------
    # Recall per fraud type (the C2-relevant story: wrong_class is hard
    # for rule-based, easy for learned variants with ROI features).
    print("\n=== Per-fraud-type recall (test split) ===")
    print(f"{'Variant':<22} | {'normal-FPR':>10} | "
          f"{'heavy_fill':>10} | {'foreign_obj':>11} | {'wrong_class':>11}")
    print("-" * 76)
    for variant in args.variants:
        r = results.get(variant, {})
        if "test" not in r:
            continue
        b = r["test"]
        normal_fpr = b.get("normal", {}).get("fpr", float("nan"))
        hf = b.get("heavy_fill", {}).get("rec", float("nan"))
        fo = b.get("foreign_object", {}).get("rec", float("nan"))
        wc = b.get("wrong_class", {}).get("rec", float("nan"))
        print(f"{variant:<22} | {normal_fpr:10.4f} | "
              f"{hf:10.4f} | {fo:11.4f} | {wc:11.4f}")
    print("  (normal-FPR = false-positive rate on legit samples; lower is better)")
    print("  (other columns = recall on each fraud subtype; higher is better)")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump({"results": results, "bbox_map": args.bbox_map,
                   "mask_map": args.mask_map}, f, indent=2)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
