from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Make `src` importable regardless of CWD.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import build_fusion_model  # noqa: E402


CLASS_NAMES = (
    "plastic_bottle", "glass_bottle", "glass_jar", "metal_can",
    "carton", "tetrapack", "paper", "plastic_bag",
    "plastic_container", "other",
)
NAME_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Tuples dataset
# ---------------------------------------------------------------------------

class TuplesDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
        roi_cache: dict[int, torch.Tensor] | None = None,
        roi_dim: int = 256,
    ) -> None:
        self.rows: list[dict[str, Any]] = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                self.rows.append(row)
        self.roi_cache = roi_cache or {}
        self.roi_dim = roi_dim

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        r = self.rows[idx]
        ann_id = int(r["annotation_id"])
        if ann_id in self.roi_cache:
            roi = self.roi_cache[ann_id]
        else:
            roi = torch.zeros(self.roi_dim)
        return {
            "annotation_id": torch.tensor(ann_id),
            "roi_features": roi,
            "volume": torch.tensor(float(r["volume_cm3"])),
            "measured_weight": torch.tensor(float(r["measured_weight_g"])),
            "reported_class": torch.tensor(int(r["reported_class_id"])),
            "true_class": torch.tensor(int(r["true_class_id"])),
            "is_fraud": torch.tensor(float(int(r["is_fraud"]))),
            "image_path": r["image_path"],
            "fraud_type": r["fraud_type"],
            "expected_weight": torch.tensor(
                float(r["expected_weight_loso_g"])
                if "expected_weight_loso_g" in r else float("nan")),
            "bbox": torch.tensor([
                float(r["bbox_x"]), float(r["bbox_y"]),
                float(r["bbox_w"]), float(r["bbox_h"]),
            ]),
        }


def collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    keys = ["annotation_id", "roi_features", "volume", "measured_weight",
            "reported_class", "true_class", "is_fraud", "bbox",
            "expected_weight"]
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    out["image_path"] = [b["image_path"] for b in batch]
    out["fraud_type"] = [b["fraud_type"] for b in batch]
    return out


# ---------------------------------------------------------------------------
# ROI feature extraction (frozen detector)
# ---------------------------------------------------------------------------

def extract_roi_features(
    detector: Any,
    rows_by_image: dict[str, list[dict[str, Any]]],
    device: str,
    log_every: int = 50,
) -> dict[int, torch.Tensor]:
    import mmcv  # noqa: F401  — required by the registered preprocessor
    from mmengine.dataset import Compose
    detector.eval()

    cache: dict[int, torch.Tensor] = {}
    cfg = detector.cfg.copy()
    test_pipeline = cfg.test_dataloader.dataset.pipeline
    pipeline = Compose(test_pipeline)

    extractor = detector.roi_head.bbox_roi_extractor
    # HTC stores 3 cascade extractors in an nn.ModuleList; take stage-0
    # (all cascade stages share the same FPN-pooled roi features here).
    if isinstance(extractor, (list, tuple, nn.ModuleList)):
        extractor = extractor[0]

    images = list(rows_by_image.items())
    t0 = time.time()
    with torch.inference_mode():
        for i, (img_path, rows) in enumerate(images):
            data = dict(img_path=img_path, img_id=i)
            data = pipeline(data)
            img_tensor = data["inputs"]
            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)
            img_tensor = img_tensor.float().to(device)
            # Replicate mmdet's data preprocessor (mean/std normalisation).
            img_tensor = detector.data_preprocessor(
                {"inputs": [img_tensor.squeeze(0)],
                 "data_samples": [data["data_samples"]]},
                training=False,
            )["inputs"]

            feats = detector.extract_feat(img_tensor)
            # Build rois tensor: (N, 5) [batch_idx, x1, y1, x2, y2]
            rois = []
            ann_ids: list[int] = []
            # Account for resize: data_samples carry scale_factor.
            sf = data["data_samples"].metainfo.get("scale_factor", (1.0, 1.0))
            sx, sy = float(sf[0]), float(sf[1])
            for r in rows:
                x = float(r["bbox_x"]); y = float(r["bbox_y"])
                w = float(r["bbox_w"]); h = float(r["bbox_h"])
                rois.append([0.0, x * sx, y * sy, (x + w) * sx, (y + h) * sy])
                ann_ids.append(int(r["annotation_id"]))
            if not rois:
                continue
            rois_t = torch.tensor(rois, device=device, dtype=torch.float32)
            roi_feats = extractor(feats[: extractor.num_inputs], rois_t)
            # roi_feats: (N, 256, 7, 7) → mean-pool.
            pooled = roi_feats.mean(dim=(2, 3)).cpu()
            for ann_id, feat in zip(ann_ids, pooled):
                cache[ann_id] = feat

            if (i + 1) % log_every == 0:
                rate = (i + 1) / (time.time() - t0 + 1e-6)
                print(f"  ROI extract {i+1}/{len(images)} "
                      f"({rate:.1f} img/s, {len(cache)} anns)", flush=True)

    return cache


def group_rows_by_image(csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            grouped[row["image_path"]].append(row)
    return dict(grouped)


def load_or_build_roi_cache(
    csv_path: Path,
    cache_path: Path,
    detector_config: Path,
    detector_ckpt: Path,
    device: str,
) -> dict[int, torch.Tensor]:
    if cache_path.exists():
        print(f"  Loading cached ROI features: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=True)
    print(f"  Building ROI cache from {detector_ckpt}")
    from mmdet.apis import init_detector
    detector = init_detector(str(detector_config), str(detector_ckpt), device=device)
    rows_by_image = group_rows_by_image(csv_path)
    cache = extract_roi_features(detector, rows_by_image, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    print(f"  Cached {len(cache)} ROI features → {cache_path}")
    return cache


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def model_kwargs(args: Any) -> dict[str, Any]:
    return {
        "dropout": getattr(args, "dropout", 0.3),
        "input_repr": getattr(args, "input_repr", "raw"),
        "use_expected_mass": not getattr(args, "no_expected_mass", False),
        "use_gate": not getattr(args, "no_gate", False),
    }


def expected_or_none(batch: dict[str, Any], device: str) -> torch.Tensor | None:
    e = batch["expected_weight"]
    if torch.isnan(e).all():
        return None
    return e.to(device)


def _apply_modality_mask(
    roi: torch.Tensor,
    volume: torch.Tensor,
    measured_weight: torch.Tensor,
    mask_modality: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if mask_modality == "none":
        return roi, volume, measured_weight
    if mask_modality == "visual":
        return torch.zeros_like(roi), volume, measured_weight
    if mask_modality == "depth":
        return roi, torch.zeros_like(volume), measured_weight
    if mask_modality == "mass":
        return roi, volume, torch.zeros_like(measured_weight)
    raise ValueError(f"Unknown mask_modality={mask_modality!r}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str,
             lambda_phys: float = 1.0,
             mu_cls: float = 0.0,
             mask_modality: str = "none") -> dict[str, float]:
    model.eval()
    tp = fp = fn = tn = 0
    losses = []
    for batch in loader:
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
            batch["is_fraud"].to(device),
            lambda_phys=lambda_phys,
            mu_cls=mu_cls,
            expected_weight=expected_or_none(batch, device),
        )
        if "loss" in out:
            losses.append(float(out["loss"]))
        pred = (torch.sigmoid(out["fraud_logit"]) > 0.5).long().cpu()
        gt = batch["is_fraud"].long()
        tp += int(((pred == 1) & (gt == 1)).sum())
        fp += int(((pred == 1) & (gt == 0)).sum())
        fn += int(((pred == 0) & (gt == 1)).sum())
        tn += int(((pred == 0) & (gt == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    fpr = fp / max(fp + tn, 1)
    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "f1": f1, "prec": prec, "rec": rec, "fpr": fpr,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def train_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float,
    lambda_phys: float,
    output_dir: Path,
    mask_modality: str = "none",
    mu_cls: float = 0.0,
    weight_decay: float = 1e-4,
) -> dict[str, Any]:
    optim_ = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    best_f1 = -1.0
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = []
        for batch in train_loader:
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
                batch["is_fraud"].to(device),
                lambda_phys=lambda_phys,
                mu_cls=mu_cls,
                expected_weight=expected_or_none(batch, device),
            )
            loss = out["loss"]
            optim_.zero_grad()
            loss.backward()
            optim_.step()
            ep_loss.append(float(loss))

        val = evaluate(model, val_loader, device, lambda_phys=lambda_phys,
                       mu_cls=mu_cls, mask_modality=mask_modality)
        train_loss = float(np.mean(ep_loss)) if ep_loss else float("nan")
        print(f"  epoch {ep:3d}/{epochs}  train_loss={train_loss:.4f}  "
              f"val_loss={val['loss']:.4f}  val_F1={val['f1']:.4f}  "
              f"prec={val['prec']:.3f}  rec={val['rec']:.3f}  fpr={val['fpr']:.3f}",
              flush=True)
        history.append({"epoch": ep, "train_loss": train_loss, **val})
        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state is not None:
        ckpt_path = output_dir / "best.pth"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "best_f1": best_f1,
                    "history": history}, ckpt_path)
        print(f"  Saved best checkpoint (val F1={best_f1:.4f}) → {ckpt_path}")
    return {"best_f1": best_f1, "history": history}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True,
                        choices=["rule", "mlp", "gated"])
    parser.add_argument("--detector-config", type=Path,
                        default=Path("configs/detectors/htc_x101.py"))
    parser.add_argument("--detector-ckpt", type=Path,
                        default=Path("experiments/work_dirs/htc_x101_v33/epoch_17.pth"))
    parser.add_argument("--tuples-train", type=Path,
                        default=Path("datasets/multimodal/v3.3/train/tuples.csv"))
    parser.add_argument("--tuples-val", type=Path,
                        default=Path("datasets/multimodal/v3.3/val/tuples.csv"))
    parser.add_argument("--cache-dir", type=Path,
                        default=Path("experiments/fusion/roi_cache/"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/fusion/"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lambda-phys", type=float, default=1.0)
    parser.add_argument("--mu-cls", type=float, default=0.0,
                        help="Weight of the CE anchor tying the auxiliary "
                             "soft-class head to the reported class on "
                             "legitimate samples (gated only).")
    parser.add_argument("--density-yaml", type=Path,
                        default=Path("configs/material_density.yaml"),
                        help="Density table loaded into the model (see run_fusion_stats.py).")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout in the fraud head (submitted: 0.3).")
    parser.add_argument("--input-repr", choices=["raw", "log"], default="raw",
                        help="Scalar input representation of the learned heads "
                             "(see MultimodalFusion).")
    parser.add_argument("--no-expected-mass", action="store_true",
                        help="Ablation: drop m_hat from the learned heads' inputs.")
    parser.add_argument("--gate-input", choices=["vdm", "vdmc", "vdmc+r"], default="vdm",
                        help="Inputs of the gate network (see MultimodalFusion).")
    parser.add_argument("--no-gate", action="store_true",
                        help="Ablation: same projections, no sigmoid gates.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-roi", action="store_true",
                        help="Skip ROI extraction even for mlp/gated "
                             "(uses zero features; sanity only).")
    parser.add_argument("--mask-modality",
                        choices=["none", "visual", "depth", "mass"],
                        default="none",
                        help="Zero-out one modality before fusion (paper "
                             "§6.4 modality ablation). visual = ROI features, "
                             "depth = volume, mass = measured weight.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible fusion training. "
                             "Sets torch / numpy / random seeds + cudnn "
                             "deterministic mode. Default: no seed (random "
                             "init each run, ±0.02 F1 variance).")
    args = parser.parse_args()

    if args.seed is not None and args.seed >= 0:
        import random as _py_random
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        _py_random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        # cudnn deterministic — small speed cost, ~5%, but reproducible.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Seed: {args.seed} (torch + numpy + python.random + cudnn-deterministic)")

    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = args.output_dir / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)

    # ROI cache (skipped for rule).
    roi_cache_train: dict[int, torch.Tensor] = {}
    roi_cache_val: dict[int, torch.Tensor] = {}
    if args.variant != "rule" and not args.no_roi:
        ckpt_tag = args.detector_ckpt.stem
        cache_train = args.cache_dir / f"{ckpt_tag}_train.pt"
        cache_val = args.cache_dir / f"{ckpt_tag}_val.pt"
        roi_cache_train = load_or_build_roi_cache(
            args.tuples_train, cache_train,
            args.detector_config, args.detector_ckpt, device,
        )
        roi_cache_val = load_or_build_roi_cache(
            args.tuples_val, cache_val,
            args.detector_config, args.detector_ckpt, device,
        )

    train_ds = TuplesDataset(args.tuples_train, roi_cache_train)
    val_ds = TuplesDataset(args.tuples_val, roi_cache_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            collate_fn=collate_fn)
    print(f"Train: {len(train_ds)} tuples / {len(roi_cache_train)} cached ROIs")
    print(f"Val:   {len(val_ds)} tuples / {len(roi_cache_val)} cached ROIs")

    model = build_fusion_model(args.variant, density_yaml=args.density_yaml, **model_kwargs(args)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Variant={args.variant}  trainable params={n_params}")

    if args.variant == "rule":
        # No training. Just evaluate at default threshold; threshold sweep
        # belongs to eval_fraud.py.
        val = evaluate(model, val_loader, device)
        print(f"  rule (τ=0.3): F1={val['f1']:.4f}  prec={val['prec']:.3f} "
              f"rec={val['rec']:.3f}  fpr={val['fpr']:.3f}")
        with open(out_dir / "metrics.json", "w") as f:
            json.dump(val, f, indent=2)
        return

    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        lambda_phys=args.lambda_phys,
        output_dir=out_dir,
        mask_modality=args.mask_modality,
        mu_cls=args.mu_cls,
    )


if __name__ == "__main__":
    main()
