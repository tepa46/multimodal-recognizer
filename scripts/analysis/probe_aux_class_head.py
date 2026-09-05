from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.fusion.train_fusion import (TuplesDataset, collate_fn, CLASS_NAMES,
                                  load_or_build_roi_cache)
from src.models.multimodal_fusion import MultimodalFusion

ap = argparse.ArgumentParser()
ap.add_argument("--fusion-dirs", nargs="+", required=True)
ap.add_argument("--tuples-test", default="datasets/multimodal/v3.3/test/tuples.csv")
ap.add_argument("--detector-config", default="configs/detectors/htc_x101.py")
ap.add_argument("--detector-ckpt", default="experiments/work_dirs/htc_x101_v33/epoch_17.pth")
ap.add_argument("--cache-dir", default="experiments/fusion/roi_cache_v33/")
ap.add_argument("--output-json", default="experiments/fusion/q_probe.json")
ap.add_argument("--device", default="cuda")
a = ap.parse_args()

dev = a.device if torch.cuda.is_available() else "cpu"
tag = Path(a.detector_ckpt).stem
cache = load_or_build_roi_cache(Path(a.tuples_test),
                                Path(a.cache_dir) / f"{tag}_test.pt",
                                Path(a.detector_config), Path(a.detector_ckpt), dev)
ds = TuplesDataset(Path(a.tuples_test), cache)
dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2, collate_fn=collate_fn)

out = {}
for d in a.fusion_dirs:
    ck = Path(d) / "best.pth"
    if not ck.exists():
        print(f"SKIP {d}: no best.pth"); continue
    m = MultimodalFusion(variant="gated", num_classes=len(CLASS_NAMES)).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev, weights_only=True)["state_dict"])
    m.eval()
    tot = corr = 0; ps = []; hs = []
    with torch.no_grad():
        for b in dl:
            keep = b["is_fraud"].to(dev) == 0          # legitimate only
            if keep.sum() == 0: continue
            r = m(b["roi_features"].to(dev), b["volume"].to(dev),
                  b["measured_weight"].to(dev), b["reported_class"].to(dev).long())
            q = r["soft_class"][keep]
            y = b["true_class"].to(dev).long()[keep]
            corr += (q.argmax(1) == y).sum().item(); tot += y.numel()
            ps.append(q.gather(1, y[:, None]).squeeze(1).cpu())
            hs.append((-(q.clamp_min(1e-12).log() * q).sum(1)).cpu())
    p = torch.cat(ps); h = torch.cat(hs)
    out[Path(d).name] = {"n_legit": tot, "top1": corr / tot,
                         "p_true": p.mean().item(), "entropy": h.mean().item(),
                         "eff_classes": math.exp(h.mean().item())}
    print(f"{Path(d).name:26s} n={tot:5d}  top1={corr/tot:.4f}  "
          f"p_true={p.mean():.4f}  H={h.mean():.4f}  eff_k={math.exp(h.mean().item()):.2f}")
print(f"\nuniform over {len(CLASS_NAMES)} classes: H={math.log(len(CLASS_NAMES)):.4f}, eff_k={len(CLASS_NAMES)}")
Path(a.output_json).write_text(json.dumps(out, indent=2))
