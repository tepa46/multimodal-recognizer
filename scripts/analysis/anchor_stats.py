from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import build_fusion_model  # noqa: E402
from src.models.multimodal_fusion import (  # noqa: E402
    load_density_config, density_lookup_tensor,
)
from scripts.fusion.train_fusion import (  # noqa: E402
    CLASS_NAMES, TuplesDataset, collate_fn, load_or_build_roi_cache,
)
from scripts.fusion.eval_fraud import collect_predictions  # noqa: E402
from scripts.fusion.run_fusion_stats import bootstrap_ci, mcnemar_exact  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stats-json", default="experiments/fusion/stats_v3_3.json")
    p.add_argument("--fusion-dir", default="experiments/fusion/v3_3")
    p.add_argument("--tuples-test",
                   default="datasets/multimodal/v3.3/test/tuples.csv")
    p.add_argument("--detector-config",
                   default="configs/detectors/htc_x101.py")
    p.add_argument("--detector-ckpt",
                   default="experiments/work_dirs/htc_x101_v33/epoch_17.pth")
    p.add_argument("--cache-dir", default="experiments/fusion/roi_cache_v33/")
    p.add_argument("--anchor-seed", type=int, default=42)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()

    stats_path = Path(args.stats_json)
    with open(stats_path) as f:
        stats = json.load(f)

    canon_group = stats["canonical"]["group"]
    s = args.anchor_seed
    canon_run = f"{canon_group}_s{s}"
    # every learned run present in the accumulated stats, at the anchor seed
    learned = [k for k in stats["runs"]
               if k.endswith(f"_s{s}") and (k.startswith("gated_") or
                                            k.startswith("mlp"))]
    learned.sort()
    print(f"canonical = {canon_run}")
    print(f"learned runs at seed {s}: {learned}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    det_ckpt = Path(args.detector_ckpt)
    tuples_test = Path(args.tuples_test)
    cache_test = load_or_build_roi_cache(
        tuples_test,
        Path(args.cache_dir) / f"{det_ckpt.stem}_test.pt",
        Path(args.detector_config), det_ckpt, device)
    loader = DataLoader(TuplesDataset(tuples_test, cache_test),
                        batch_size=1024, shuffle=False, collate_fn=collate_fn)

    packs: dict[str, tuple] = {}
    taus: dict[str, float] = {}

    # rule baseline: no checkpoint, score = normalised mass residual
    cfg = load_density_config("configs/material_density.yaml")
    densities = density_lookup_tensor(cfg["densities"], list(CLASS_NAMES))
    packs["rule"] = collect_predictions(None, densities, loader, device)
    taus["rule"] = float(stats["runs"]["rule"]["val_threshold"])
    print(f"rule: tau*={taus['rule']:.4f}")

    for run in learned:
        variant = "mlp" if run.startswith("mlp") else "gated"
        ckpt = Path(args.fusion_dir) / run / "best.pth"
        if not ckpt.exists():
            print(f"  SKIP {run} - {ckpt} missing")
            continue
        model = build_fusion_model(variant).to(device)
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["state_dict"])
        model.eval()
        packs[run] = collect_predictions(model, None, loader, device)
        taus[run] = float(stats["runs"][run]["val_threshold"])
        print(f"{run}: tau*={taus[run]:.4f}")

    stats["bootstrap"] = {}
    for run in packs:
        score, gt, types = packs[run]
        stats["bootstrap"][run] = {
            "run_key": run, "threshold": taus[run],
            **bootstrap_ci(score, gt, types, taus[run], n_boot=args.n_boot),
        }
        b = stats["bootstrap"][run]
        print(f"bootstrap {run:26s} F1 {b['f1']['mean']:.4f} "
              f"[{b['f1']['lo95']:.4f},{b['f1']['hi95']:.4f}]  "
              f"wc {b['wrong_class_rec']['mean']:.4f} "
              f"[{b['wrong_class_rec']['lo95']:.4f},"
              f"{b['wrong_class_rec']['hi95']:.4f}]")

    def correctness(run: str) -> np.ndarray:
        score, gt, _ = packs[run]
        return (score > taus[run]).long().numpy() == gt.numpy()

    # every run at the anchor seed, tested against the canonical one
    stats["mcnemar"] = {}
    pairs = {f"{r}_vs_canon": (r, canon_run) for r in packs if r != canon_run}
    for name, (ka, kb) in pairs.items():
        if ka not in packs or kb not in packs:
            continue
        stats["mcnemar"][name] = {
            "a": ka, "b": kb, **mcnemar_exact(correctness(ka), correctness(kb)),
        }
        m = stats["mcnemar"][name]
        print(f"mcnemar {name:34s} b={m['b']:4d} c={m['c']:4d} "
              f"p={m['p_value']:.3e}")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nMerged into {stats_path}")


if __name__ == "__main__":
    main()
