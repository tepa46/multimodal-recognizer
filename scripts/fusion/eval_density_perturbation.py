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
from src.models.multimodal_fusion import (  # noqa: E402
    load_density_config, density_lookup_tensor,
)
from scripts.fusion.train_fusion import (  # noqa: E402
    CLASS_NAMES, TuplesDataset, collate_fn, load_or_build_roi_cache,
)
from scripts.fusion.eval_fraud import (  # noqa: E402
    collect_predictions, per_type_breakdown,
)
from scripts.fusion.run_fusion_stats import sweep_threshold_on_scores  # noqa: E402


def summarise(breakdowns: list[dict[str, Any]]) -> dict[str, Any]:
    """mean ± std of the key metrics across perturbation draws."""
    keys = (("f1", ("overall", "f1")), ("fpr", ("overall", "fpr")),
            ("heavy_fill_rec", ("heavy_fill", "rec")),
            ("foreign_object_rec", ("foreign_object", "rec")),
            ("wrong_class_rec", ("wrong_class", "rec")))
    out: dict[str, Any] = {}
    for name, (sect, met) in keys:
        vals = np.array([b[sect][met] for b in breakdowns], dtype=np.float64)
        out[name] = {"mean": float(vals.mean()),
                     "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                     "min": float(vals.min()), "max": float(vals.max())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gated-ckpt", type=Path, required=True)
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
    parser.add_argument("--sigma-rho", type=float, default=0.2)
    parser.add_argument("--n-draws", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json", type=Path,
                        default=Path("experiments/fusion/density_perturbation_v33.json"))
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    ckpt_tag = args.detector_ckpt.stem
    cache_val = load_or_build_roi_cache(
        args.tuples_val, args.cache_dir / f"{ckpt_tag}_val.pt",
        args.detector_config, args.detector_ckpt, device)
    cache_test = load_or_build_roi_cache(
        args.tuples_test, args.cache_dir / f"{ckpt_tag}_test.pt",
        args.detector_config, args.detector_ckpt, device)
    val_loader = DataLoader(TuplesDataset(args.tuples_val, cache_val),
                            batch_size=1024, shuffle=False,
                            collate_fn=collate_fn)
    test_loader = DataLoader(TuplesDataset(args.tuples_test, cache_test),
                             batch_size=1024, shuffle=False,
                             collate_fn=collate_fn)

    cfg = load_density_config("configs/material_density.yaml")
    rho0 = density_lookup_tensor(cfg["densities"], list(CLASS_NAMES))

    # ---------------- unperturbed operating points ----------------
    # Rule: tau* on unperturbed val. Gated: threshold swept on unperturbed
    # val scores (deployment fixes the threshold before densities drift).
    rule_grid = np.linspace(0.05, 0.5, 19)
    learned_grid = np.linspace(0.05, 0.95, 91)

    val_pack = collect_predictions(None, rho0, val_loader, device)
    rule_tau, _ = sweep_threshold_on_scores(val_pack[0], val_pack[1], rule_grid)
    test_pack0 = collect_predictions(None, rho0, test_loader, device)
    rule_base = per_type_breakdown(*test_pack0, rule_tau)

    model = build_fusion_model("gated").to(device)
    state = torch.load(args.gated_ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.eval()
    gval = collect_predictions(model, None, val_loader, device)
    gated_tau, _ = sweep_threshold_on_scores(gval[0], gval[1], learned_grid)
    gtest0 = collect_predictions(model, None, test_loader, device)
    gated_base = per_type_breakdown(*gtest0, gated_tau)

    print(f"Unperturbed: rule tau*={rule_tau:.3f} F1={rule_base['overall']['f1']:.4f} | "
          f"gated tau*={gated_tau:.3f} F1={gated_base['overall']['f1']:.4f} "
          f"wrong-class={gated_base['wrong_class']['rec']:.4f}")

    # ---------------- perturbation draws ----------------
    rng = np.random.default_rng(args.seed)
    rule_bd: list[dict[str, Any]] = []
    gated_bd: list[dict[str, Any]] = []
    factors_log: list[list[float]] = []
    for i in range(args.n_draws):
        factors = np.exp(rng.normal(0.0, args.sigma_rho, len(CLASS_NAMES)))
        factors_log.append([float(x) for x in factors])
        rho_hat = rho0 * torch.tensor(factors, dtype=torch.float32)

        # rule with perturbed lookup
        tpack = collect_predictions(None, rho_hat, test_loader, device)
        rule_bd.append(per_type_breakdown(*tpack, rule_tau))

        # gated with perturbed lookup (expected-mass input feature)
        model.density_table.copy_(rho_hat.to(device))
        gpack = collect_predictions(model, None, test_loader, device)
        gated_bd.append(per_type_breakdown(*gpack, gated_tau))
        model.density_table.copy_(rho0.to(device))

        print(f"  draw {i+1:2d}/{args.n_draws}: "
              f"rule F1={rule_bd[-1]['overall']['f1']:.4f} | "
              f"gated F1={gated_bd[-1]['overall']['f1']:.4f} "
              f"wrong-class={gated_bd[-1]['wrong_class']['rec']:.4f}",
              flush=True)

    rule_sum = summarise(rule_bd)
    gated_sum = summarise(gated_bd)
    result = {
        "sigma_rho": args.sigma_rho, "n_draws": args.n_draws,
        "seed": args.seed, "gated_ckpt": str(args.gated_ckpt),
        "thresholds": {"rule": rule_tau, "gated": gated_tau},
        "baseline": {
            "rule": {k: rule_base[s][m] for k, (s, m) in
                     (("f1", ("overall", "f1")), ("fpr", ("overall", "fpr")),
                      ("wrong_class_rec", ("wrong_class", "rec")))},
            "gated": {k: gated_base[s][m] for k, (s, m) in
                      (("f1", ("overall", "f1")), ("fpr", ("overall", "fpr")),
                       ("wrong_class_rec", ("wrong_class", "rec")))},
        },
        "perturbed": {"rule": rule_sum, "gated": gated_sum},
        "delta": {
            "rule_f1": rule_sum["f1"]["mean"] - rule_base["overall"]["f1"],
            "gated_f1": gated_sum["f1"]["mean"] - gated_base["overall"]["f1"],
            "gated_wrong_class": (gated_sum["wrong_class_rec"]["mean"]
                                  - gated_base["wrong_class"]["rec"]),
        },
        "factors": factors_log,
    }

    print("\n=== Density perturbation summary "
          f"(sigma_rho={args.sigma_rho}, {args.n_draws} draws) ===")
    for name, summ, base in (("rule", rule_sum, rule_base),
                             ("gated", gated_sum, gated_base)):
        print(f"  {name}: F1 {summ['f1']['mean']:.3f}±{summ['f1']['std']:.3f} "
              f"(Δ{summ['f1']['mean'] - base['overall']['f1']:+.3f})  "
              f"wrong-class {summ['wrong_class_rec']['mean']:.3f}"
              f"±{summ['wrong_class_rec']['std']:.3f} "
              f"(Δ{summ['wrong_class_rec']['mean'] - base['wrong_class']['rec']:+.3f})")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
