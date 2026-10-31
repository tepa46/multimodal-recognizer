from __future__ import annotations

import argparse
import json
import sys
import time
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
    model_kwargs,
    CLASS_NAMES, TuplesDataset, collate_fn, load_or_build_roi_cache,
    train_loop,
)
from scripts.fusion.eval_fraud import (  # noqa: E402
    collect_predictions, per_type_breakdown, _binary_metrics,
    rule_tolerance_grid,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int, threads: int = 1) -> None:
    import random as _py_random
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    np.random.seed(seed)
    _py_random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sweep_threshold_on_scores(
    score: torch.Tensor, gt: torch.Tensor, grid: np.ndarray,
) -> tuple[float, float]:
    """Return (best_threshold, best_F1) maximising F1 on (score, gt)."""
    best_f1, best_tau = -1.0, float(grid[0])
    for tau in grid:
        m = _binary_metrics((score > tau).long(), gt)
        if m["f1"] > best_f1:
            best_f1, best_tau = m["f1"], float(tau)
    return best_tau, best_f1


def auc_metrics(score: torch.Tensor, gt: torch.Tensor) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = gt.numpy()
    s = score.numpy()
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
    }


def bootstrap_ci(
    score: torch.Tensor,
    gt: torch.Tensor,
    fraud_type: list[str],
    threshold: float,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap 95% CI (percentile) for test F1 and wrong-class recall."""
    rng = np.random.default_rng(seed)
    s = score.numpy()
    y = gt.numpy()
    t = np.array(fraud_type)
    n = len(y)
    f1s, wcs = [], []
    pred_full = (s > threshold).astype(np.int64)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        p, g, ft = pred_full[idx], y[idx], t[idx]
        tp = int(((p == 1) & (g == 1)).sum())
        fp = int(((p == 1) & (g == 0)).sum())
        fn = int(((p == 0) & (g == 1)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-9))
        wc = ft == "wrong_class"
        if wc.any():
            wcs.append(float((p[wc] == 1).mean()))
    def ci(vals: list[float]) -> dict[str, float]:
        a = np.array(vals)
        return {"mean": float(a.mean()),
                "lo95": float(np.percentile(a, 2.5)),
                "hi95": float(np.percentile(a, 97.5))}
    return {"f1": ci(f1s), "wrong_class_rec": ci(wcs), "n_boot": n_boot}


def mcnemar_exact(
    correct_a: np.ndarray, correct_b: np.ndarray,
) -> dict[str, float]:
    """Exact McNemar test on paired per-sample correctness arrays."""
    from scipy.stats import binomtest
    b = int((correct_a & ~correct_b).sum())   # A right, B wrong
    c = int((~correct_a & correct_b).sum())   # A wrong, B right
    if b + c == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    p = binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue
    return {"b": b, "c": c, "p_value": float(p)}


def evaluate_scores(
    val_pack: tuple[torch.Tensor, torch.Tensor, list[str]],
    test_pack: tuple[torch.Tensor, torch.Tensor, list[str]],
    grid: np.ndarray,
) -> dict[str, Any]:
    """Full evaluation record from raw val/test scores."""
    val_score, val_gt, val_types = val_pack
    test_score, test_gt, test_types = test_pack
    tau_star, val_f1_star = sweep_threshold_on_scores(val_score, val_gt, grid)
    rec: dict[str, Any] = {
        "val_threshold": tau_star,
        "val_f1_at_val_tau": val_f1_star,
        "val": {
            "fixed_0.5": per_type_breakdown(val_score, val_gt, val_types, 0.5),
            "val_tau": per_type_breakdown(val_score, val_gt, val_types, tau_star),
        },
        "test": {
            "fixed_0.5": per_type_breakdown(test_score, test_gt, test_types, 0.5),
            "val_tau": per_type_breakdown(test_score, test_gt, test_types, tau_star),
        },
        "test_auc": auc_metrics(test_score, test_gt),
        "val_auc": auc_metrics(val_score, val_gt),
    }
    return rec


def agg(values: list[float]) -> dict[str, float]:
    a = np.array(values, dtype=np.float64)
    return {"mean": float(a.mean()),
            "std": float(a.std(ddof=1)) if len(a) > 1 else 0.0,
            "n": len(a), "values": [float(v) for v in a]}


KEY_METRICS = (
    ("f1_0.5", lambda r: r["test"]["fixed_0.5"]["overall"]["f1"]),
    ("fpr_0.5", lambda r: r["test"]["fixed_0.5"]["overall"]["fpr"]),
    ("f1_valtau", lambda r: r["test"]["val_tau"]["overall"]["f1"]),
    ("fpr_valtau", lambda r: r["test"]["val_tau"]["overall"]["fpr"]),
    ("heavy_fill_rec", lambda r: r["test"]["val_tau"]["heavy_fill"]["rec"]),
    ("foreign_object_rec", lambda r: r["test"]["val_tau"]["foreign_object"]["rec"]),
    ("wrong_class_rec", lambda r: r["test"]["val_tau"]["wrong_class"]["rec"]),
    ("normal_fpr", lambda r: r["test"]["val_tau"]["normal"]["fpr"]),
    ("roc_auc", lambda r: r["test_auc"]["roc_auc"]),
    ("pr_auc", lambda r: r["test_auc"]["pr_auc"]),
    ("val_f1", lambda r: r["val_f1_at_val_tau"]),
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=[0.0, 0.5, 1.0, 2.0, 5.0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--mu-cls", type=float, default=0.0)
    parser.add_argument("--skip-mlp", action="store_true")
    parser.add_argument("--mlp-only", action="store_true",
                        help="Train only the MLP baseline (no gated runs).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mask-modality", choices=["none", "visual", "depth", "mass"],
                        default="none", help="Zero one input at train and test "
                        "time (modality ablation). With explicit expected mass "
                        "in the tuples, zeroing the volume no longer zeroes m_hat.")
    parser.add_argument("--tuples-train", type=Path,
                        default=Path("datasets/multimodal/v3.3/train/tuples.csv"))
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
    parser.add_argument("--output-dir", type=Path,
                        default=Path("experiments/fusion/v3_3/"))
    parser.add_argument("--stats-json", type=Path,
                        default=Path("experiments/fusion/stats_v3_3.json"))
    parser.add_argument("--append", action="store_true",
                        help="Merge runs into an existing stats JSON.")
    parser.add_argument("--narrow-rule-grid", action="store_true",
                        help="Submitted tolerance grid [0.05, 0.5] for the rule.")
    parser.add_argument("--anchor-seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--density-yaml", type=Path,
                        default=Path("configs/material_density.yaml"),
                        help="Density table loaded into the models and used by "
                             "the rule when the tuples carry no explicit expected "
                             "mass. For calibrated tuples pass the SAME dev "
                             "calibration yaml the tuples were generated from.")
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
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU threads per run; part of the reproducibility "
                             "contract (see set_seed).")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    grid = np.linspace(0.05, 0.95, 91)        # sigmoid-score grid (learned)
    rule_grid = rule_tolerance_grid(not args.narrow_rule_grid)

    # ---------------- data (loaded once) ----------------
    ckpt_tag = args.detector_ckpt.stem
    caches = {
        split: load_or_build_roi_cache(
            tuples, args.cache_dir / f"{ckpt_tag}_{split}.pt",
            args.detector_config, args.detector_ckpt, device,
        )
        for split, tuples in (("train", args.tuples_train),
                              ("val", args.tuples_val),
                              ("test", args.tuples_test))
    }
    ds = {
        "train": TuplesDataset(args.tuples_train, caches["train"]),
        "val": TuplesDataset(args.tuples_val, caches["val"]),
        "test": TuplesDataset(args.tuples_test, caches["test"]),
    }
    eval_loaders = {
        split: DataLoader(ds[split], batch_size=1024, shuffle=False,
                          num_workers=args.num_workers, collate_fn=collate_fn)
        for split in ("val", "test")
    }
    print(f"Tuples: train={len(ds['train'])} val={len(ds['val'])} "
          f"test={len(ds['test'])}  (ROI cache tag: {ckpt_tag})")

    cfg = load_density_config(args.density_yaml)
    densities = density_lookup_tensor(cfg["densities"], list(CLASS_NAMES))

    stats: dict[str, Any] = {"runs": {}, "aggregates": {}, "meta": {
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "detector_ckpt": str(args.detector_ckpt),
        "seeds": args.seeds, "lambdas": args.lambdas, "mu_cls": args.mu_cls,
        "anchor_seed": args.anchor_seed, "density_yaml": str(args.density_yaml),
        "tuples": {"train": str(args.tuples_train), "val": str(args.tuples_val),
                   "test": str(args.tuples_test)},
        "model": model_kwargs(args), "threads": args.threads,
        "weight_decay": args.weight_decay, "mask_modality": args.mask_modality,
        "torch": torch.__version__,
    }}
    has_loso = all("expected_weight_loso_g" in open(p).readline()
                   for p in (args.tuples_train, args.tuples_val, args.tuples_test))
    stats["meta"]["expected_mass_from_tuples"] = has_loso
    print(f"expected mass from tuples (leave-one-specimen-out): {has_loso}; "
          f"density table: {args.density_yaml}")
    if args.append and args.stats_json.exists():
        with open(args.stats_json) as f:
            stats = json.load(f)
        stats.setdefault("runs", {})
        stats.setdefault("aggregates", {})

    raw_scores: dict[str, dict[str, Any]] = {}  # run_key -> packs for stats

    # ---------------- rule baseline (deterministic) ----------------
    print("\n=== rule baseline ===")
    val_pack = collect_predictions(None, densities, eval_loaders["val"], device)
    test_pack = collect_predictions(None, densities, eval_loaders["test"], device)
    tau_star, val_f1 = sweep_threshold_on_scores(val_pack[0], val_pack[1],
                                                 rule_grid)
    rule_rec = {
        "val_threshold": tau_star,
        "val_f1_at_val_tau": val_f1,
        "val": {"val_tau": per_type_breakdown(*val_pack, tau_star)},
        "test": {"val_tau": per_type_breakdown(*test_pack, tau_star),
                 # rule has no natural 0.5 operating point; mirror val_tau
                 "fixed_0.5": per_type_breakdown(*test_pack, tau_star)},
        "test_auc": auc_metrics(test_pack[0], test_pack[1]),
        "val_auc": auc_metrics(val_pack[0], val_pack[1]),
    }
    stats["runs"]["rule"] = rule_rec
    raw_scores["rule"] = {"test": test_pack, "tau": tau_star}
    ov = rule_rec["test"]["val_tau"]["overall"]
    print(f"  tau*={tau_star:.3f}  test F1={ov['f1']:.4f}  "
          f"FPR={ov['fpr']:.4f}  ROC-AUC={rule_rec['test_auc']['roc_auc']:.4f}")

    # ---------------- learned runs ----------------
    def run_one(variant: str, lam: float, mu: float, seed: int) -> str:
        run_key = (f"{variant}_lam{lam:g}_mu{mu:g}_s{seed}"
                   if variant == "gated" else f"{variant}_s{seed}")
        run_dir = args.output_dir / run_key
        print(f"\n=== {run_key} ===", flush=True)
        set_seed(seed, args.threads)
        model = build_fusion_model(variant, density_yaml=args.density_yaml, **model_kwargs(args)).to(device)
        train_loader = DataLoader(ds["train"], batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=collate_fn)
        t0 = time.time()
        train_loop(
            model=model,
            train_loader=train_loader,
            val_loader=eval_loaders["val"],
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            lambda_phys=lam,
            output_dir=run_dir,
            mu_cls=mu,
            weight_decay=args.weight_decay,
            mask_modality=args.mask_modality,
        )
        # Reload the best checkpoint for evaluation.
        state = torch.load(run_dir / "best.pth", map_location=device,
                           weights_only=True)
        model.load_state_dict(state["state_dict"])
        model.eval()
        val_pack = collect_predictions(model, None, eval_loaders["val"], device,
                                       mask_modality=args.mask_modality)
        test_pack = collect_predictions(model, None, eval_loaders["test"], device,
                                        mask_modality=args.mask_modality)
        rec = evaluate_scores(val_pack, test_pack, grid)
        rec["train_seconds"] = round(time.time() - t0, 1)
        rec["seed"] = seed
        if variant == "gated":
            rec["lambda"] = lam
            rec["mu_cls"] = mu
        stats["runs"][run_key] = rec
        raw_scores[run_key] = {"test": test_pack, "tau": rec["val_threshold"]}
        ov = rec["test"]["val_tau"]["overall"]
        print(f"  test F1@tau*={ov['f1']:.4f}  FPR={ov['fpr']:.4f}  "
              f"wrong_class={rec['test']['val_tau']['wrong_class']['rec']:.4f}  "
              f"ROC-AUC={rec['test_auc']['roc_auc']:.4f}  "
              f"[{rec['train_seconds']}s]", flush=True)
        return run_key

    if not args.mlp_only:
        for lam in args.lambdas:
            for seed in args.seeds:
                run_one("gated", lam, args.mu_cls, seed)
    if not args.skip_mlp:
        for seed in args.seeds:
            run_one("mlp", 0.0, 0.0, seed)

    # ---------------- aggregates over seeds ----------------
    def aggregate(group_key: str, run_keys: list[str]) -> None:
        recs = [stats["runs"][k] for k in run_keys if k in stats["runs"]]
        if not recs:
            return
        stats["aggregates"][group_key] = {
            name: agg([fn(r) for r in recs]) for name, fn in KEY_METRICS
        }
        stats["aggregates"][group_key]["run_keys"] = run_keys

    for lam in ([] if args.mlp_only else args.lambdas):
        aggregate(
            f"gated_lam{lam:g}_mu{args.mu_cls:g}",
            [f"gated_lam{lam:g}_mu{args.mu_cls:g}_s{s}" for s in args.seeds],
        )
    if not args.skip_mlp:
        aggregate("mlp", [f"mlp_s{s}" for s in args.seeds])

    # ---------------- canonical lambda selection ----------------
    gated_groups = {k: v for k, v in stats["aggregates"].items()
                    if k.startswith("gated_lam")}
    if gated_groups:
        canon_key = max(gated_groups, key=lambda k: gated_groups[k]["val_f1"]["mean"])
        stats["canonical"] = {
            "group": canon_key,
            "selection": "argmax mean val F1 at val-swept threshold",
            "mean_val_f1": gated_groups[canon_key]["val_f1"]["mean"],
        }
        print(f"\nCanonical gated config: {canon_key} "
              f"(mean val F1 = {gated_groups[canon_key]['val_f1']['mean']:.4f})")

        # ---------------- bootstrap CIs + McNemar at anchor seed ---------
        canon_lam = canon_key.split("_lam")[1].split("_mu")[0]
        anchor = {
            "rule": "rule",
            "mlp": f"mlp_s{args.anchor_seed}",
            "gated_lam0": f"gated_lam0_mu{args.mu_cls:g}_s{args.anchor_seed}",
            "gated_canon": (f"gated_lam{canon_lam}_mu{args.mu_cls:g}"
                            f"_s{args.anchor_seed}"),
        }
        stats["bootstrap"] = stats.get("bootstrap", {}) if args.append else {}
        for name, key in anchor.items():
            if key not in raw_scores:
                continue
            score, gt, types = raw_scores[key]["test"]
            tau = raw_scores[key]["tau"]
            stats["bootstrap"][name] = {
                "run_key": key, "threshold": tau,
                **bootstrap_ci(score, gt, types, tau, n_boot=args.n_boot),
            }
            b = stats["bootstrap"][name]
            print(f"  bootstrap {name}: F1 {b['f1']['mean']:.3f} "
                  f"[{b['f1']['lo95']:.3f}, {b['f1']['hi95']:.3f}]  "
                  f"wrong-class {b['wrong_class_rec']['mean']:.3f} "
                  f"[{b['wrong_class_rec']['lo95']:.3f}, "
                  f"{b['wrong_class_rec']['hi95']:.3f}]")

        def correctness(key: str) -> np.ndarray:
            score, gt, _types = raw_scores[key]["test"]
            pred = (score > raw_scores[key]["tau"]).long().numpy()
            return pred == gt.numpy()

        stats["mcnemar"] = stats.get("mcnemar", {}) if args.append else {}
        for name, (ka, kb) in {
            "rule_vs_gated_canon": (anchor["rule"], anchor["gated_canon"]),
            "gated_lam0_vs_gated_canon": (anchor["gated_lam0"],
                                          anchor["gated_canon"]),
            "mlp_vs_gated_canon": (anchor["mlp"], anchor["gated_canon"]),
        }.items():
            if ka in raw_scores and kb in raw_scores:
                stats["mcnemar"][name] = {
                    "a": ka, "b": kb,
                    **mcnemar_exact(correctness(ka), correctness(kb)),
                }
                m = stats["mcnemar"][name]
                print(f"  McNemar {name}: b={m['b']} c={m['c']} "
                      f"p={m['p_value']:.2e}")

    # ---------------- summary table ----------------
    print("\n=== Aggregate summary (test, val-swept threshold) ===")
    hdr = (f"{'config':<26} | {'F1':>13} | {'FPR':>13} | "
           f"{'wrong-class':>13} | {'ROC-AUC':>13}")
    print(hdr); print("-" * len(hdr))
    rows = ["rule"] if "rule" in stats["runs"] else []
    if rows:
        r = stats["runs"]["rule"]
        ov = r["test"]["val_tau"]["overall"]
        wc = r["test"]["val_tau"]["wrong_class"]["rec"]
        print(f"{'rule':<26} | {ov['f1']:>13.4f} | {ov['fpr']:>13.4f} | "
              f"{wc:>13.4f} | {r['test_auc']['roc_auc']:>13.4f}")
    for gkey, a in stats["aggregates"].items():
        if gkey == "run_keys":
            continue
        def ms(name: str) -> str:
            return f"{a[name]['mean']:.3f}±{a[name]['std']:.3f}"
        print(f"{gkey:<26} | {ms('f1_valtau'):>13} | {ms('fpr_valtau'):>13} | "
              f"{ms('wrong_class_rec'):>13} | {ms('roc_auc'):>13}")

    args.stats_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nWrote {args.stats_json}")


if __name__ == "__main__":
    main()
