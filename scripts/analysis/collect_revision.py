"""Collect every number of the revised paper into results/revision/numbers.json."""
from __future__ import annotations

import csv, glob, json, re, statistics as st, sys
from pathlib import Path
import yaml

METRICS = ("f1_valtau", "fpr_valtau", "heavy_fill_rec", "foreign_object_rec", "wrong_class_rec", "roc_auc", "pr_auc", "val_f1")
PARTS = ["", "_rep1", "_rep2", "_rep3"]
R: dict = {}
F = sys.argv[1] if len(sys.argv) > 1 else "experiments/fusion"


def agg_metric(vals: list[float]) -> dict:
    import math
    vals = [float(v) for v in vals if not math.isnan(float(v))]
    if not vals:
        return {"mean": None, "sd": None, "n": 0}
    return {"mean": st.mean(vals), "sd": st.stdev(vals) if len(vals) > 1 else 0.0, "n": len(vals)}


def learned(tag: str, group: str, parts=PARTS, folds=range(4)):
    per_part = []; fold_means = {m: [] for m in METRICS}; seed_sd = []
    for p in parts:
        fm = {m: [] for m in METRICS}
        for k in folds:
            a = json.load(open(f"{F}/v3_4_folds{p}/fold{k}/stats_{tag}.json"))["aggregates"][group]
            for m in METRICS:
                fm[m].append(a[m]["mean"]); fold_means[m].append(a[m]["mean"])
            seed_sd.append(a["f1_valtau"]["std"])
        per_part.append({m: st.mean(fm[m]) for m in METRICS})
    out = {m: {"mean": st.mean(x[m] for x in per_part),
               "part_sd": st.stdev([x[m] for x in per_part]) if len(per_part) > 1 else None,
               "fold_sd": st.stdev(fold_means[m]) if len(fold_means[m]) > 1 else None} for m in METRICS}
    out["seed_sd_f1"] = st.mean(seed_sd); out["n_folds"] = len(fold_means["f1_valtau"])
    return out


def rule_rec(run: dict) -> dict:
    ov = run["test"]["val_tau"]
    return {"f1_valtau": ov["overall"]["f1"], "fpr_valtau": ov["overall"]["fpr"], "heavy_fill_rec": ov["heavy_fill"]["rec"],
            "foreign_object_rec": ov["foreign_object"]["rec"], "wrong_class_rec": ov["wrong_class"]["rec"],
            "roc_auc": run["test_auc"]["roc_auc"], "pr_auc": run["test_auc"]["pr_auc"], "val_f1": run["val_f1_at_val_tau"], "tau": run["val_threshold"]}


def rule(parts=PARTS, folds=range(4)):
    per_part = []; fold_vals = {m: [] for m in METRICS + ("tau",)}
    for p in parts:
        fm = {m: [] for m in METRICS + ("tau",)}
        for k in folds:
            r = rule_rec(json.load(open(f"{F}/v3_4_folds{p}/fold{k}/stats_proj.json"))["runs"]["rule"])
            for m in fm: fm[m].append(r[m]); fold_vals[m].append(r[m])
        per_part.append({m: st.mean(fm[m]) for m in fm})
    return {m: {"mean": st.mean(x[m] for x in per_part), "part_sd": st.stdev(x[m] for x in per_part), "fold_sd": st.stdev(fold_vals[m])} for m in fold_vals} | {"n_folds": len(fold_vals["tau"])}


def paired(tagA, gA, tagB, gB):
    wins = n = 0; deltas = []; folds_pos = 0
    for p in PARTS:
        for k in range(4):
            A = json.load(open(f"{F}/v3_4_folds{p}/fold{k}/stats_{tagA}.json"))["runs"]
            B = json.load(open(f"{F}/v3_4_folds{p}/fold{k}/stats_{tagB}.json"))["runs"]
            d = [A[f"{gA}_s{s}"]["test"]["val_tau"]["overall"]["f1"] - B[f"{gB}_s{s}"]["test"]["val_tau"]["overall"]["f1"] for s in range(42, 52)]
            wins += sum(x > 0 for x in d); n += len(d); deltas += d; folds_pos += st.mean(d) > 0
    return {"wins": wins, "n": n, "mean_delta": st.mean(deltas), "folds_positive": folds_pos, "n_folds": 16}


R["rule"] = rule()
R["heads"] = {
    "mlp": learned("proj", "mlp"), "mlp_nohat": learned("mlp_nohat", "mlp"),
    "proj_nohat": learned("proj_nohat", "gated_lam0_mu0"),
    **{f"proj_lam{l}": learned("proj", f"gated_lam{l}_mu0") for l in ("0", "0.5", "1", "2", "5")},
    "gate_lam0": learned("gate", "gated_lam0_mu0"), "gate_nohat": learned("gate_nohat", "gated_lam0_mu0"),
}
R["canonical"] = "proj_lam5"
R["paired"] = {
    "proj_lam5_vs_lam0": paired("proj", "gated_lam5_mu0", "proj", "gated_lam0_mu0"),
    "proj_lam2_vs_lam0": paired("proj", "gated_lam2_mu0", "proj", "gated_lam0_mu0"),
    "proj_vs_gate": paired("proj", "gated_lam0_mu0", "gate", "gated_lam0_mu0"),
    "proj_hat_vs_nohat": paired("proj", "gated_lam0_mu0", "proj_nohat", "gated_lam0_mu0"),
    "mlp_hat_vs_nohat": paired("proj", "mlp", "mlp_nohat", "mlp"),
    "proj_vs_mlp": paired("proj", "gated_lam0_mu0", "proj", "mlp"),
}
sel = []
for p in PARTS:
    for k in range(4):
        a = json.load(open(f"{F}/v3_4_folds{p}/fold{k}/stats_proj.json"))["aggregates"]
        v = {l: a[f"gated_lam{l}_mu0"]["val_f1"]["mean"] for l in ("0", "0.5", "1", "2", "5")}; sel.append(max(v, key=v.get))
from collections import Counter
R["lambda_selected_by_val"] = dict(Counter(sel))
gaps = []
for p in PARTS:
    for k in range(4):
        d = json.load(open(f"{F}/v3_4_folds{p}/fold{k}/stats_proj.json"))
        gaps.append(d["aggregates"]["gated_lam5_mu0"]["f1_valtau"]["mean"] - d["runs"]["rule"]["test"]["val_tau"]["overall"]["f1"])
R["head_minus_rule_f1"] = {"min": min(gaps), "max": max(gaps), "mean": st.mean(gaps), "folds_positive": sum(g > 0 for g in gaps)}

S = {}
for p in glob.glob(f"{F}/v3_4_sens/fold0/stats_*.json"):
    tag = re.search(r"stats_(.+)\.json", p).group(1); d = json.load(open(p)); r = rule_rec(d["runs"]["rule"]); a = d["aggregates"]["gated_lam0_mu0"]
    S[tag] = {"rule": r, "head": {m: a[m]["mean"] for m in METRICS} | {"f1_sd": a["f1_valtau"]["std"]}}
d = json.load(open(f"{F}/v3_4_folds/fold0/stats_proj.json")); a = d["aggregates"]["gated_lam0_mu0"]
S["specimens"] = {"rule": rule_rec(d["runs"]["rule"]), "head": {m: a[m]["mean"] for m in METRICS} | {"f1_sd": a["f1_valtau"]["std"]}}
R["sensitivity_fold0"] = S

if glob.glob(f"{F}/v3_4_ablation/fold*/stats_*.json"):
    R["ablation"] = {"full": learned("proj", "gated_lam5_mu0", parts=[""])}
    for mod in ("visual", "depth", "mass"):
        fm = {m: [] for m in METRICS}
        for k in range(4):
            a = json.load(open(f"{F}/v3_4_ablation/fold{k}/stats_{mod}.json"))["aggregates"]["gated_lam5_mu0"]
            for m in METRICS: fm[m].append(a[m]["mean"])
        R["ablation"][mod] = {m: {"mean": st.mean(fm[m]), "fold_sd": st.stdev(fm[m])} for m in METRICS}
    R["ablation_paired"] = {}
    for mod in ("visual", "depth", "mass"):
        wins = {m: 0 for m in ("f1", "heavy_fill", "foreign_object", "wrong_class")}; deltas = {m: [] for m in wins}
        for k in range(4):
            A = json.load(open(f"{F}/v3_4_folds/fold{k}/stats_proj.json"))["runs"]
            B = json.load(open(f"{F}/v3_4_ablation/fold{k}/stats_{mod}.json"))["runs"]
            for s in range(42, 52):
                a = A[f"gated_lam5_mu0_s{s}"]["test"]["val_tau"]; b = B[f"gated_lam5_mu0_s{s}"]["test"]["val_tau"]
                for m in wins:
                    d = (a["overall"]["f1"] - b["overall"]["f1"]) if m == "f1" else (a[m]["rec"] - b[m]["rec"])
                    deltas[m].append(d); wins[m] += d > 0
        R["ablation_paired"][mod] = {m: {"wins": wins[m], "n": len(deltas[m]), "mean_delta": st.mean(deltas[m])} for m in wins}
if glob.glob(f"{F}/v3_4_noise/fold*/stats_*.json"):
    N = {}
    for sw in ("0.02", "0.05", "0.10"):
        for sd in ("0.05", "0.10", "0.20"):
            f1 = []; rf = []
            for k in range(4):
                p = f"{F}/v3_4_noise/fold{k}/stats_sw{sw}_sd{sd}.json"
                if not Path(p).exists(): continue
                d = json.load(open(p)); f1.append(d["aggregates"]["gated_lam5_mu0"]["f1_valtau"]["mean"]); rf.append(d["runs"]["rule"]["test"]["val_tau"]["overall"]["f1"])
            if f1: N[f"sw{sw}_sd{sd}"] = {"head_f1": st.mean(f1), "head_fold_sd": st.stdev(f1) if len(f1) > 1 else 0, "rule_f1": st.mean(rf), "n_folds": len(f1)}
    R["noise_grid"] = N

for name in ("beachlitter", "real495"):
    p = f"{F}/v3_4_ood/{name}.json"
    if Path(p).exists():
        d = json.load(open(p)); out = {}
        for k, v in d["folds"].items():
            s = v["seeds"]
            nm = lambda key: agg_metric([x[key]["rec"] for x in s.values()])["mean"]
            out[k] = {"rule": {"f1": v["rule"]["overall"]["f1"], "fpr": v["rule"]["normal"]["fpr"], "heavy": v["rule"]["heavy_fill"]["rec"], "foreign": v["rule"]["foreign_object"]["rec"], "wrong": v["rule"]["wrong_class"]["rec"], "tau": v["rule"]["tau"]},
                      "head": {"f1": st.mean(x["overall"]["f1"] for x in s.values()), "f1_sd": st.stdev(x["overall"]["f1"] for x in s.values()), "fpr": st.mean(x["normal"]["fpr"] for x in s.values()),
                               "heavy": nm("heavy_fill"), "foreign": nm("foreign_object"), "wrong": nm("wrong_class"), "auc": st.mean(x["roc_auc"] for x in s.values())}}
        agg = {side: {m: agg_metric([out[k][side][m] for k in out if out[k][side][m] is not None]) for m in out["0"][side]} for side in ("rule", "head")}
        R[f"frozen_{name}"] = {"per_fold": out, "mean_over_folds": agg}
if Path(f"{F}/v3_4_ood/real495_rocs.json").exists():
    R["frozen_real495"]["head_rocs"] = json.load(open(f"{F}/v3_4_ood/real495_rocs.json"))

G = {}
for name, path, group in (("submitted_raw", f"{F}/v3_3_gate_pinned", "gated_lam0_mu0"), ("sigma0_log", f"{F}/v3_4_gate_sigma0", "gated_lam0_mu0")):
    g = json.load(open(f"{path}/stats_gate.json")); n = json.load(open(f"{path}/stats_nogate.json"))
    rec = {"gate": {m: g["aggregates"][group][m]["mean"] for m in METRICS} | {"f1_sd": g["aggregates"][group]["f1_valtau"]["std"]},
           "nogate": {m: n["aggregates"][group][m]["mean"] for m in METRICS} | {"f1_sd": n["aggregates"][group]["f1_valtau"]["std"]},
           "gate_wins": sum(g["runs"][f"{group}_s{s}"]["test"]["val_tau"]["overall"]["f1"] > n["runs"][f"{group}_s{s}"]["test"]["val_tau"]["overall"]["f1"] for s in range(42, 52))}
    if "mlp" in g["aggregates"]: rec["mlp"] = {m: g["aggregates"]["mlp"][m]["mean"] for m in METRICS} | {"f1_sd": g["aggregates"]["mlp"]["f1_valtau"]["std"]}
    G[name] = rec
H = {}
for p in glob.glob(f"{F}/v3_4_hparam/stats_*.json"):
    m = re.search(r"stats_(gate|nogate)_lr([\d.e-]+)_ep(\d+)_dr([\d.]+)_wd([\d.e-]+)\.json", p); head, lr, ep, dr, wd = m.groups()
    a = json.load(open(p))["aggregates"]["gated_lam0_mu0"]; H[(head, lr, ep, dr, wd)] = (a["val_f1"]["mean"], a["f1_valtau"]["mean"])
pts = [k[1:] for k in H if k[0] == "gate"]
G["hparam_grid"] = {"n_points": len(pts),
                    "gate_ahead_val": sum(H[("gate",) + k][0] > H[("nogate",) + k][0] for k in pts),
                    "gate_ahead_test": sum(H[("gate",) + k][1] > H[("nogate",) + k][1] for k in pts),
                    "best_by_val": {h: {"point": max((k for k in H if k[0] == h), key=lambda k: H[k][0])[1:], "val": max(H[k][0] for k in H if k[0] == h),
                                        "test": H[max((k for k in H if k[0] == h), key=lambda k: H[k][0])][1]} for h in ("gate", "nogate")}}
R["gate_controls"] = G

if Path(f"{F}/v3_4_rawctl/fold3/stats_gate_raw.json").exists():
    RC = {}
    def _f1(d, grp, s): return d["runs"][f"{grp}_s{s}"]["test"]["val_tau"]["overall"]["f1"]
    for name, ltag, rtag, grp in (("mlp", "proj", "proj_raw", "mlp"), ("proj", "proj", "proj_raw", "gated_lam0_mu0"), ("gate", "gate", "gate_raw", "gated_lam0_mu0")):
        Lg = [json.load(open(f"{F}/v3_4_folds/fold{k}/stats_{ltag}.json")) for k in range(4)]
        Rw = [json.load(open(f"{F}/v3_4_rawctl/fold{k}/stats_{rtag}.json")) for k in range(4)]
        RC[name] = {"log": st.mean(d["aggregates"][grp]["f1_valtau"]["mean"] for d in Lg),
                    "raw": st.mean(d["aggregates"][grp]["f1_valtau"]["mean"] for d in Rw),
                    "log_wins": sum(_f1(l, grp, s) > _f1(w, grp, s) for l, w in zip(Lg, Rw) for s in range(42, 52)), "n": 40}
    Gr = [json.load(open(f"{F}/v3_4_rawctl/fold{k}/stats_gate_raw.json")) for k in range(4)]
    Pr = [json.load(open(f"{F}/v3_4_rawctl/fold{k}/stats_proj_raw.json")) for k in range(4)]
    RC["raw_gate_wins"] = sum(_f1(g, "gated_lam0_mu0", s) > _f1(p, "gated_lam0_mu0", s) for g, p in zip(Gr, Pr) for s in range(42, 52))
    if Path("results/revision/raw_control_gates.json").exists():
        RC["gates"] = json.load(open("results/revision/raw_control_gates.json"))
    R["raw_control"] = RC

cal = yaml.safe_load(open("configs/calibration_packphys.yaml"))
fills = sorted(s["fill_ratio"] for L in cal["specimens"].values() for s in L if s["fill_ratio"])
R["calibration"] = {"n_specimens": cal["n_specimens"], "densities": cal["densities"], "density_sigma": cal["density_sigma"],
                    "volume_scale": cal["volume_scale"], "volume_sigma": cal["volume_sigma"],
                    "n_per_class": {c: len(L) for c, L in cal["specimens"].items()}, "n_fill_per_class": {c: sum(1 for s in L if s["fill_ratio"]) for c, L in cal["specimens"].items()},
                    "fill_ratio": {"n": len(fills), "min": fills[0], "q10": fills[len(fills) // 10], "median": st.median(fills), "q90": fills[9 * len(fills) // 10], "max": fills[-1]}}
man = json.load(open("datasets/calibration/folds/manifest.json"))
R["folds"] = {"k": man["k"], "n_partitions": len(PARTS), "test_sizes": [len(f["test_specimens"]) for f in man["folds"]], "dev_sizes": [len(f["dev_specimens"]) for f in man["folds"]],
              "test_per_class_min": min(min(f["test_per_class"].values()) for f in man["folds"]), "test_per_class_max": max(max(f["test_per_class"].values()) for f in man["folds"])}
sizes = {}
for sp in ("train", "val", "test"):
    n = [sum(1 for _ in csv.DictReader(open(f"datasets/multimodal/v3.4_folds/fold{k}/{sp}/tuples.csv"))) for k in range(4)]
    sizes[sp] = {"min": min(n), "max": max(n), "mean": st.mean(n)}
old = {sp: sum(1 for _ in csv.DictReader(open(f"datasets/multimodal/v3.3/{sp}/tuples.csv"))) for sp in ("train", "val", "test")}
R["tuples"] = {"folds": sizes, "submitted": old, "dropped_small_pct": 100 * (1 - sum(sizes[s]["mean"] for s in sizes) / sum(old.values()))}
rows = list(csv.DictReader(open("datasets/real495/measurements_specimens.csv")))
R["real495"] = {"n_rows": len(rows), "n_images": len({r["filename"] for r in rows}), "n_specimens": len({r["specimen_id"] for r in rows}),
                "conditions": dict(Counter(r["condition"] for r in rows)), "measured_eval": json.load(open("results/real495/measured_eval.json"))["protocols"]}
for det in ("htc_x101", "htc_r50"):
    p = f"results/real495/safety_net_revised_{det}.json"
    if Path(p).exists(): R["real495"][f"safety_net_{det}"] = json.load(open(p))
Path("results/revision").mkdir(parents=True, exist_ok=True)
json.dump(R, open("results/revision/numbers.json", "w"), indent=1)
print("wrote results/revision/numbers.json;", "sections:", list(R.keys()))
