"""Generate every LaTeX table and number macro of the revised paper from the stored results."""
from __future__ import annotations

import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--numbers", type=Path, default=Path("results/revision/numbers.json"))
ap.add_argument("--manuscript", type=Path, default=Path("results/revision"))
a = ap.parse_args()
R = json.load(open(a.numbers)); H = R["heads"]; RU = R["rule"]; T = a.manuscript / "tables"
M = ("f1_valtau", "fpr_valtau", "heavy_fill_rec", "foreign_object_rec", "wrong_class_rec", "roc_auc", "pr_auc")
P = R["paired"]
e = R["real495"]["measured_eval"]["calibration_89_specimens"]
FR = R["frozen_real495"]["mean_over_folds"]
B = R["frozen_beachlitter"]["mean_over_folds"]
_rocs = R["frozen_real495"]["head_rocs"]
hr = {k: sum(v[k] for v in _rocs.values()) / len(_rocs) for k in ("auc", "r5", "r10", "r20")}


def f3(x): return f"{x:.3f}"
def cell(h, m): return f"{h[m]['mean']:.3f}\\,({h[m]['part_sd']:.3f})"
def rcell(m): return f"{RU[m]['mean']:.3f}\\,({RU[m]['part_sd']:.3f})"
def best(keys, m, lower=False):
    vals = {k: (RU if k == "rule" else H[k])[m]["mean"] for k in keys}
    return min(vals, key=vals.get) if lower else max(vals, key=vals.get)
def bold(k, b, txt): return f"\\textbf{{{txt}}}" if k == b else txt
def pw(k): return f"{P[k]['wins']}/{P[k]['n']}"


rows = [("rule", "Density check ($\\tau$ selected on validation)"),
        ("mlp", "MLP"),
        ("proj_lam0", "Projection head, $\\lambda{=}0$"),
        ("gate_lam0", "Projection head + gates, Eq.~\\eqref{eq:gated}, $\\lambda{=}0$"),
        ("proj_lam0.5", "Projection head, $\\lambda{=}0.5$"),
        ("proj_lam1", "Projection head, $\\lambda{=}1$"),
        ("proj_lam2", "Projection head, $\\lambda{=}2$"),
        ("proj_lam5", "Projection head, $\\lambda{=}5$$^{a}$")]
keys = [k for k, _ in rows]
b = {m: best(keys, m, lower=(m == "fpr_valtau")) for m in ("f1_valtau", "fpr_valtau", "roc_auc", "pr_auc")}
lines = []
for k, name in rows:
    c = [rcell(m) if k == "rule" else cell(H[k], m) for m in ("f1_valtau", "fpr_valtau", "roc_auc", "pr_auc")]
    lines.append(f"{name} & " + " & ".join(bold(k, b[m], x) for m, x in zip(("f1_valtau", "fpr_valtau", "roc_auc", "pr_auc"), c)) + " \\\\")
(T / "tab3_fusion.tex").write_text(r"""\begin{table*}[tb]
\centering
\caption{\revcap{Fusion variants on the test tuples of the specimen-disjoint
protocol: mean over the 16 folds, standard deviation across partitions in
parentheses. Best per column in bold.}}
\label{tab:fusion}
\begin{tabular}{lrrrr}
\toprule
Variant & F1 & FPR & ROC-AUC & PR-AUC \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\multicolumn{5}{l}{\footnotesize $^{a}$Default setting; validation selects $\lambda{=}5$ in \nLamFive{} of the 16 folds and $\lambda{=}2$ in the others.}
\end{tabular}
\end{table*}
""")

PF = ("fpr_valtau", "heavy_fill_rec", "foreign_object_rec", "wrong_class_rec")
learned = [k for k in keys if k != "rule"]
b4 = {m: best(learned, m, lower=(m == "fpr_valtau")) for m in PF}
lines = []
for k, name in rows:
    c = [rcell(m) if k == "rule" else cell(H[k], m) for m in PF]
    lines.append(f"{name} & " + " & ".join(bold(k, b4[m], x) for m, x in zip(PF, c)) + " \\\\")
(T / "tab4_perfraud.tex").write_text(r"""\begin{table*}[tb]
\centering
\caption{\revcap{Recall by generated fraud scenario and false-positive rate on
legitimate items. Best per column among the learned variants in bold.}}
\label{tab:perfraud}
\begin{tabular}{lrrrr}
\toprule
Variant & FPR, legitimate $\downarrow$ & heavy-fill $\uparrow$ & foreign-object $\uparrow$ & wrong-class $\uparrow$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

abl_rows = [("MLP", "without", "mlp_nohat"), ("MLP", "with", "mlp"),
            ("projections", "without", "proj_nohat"), ("projections", "with", "proj_lam0"),
            ("projections + gates", "without", "gate_nohat"), ("projections + gates", "with", "gate_lam0")]
lines = [f"{head} & {inp} & {cell(H[k], 'f1_valtau')} & {cell(H[k], 'fpr_valtau')} & "
         + " & ".join(f3(H[k][m]["mean"]) for m in ("heavy_fill_rec", "foreign_object_rec", "wrong_class_rec")) + " \\\\"
         for head, inp, k in abl_rows]
(T / "tab_ablation.tex").write_text(r"""\begin{table*}[tb]
\centering
\caption{\revcap{Contribution of the density-derived input $\hat{m}$ and of
the gates ($\lambda=0$). Each row trains the head from scratch with the
stated inputs.}}
\label{tab:ablation}
\begin{tabular}{lcrrrrr}
\toprule
Head & $\hat{m}$ input & F1 & FPR & heavy-fill & foreign-object & wrong-class \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

S = R["sensitivity_fold0"]
SM = ("f1_valtau", "fpr_valtau", "heavy_fill_rec", "foreign_object_rec", "wrong_class_rec")
def srow(label, key):
    v = S[key]
    return f"{label} & " + " & ".join(f3(v["rule"][m]) for m in SM) + " & " + " & ".join(f3(v["head"][m]) for m in SM) + " \\\\"
top = [srow("$U(2.5,6)$ / $U(1.8,3.5)$ (heuristic)", "mult_submitted"),
       srow("measured fill ratios (default)", "specimens"),
       srow("$U(1.5,3)$ / $U(1.3,2)$", "mult_moderate"),
       srow("$U(1.2,2.5)$ / $U(1.1,1.8)$", "mult_overlap"),
       srow("$U(1.05,1.5)$ / $U(1.05,1.3)$", "mult_strong")]
bottom = [srow("no density variation ($\\sigma=0$)", "spread0")] + [srow(f"$\\sigma={s}$", f"spread{s}") for s in ("0.1", "0.2", "0.3", "0.5")]
(T / "tab_sensitivity.tex").write_text(r"""\begin{table*}[tb]
\centering
\caption{\revcap{Sensitivity on fold 0 of the main partition (projection head,
$\lambda=0$). Top: ranges of the fraud mass multipliers; bottom:
log-normal density draws of width $\sigma$ in place of real specimens.}}
\label{tab:sensitivity}
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\begin{tabular}{lrrrrrrrrrr}
\toprule
 & \multicolumn{5}{c}{Density check} & \multicolumn{5}{c}{Projection head} \\
\cmidrule(lr){2-6}\cmidrule(lr){7-11}
Setting & F1 & FPR & heavy & for. & wrong & F1 & FPR & heavy & for. & wrong \\
\midrule
""" + "\n".join(top) + r"""
\midrule
""" + "\n".join(bottom) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

OM = ("f1_valtau", "fpr_valtau", "heavy_fill_rec", "foreign_object_rec", "wrong_class_rec")
BM = ("f1", "fpr", "heavy", "foreign", "wrong")
def bcell(g, m): return f"{B[g][m]['mean']:.3f}\\,({B[g][m]['sd']:.3f})"
(T / "tab8_ood.tex").write_text(r"""\begin{table*}[tb]
\centering
\caption{\revcap{Density check and projection head in domain and on BeachLitter
tuples at thresholds frozen on in-domain validation. Standard deviation in
parentheses: across partitions in domain, across folds on BeachLitter.}}
\label{tab:oodfusion}
\begin{tabular}{lrrrrr}
\toprule
Variant & F1 & FPR & heavy-fill $\uparrow$ & foreign-object $\uparrow$ & wrong-class $\uparrow$ \\
\midrule
\multicolumn{6}{@{}l}{\emph{In-domain (Hybrid-v3.3 test, 16 folds)}} \\
Density check & """ + " & ".join(rcell(m) for m in OM) + r""" \\
Projection head ($\lambda{=}5$) & """ + " & ".join(cell(H["proj_lam5"], m) for m in OM) + r""" \\
\midrule
\multicolumn{6}{@{}l}{\emph{Unseen source (BeachLitter, frozen thresholds, 4 folds)}} \\
Density check & """ + " & ".join(bcell("rule", m) for m in BM) + r""" \\
Projection head ($\lambda{=}5$) & """ + " & ".join(bcell("head", m) for m in BM) + r""" \\
\bottomrule
\end{tabular}
\end{table*}
""")

(T / "tab_measured.tex").write_text(r"""\begin{table}[!tb]
\centering
\caption{\revcap{Density check and frozen projection heads on the 495 real495
objects with measured mass and volume: ROC-AUC and recall of the tampered
objects at fixed false-positive rates on the intact objects.}}
\label{tab:measured}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrr}
\toprule
 & ROC-AUC & \multicolumn{3}{c}{recall at FPR 5\,/\,10\,/\,20\%} \\
\midrule
Density check & """ + " & ".join(f3(e[k]) for k in ("auc", "recall_at_fpr_5", "recall_at_fpr_10", "recall_at_fpr_20")) + r""" \\
Projection head (frozen) & """ + " & ".join(f3(hr[k]) for k in ("auc", "r5", "r10", "r20")) + r""" \\
\bottomrule
\end{tabular}
\end{table}
""")

A = R["ablation"]
AM = ("f1_valtau", "fpr_valtau", "heavy_fill_rec", "foreign_object_rec", "wrong_class_rec")
mod_rows = [("none (full)", "full"), ("visual (RGB)", "visual"),
            ("volume (box proxy)", "depth"), ("mass", "mass")]
_best = {m: (min if m == "fpr_valtau" else max)(f3(A[k][m]["mean"]) for _, k in mod_rows) for m in AM}
lines = [f"{label} & " + " & ".join((f"\\textbf{{{f3(A[k][m]['mean'])}}}" if f3(A[k][m]["mean"]) == _best[m] else f3(A[k][m]["mean"])) for m in AM) + " \\\\"
         for label, k in mod_rows]
(T / "tab6_modality.tex").write_text(r"""\begin{table*}[tb]
\centering
\caption{\revcap{Modality ablation (projection head, $\lambda=5$, main
partition): each row zeroes one input and retrains the head. Best per
column in bold.}}
\label{tab:modality}
\begin{tabular}{lrrrrr}
\toprule
Input removed & Test F1 $\uparrow$ & Test FPR $\downarrow$ & heavy-fill rec $\uparrow$ & foreign-object rec $\uparrow$ & wrong-class rec $\uparrow$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

N = R["noise_grid"]
SW, SD = ("0.02", "0.05", "0.10"), ("0.05", "0.10", "0.20")
rows9 = [f"{sw} & " + " & ".join(f3(N[f'sw{sw}_sd{sd}']["head_f1"]) + ("$^\\star$" if (sw, sd) == ("0.05", "0.10") else "") for sd in SD) + " \\\\"
         for sw in SW]
(T / "tab7_sensor.tex").write_text(r"""\begin{table}[!tb]
\centering
\caption{\revcap{Test F1 of the projection head ($\lambda=5$, main partition)
over mass noise $\sigma_w$ and volume-proxy log-noise $\sigma_d$;
$\star$: the mass noise of the main protocol.}}
\label{tab:sensor}
\begin{tabular}{lrrr}
\toprule
$\sigma_w$ \textbackslash\ $\sigma_d$ & 0.05 & 0.10 & 0.20 \\
\midrule
""" + "\n".join(rows9) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
def mac(name, val): return f"\\newcommand{{\\{name}}}{{{val}}}"
L = [mac("nRuleF", f3(RU["f1_valtau"]["mean"])), mac("nRuleFPR", f3(RU["fpr_valtau"]["mean"])), mac("nRuleWC", f3(RU["wrong_class_rec"]["mean"])),
     mac("nRuleHF", f3(RU["heavy_fill_rec"]["mean"])), mac("nRuleFO", f3(RU["foreign_object_rec"]["mean"])), mac("nRuleAUC", f3(RU["roc_auc"]["mean"])),
     mac("nHeadF", f3(H["proj_lam5"]["f1_valtau"]["mean"])), mac("nHeadFsd", f3(H["proj_lam5"]["f1_valtau"]["part_sd"])), mac("nHeadFPR", f3(H["proj_lam5"]["fpr_valtau"]["mean"])),
     mac("nHeadHF", f3(H["proj_lam5"]["heavy_fill_rec"]["mean"])), mac("nHeadFO", f3(H["proj_lam5"]["foreign_object_rec"]["mean"])), mac("nHeadWC", f3(H["proj_lam5"]["wrong_class_rec"]["mean"])), mac("nHeadAUC", f3(H["proj_lam5"]["roc_auc"]["mean"])),
     mac("nHeadLamZeroF", f3(H["proj_lam0"]["f1_valtau"]["mean"])), mac("nMlpF", f3(H["mlp"]["f1_valtau"]["mean"])), mac("nMlpNohatF", f3(H["mlp_nohat"]["f1_valtau"]["mean"])),
     mac("nProjNohatF", f3(H["proj_nohat"]["f1_valtau"]["mean"])), mac("nGateF", f3(H["gate_lam0"]["f1_valtau"]["mean"])), mac("nGateNohatF", f3(H["gate_nohat"]["f1_valtau"]["mean"])),
     mac("nSeedSd", f3(H["proj_lam5"]["seed_sd_f1"])), mac("nGateSeedSd", f3(H["gate_lam0"]["seed_sd_f1"])),
     mac("nGapMin", f"{R['head_minus_rule_f1']['min']:.2f}"), mac("nGapMax", f"{R['head_minus_rule_f1']['max']:.2f}"), mac("nGapMean", f"{R['head_minus_rule_f1']['mean']:.2f}"),
     mac("nWinsLam", pw("proj_lam5_vs_lam0")), mac("nDeltaLam", f"{P['proj_lam5_vs_lam0']['mean_delta']:+.3f}"), mac("nFoldsLam", str(P["proj_lam5_vs_lam0"]["folds_positive"])),
     mac("nWinsHat", pw("proj_hat_vs_nohat")), mac("nDeltaHat", f"{P['proj_hat_vs_nohat']['mean_delta']:+.3f}"), mac("nWinsHatMlp", pw("mlp_hat_vs_nohat")), mac("nDeltaHatMlp", f"{P['mlp_hat_vs_nohat']['mean_delta']:+.3f}"),
     mac("nWinsGate", pw("proj_vs_gate")), mac("nDeltaGate", f"{-P['proj_vs_gate']['mean_delta']:+.3f}"), mac("nFoldsGate", str(P["proj_vs_gate"]["folds_positive"])),
     mac("nWinsArch", pw("proj_vs_mlp")), mac("nDeltaArch", f"{P['proj_vs_mlp']['mean_delta']:+.3f}"), mac("nFoldsArch", str(P["proj_vs_mlp"]["folds_positive"])),
     mac("nLamFive", str(R["lambda_selected_by_val"].get("5", 0))), mac("nLamTwo", str(R["lambda_selected_by_val"].get("2", 0))),
     mac("nCalibN", str(R["calibration"]["n_specimens"])), mac("nFillN", str(R["calibration"]["fill_ratio"]["n"])), mac("nFillMed", f"{R['calibration']['fill_ratio']['median']:.0f}"),
     mac("nFillMin", f"{R['calibration']['fill_ratio']['min']:.1f}"), mac("nFillMax", f"{R['calibration']['fill_ratio']['max']:.0f}"),
     mac("nRealSpec", str(R["real495"]["n_specimens"])), mac("nDroppedPct", f"{R['tuples']['dropped_small_pct']:.1f}"),
     mac("nMeasAUC", f3(e["auc"])), mac("nMeasAUClo", f3(e["auc_ci95"][0])), mac("nMeasAUChi", f3(e["auc_ci95"][1])), mac("nMeasRecTen", f3(e["recall_at_fpr_10"])), mac("nMeasFPRtau", f"{e['fpr_at_tau_0.25']:.2f}"),
     mac("nMeasHeadAUC", f3(hr["auc"])), mac("nMeasHeadRecTen", f3(hr["r10"])), mac("nMeasHeadRecTwenty", f3(hr["r20"])), mac("nMeasHeadFPRfrozen", f3(FR["head"]["fpr"]["mean"])), mac("nMeasRuleFPRfrozen", f3(FR["rule"]["fpr"]["mean"])),
     mac("nOodRuleF", f3(B["rule"]["f1"]["mean"])), mac("nOodRuleFsd", f3(B["rule"]["f1"]["sd"])), mac("nOodHeadF", f3(B["head"]["f1"]["mean"])), mac("nOodHeadFsd", f3(B["head"]["f1"]["sd"])), mac("nOodHeadFPR", f3(B["head"]["fpr"]["mean"])), mac("nOodRuleFPR", f3(B["rule"]["fpr"]["mean"])), mac("nOodRuleFPRsd", f3(B["rule"]["fpr"]["sd"])), mac("nOodHeadAUC", f3(B["head"]["auc"]["mean"])), mac("nOodHeadWC", f3(B["head"]["wrong"]["mean"])), mac("nOodRuleWC", f3(B["rule"]["wrong"]["mean"]))]
A = R["ablation"]
L += [mac("nAblFull", f3(A["full"]["f1_valtau"]["mean"])), mac("nAblNoVis", f3(A["visual"]["f1_valtau"]["mean"])), mac("nAblNoVol", f3(A["depth"]["f1_valtau"]["mean"])), mac("nAblNoMass", f3(A["mass"]["f1_valtau"]["mean"])),
      mac("nAblFullWC", f3(A["full"]["wrong_class_rec"]["mean"])), mac("nAblNoVisWC", f3(A["visual"]["wrong_class_rec"]["mean"])), mac("nAblNoVolWC", f3(A["depth"]["wrong_class_rec"]["mean"])), mac("nAblNoMassWC", f3(A["mass"]["wrong_class_rec"]["mean"])),
      mac("nAblFullHF", f3(A["full"]["heavy_fill_rec"]["mean"])), mac("nAblNoMassHF", f3(A["mass"]["heavy_fill_rec"]["mean"])), mac("nAblNoMassFO", f3(A["mass"]["foreign_object_rec"]["mean"])), mac("nAblFullFO", f3(A["full"]["foreign_object_rec"]["mean"])), mac("nAblNoVolFO", f3(A["depth"]["foreign_object_rec"]["mean"])), mac("nAblNoVolHF", f3(A["depth"]["heavy_fill_rec"]["mean"])),
      mac("nAblNoVisFPR", f3(A["visual"]["fpr_valtau"]["mean"])), mac("nAblFullFPR", f3(A["full"]["fpr_valtau"]["mean"])), mac("nAblNoVisHF", f3(A["visual"]["heavy_fill_rec"]["mean"])), mac("nAblNoVisFO", f3(A["visual"]["foreign_object_rec"]["mean"])), mac("nAblNoMassFPR", f3(A["mass"]["fpr_valtau"]["mean"]))]
AP = R.get("ablation_paired")
if AP:
    dp = AP["depth"]
    L += [mac("nAblVolWins", f"{dp['f1']['wins']}/{dp['f1']['n']}"), mac("nAblVolDeltaF", f"{dp['f1']['mean_delta']:+.3f}"),
          mac("nAblVolWinsWC", f"{dp['wrong_class']['wins']}/{dp['wrong_class']['n']}"), mac("nAblVolDeltaWC", f"{dp['wrong_class']['mean_delta']:+.3f}"),
          mac("nAblVolWinsHF", f"{dp['heavy_fill']['wins']}/{dp['heavy_fill']['n']}")]
N = R["noise_grid"]; nf = [v["head_f1"] for v in N.values()]
L += [mac("nNoiseMin", f3(min(nf))), mac("nNoiseMax", f3(max(nf))), mac("nNoiseCorner", f3(N["sw0.10_sd0.20"]["head_f1"])), mac("nNoiseRuleMin", f3(min(v["rule_f1"] for v in N.values()))), mac("nNoiseRuleMax", f3(max(v["rule_f1"] for v in N.values())))]
G = R["gate_controls"]
L += [mac("nCtrlRawGate", f3(G["submitted_raw"]["gate"]["f1_valtau"])), mac("nCtrlRawGateSd", f3(G["submitted_raw"]["gate"]["f1_sd"])), mac("nCtrlRawNogate", f3(G["submitted_raw"]["nogate"]["f1_valtau"])), mac("nCtrlRawNogateSd", f3(G["submitted_raw"]["nogate"]["f1_sd"])), mac("nCtrlRawMlp", f3(G["submitted_raw"]["mlp"]["f1_valtau"])), mac("nCtrlRawWins", str(G["submitted_raw"]["gate_wins"])),
      mac("nCtrlSzGate", f3(G["sigma0_log"]["gate"]["f1_valtau"])), mac("nCtrlSzNogate", f3(G["sigma0_log"]["nogate"]["f1_valtau"])), mac("nCtrlSzWins", str(G["sigma0_log"]["gate_wins"])),
      mac("nHpPoints", str(G["hparam_grid"]["n_points"])), mac("nHpGateVal", str(G["hparam_grid"]["gate_ahead_val"])), mac("nHpGateTest", str(G["hparam_grid"]["gate_ahead_test"])), mac("nHpBestGate", f3(G["hparam_grid"]["best_by_val"]["gate"]["test"])), mac("nHpBestNogate", f3(G["hparam_grid"]["best_by_val"]["nogate"]["test"]))]
RC = R.get("raw_control")
if RC:
    L += [mac("nRawMlp", f3(RC["mlp"]["raw"])), mac("nRawProj", f3(RC["proj"]["raw"])), mac("nRawGate", f3(RC["gate"]["raw"])),
          mac("nLogMlp", f3(RC["mlp"]["log"])), mac("nLogProj", f3(RC["proj"]["log"])), mac("nLogGate", f3(RC["gate"]["log"])),
          mac("nRawLogWinsMin", str(min(RC[k]["log_wins"] for k in ("mlp", "proj", "gate")))), mac("nRawPairs", str(RC["mlp"]["n"])),
          mac("nRawGateWins", str(RC["raw_gate_wins"])), mac("nRawGateVol", f"{RC['gates']['raw']['volume']:.2f}"),
          mac("nRawProjOverMlp", f3(RC["proj"]["raw"] - RC["mlp"]["raw"])), mac("nRawGateOverProj", f3(RC["gate"]["raw"] - RC["proj"]["raw"])),
          *(lambda F: [mac("nOodRuleFPRmin", f"{min(v['rule']['fpr'] for v in F):.2f}"), mac("nOodRuleFPRmax", f"{max(v['rule']['fpr'] for v in F):.2f}"),
                       mac("nOodRuleTauAtMax", f"{max(F, key=lambda v: v['rule']['fpr'])['rule']['tau']:.2f}"),
                       mac("nOodRuleTauMin", f"{min(v['rule']['tau'] for v in F):.2f}")])(list(R["frozen_beachlitter"]["per_fold"].values()))]
for det in ("htc_x101", "htc_r50"):
    sn = R["real495"][f"safety_net_{det}"]; tag = "X" if det == "htc_x101" else "R"
    L += [mac(f"nSn{tag}Loc", str(sn["n_localised"])), mac(f"nSn{tag}Cor", str(sn["n_correct"])), mac(f"nSn{tag}Conf", str(sn["n_confused"])), mac(f"nSn{tag}Spec", str(sn["confused_specimens"])),
          mac(f"nSn{tag}AUC", f3(sn["auc_confused_vs_correct"])), mac(f"nSn{tag}FlagTau", str(sn["at_tau_0.25"]["confusions_flagged"])), mac(f"nSn{tag}RejTau", str(sn["at_tau_0.25"]["correct_rejected"])),
          mac(f"nSn{tag}FlagTen", str(sn["at_false_reject_10"]["confusions_flagged"])), mac(f"nSn{tag}FlagTwenty", str(sn["at_false_reject_20"]["confusions_flagged"]))]
(T / "macros_numbers.tex").write_text("% Generated by scripts/analysis/make_revision_tables.py from results/revision/numbers.json. Do not edit.\n" + "\n".join(L) + "\n")
print("wrote", len(L), "macros and 8 tables to", T)

import csv as _csv, yaml as _yaml
CL = ["plastic_bottle", "glass_bottle", "glass_jar", "metal_can", "carton", "tetrapack", "paper", "plastic_bag", "plastic_container", "other"]
def thou(n, sep="{,}"): return f"{n:,}".replace(",", sep)

cal = _yaml.safe_load(open("configs/calibration_packphys.yaml"))
MAT = dict(zip(CL, ["PET", "Glass", "Glass", "Aluminium", "Cardboard", "Composite", "Paper", "LDPE", "HDPE", "Mixed"]))
rows0 = "\n".join(f"\\texttt{{{c.replace('_', chr(92) + '_')}}} & {MAT[c]} & {cal['densities'][c]:.3f} & {len(cal['specimens'][c])} & {cal['density_sigma'][c]:.2f} \\\\" for c in CL)
(T / "tab0_densities.tex").write_text(f"""\\begin{{table}}[!tb]
\\centering
\\caption{{\\revcap{{Effective bulk density $\\rho_c$, number of calibration
specimens $n_c$ and between-specimen standard deviation of $\\log\\rho$,
$\\sigma_c$, for the ten classes, from the {cal['n_specimens']} calibration specimens.}}}}
\\label{{tab:densities}}
\\small
\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{lcccc}}
\\toprule
Class & Material & $\\rho_c$ (g\\,cm$^{{-3}}$) & $n_c$ & $\\sigma_c$ \\\\
\\midrule
{rows0}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

man = json.load(open("datasets/processed_v3.3/manifest.json"))
SRC = [("taco", "TACO~\\cite{proenca2020taco}"), ("zerowaste", "ZeroWaste-f~\\cite{bashkirova2022zerowaste}"),
       ("drinking_waste", "Drinking Waste~\\cite{drinkingwaste2021}"), ("dwsd", "DWSD~\\cite{dwsd2024}"), ("real495", "real495 (ours, train split only)")]
def rel(src): return man["sources"][src]["released"]["total"]
rows1 = "\n".join(f"{name:<42} & {thou(rel(src)['images']):<7} & {thou(rel(src)['annotations']):<8} \\\\" for src, name in SRC)
tot = {k: sum(rel(src)[k] for src, _ in SRC) for k in ("images", "annotations")}; spl = man["totals"]
(T / "tab1_dataset.tex").write_text(f"""\\begin{{table}}[!tb]
\\centering
\\caption{{Hybrid-v3.3 composition: the five training sources and the
stratified 70/15/15 split. BeachLitter is held out entirely.}}
\\label{{tab:dataset}}
\\small
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{lrr}}
\\toprule
Source & Images & Annotations \\\\
\\midrule
{rows1}
\\midrule
\\textbf{{Total}}  & \\textbf{{{thou(tot['images'])}}} & \\textbf{{{thou(tot['annotations'])}}} \\\\
\\bottomrule
\\multicolumn{{3}}{{l}}{{\\footnotesize train {thou(spl['train']['images'], ',')} / val {thou(spl['val']['images'], ',')} / test {thou(spl['test']['images'], ',')}}}
\\end{{tabular}}
\\end{{table}}
""")

bench = json.load(open("results/detectors/bench_summary.json")); pc = json.load(open("results/detectors/perclass_ap.json"))
DET = [("htc_r50", "HTC R-50"), ("htc_x101", "HTC ResNeXt-101-64x4d"), ("mask_rcnn_r50", "Mask R-CNN R-50"),
       ("faster_rcnn_r50", "Faster R-CNN R-50"), ("cascade_mask_rcnn_r50", "Cascade Mask R-CNN R-50")]
def bm(d, k): v = bench[d]["v3.3"].get(k); return None if v is None else float(v)
bb = max(DET, key=lambda d: bm(d[0], "bbox_mAP"))[0]; bs = max((d for d in DET if bm(d[0], "segm_mAP") is not None), key=lambda d: bm(d[0], "segm_mAP"))[0]
def c2(d, k, b):
    v = bm(d, k)
    if v is None: return "n/a"
    return f"\\textbf{{{v:.3f}}}" if d == b else f"{v:.3f}"
rows2 = "\n".join(f"{name:<23} & {c2(d, 'bbox_mAP', bb):<14} & {c2(d, 'segm_mAP', bs):<14} \\\\".rstrip() for d, name in DET)
(T / "tab2_detectors.tex").write_text(f"""\\begin{{table}}[!tb]
\\centering
\\caption{{Detector comparison on the Hybrid-v3.3 test split (COCO
bbox/mask mAP, val-best checkpoint). Best per column in bold;
Faster R-CNN has no mask head.}}
\\label{{tab:detectors}}
\\begin{{tabular}}{{lrr}}
\\toprule
Architecture & test bbox mAP & test mask mAP \\\\
\\midrule
{rows2}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
""")

PCOL = ["htc_x101", "htc_r50", "mask_rcnn_r50", "cascade_mask_rcnn_r50", "faster_rcnn_r50"]
test = json.load(open("datasets/processed_v3.3/annotations/instances_test.json"))
cid = {c["id"]: c["name"] for c in test["categories"]}
ntest = {c: 0 for c in CL}
for x in test["annotations"]: ntest[cid[x["category_id"]]] += 1
def prow(vals):
    b = max(range(len(vals)), key=lambda i: vals[i])
    return " & ".join(f"\\textbf{{{v:.3f}}}" if i == b else f"{v:.3f}" for i, v in enumerate(vals))
rows5 = "\n".join(f"{c.replace('_', chr(92) + '_'):<18} & {prow([pc[d]['bbox'][c] for d in PCOL])} & {thou(ntest[c], ',')} \\\\" for c in CL)
means = [bm(d, "bbox_mAP") for d in PCOL]
(T / "tab5_perclass.tex").write_text(f"""\\begin{{table*}}[tb]
\\centering
\\small
\\caption{{Per-class test bbox~AP across detector architectures
(val-best checkpoint; best per row in bold). \\texttt{{glass\\_jar}} and
\\texttt{{tetrapack}} have the thinnest test support, {ntest['glass_jar']} and {ntest['tetrapack']}
instances, and their rows should not be read as model properties. The
final row is the mean over classes, equal to the bbox~mAP of
Table~\\ref{{tab:detectors}}.}}
\\label{{tab:perclass}}
\\begin{{tabular}}{{lrrrrrr}}
\\toprule
class & htc\\_x101 & htc\\_r50 & mask\\_rcnn & cascade & faster\\_rcnn & $n_{{\\text{{test}}}}$ \\\\
\\midrule
{rows5}
\\midrule
\\textbf{{mean (= bbox mAP)}} & {prow(means)} & {thou(sum(ntest.values()), ',')} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")

cond = R["real495"]["conditions"]
(T / "tab_evidence.tex").write_text(f"""\\begin{{table*}}[tb]
\\centering
\\caption{{\\revcap{{Origin of the channels in every experiment. Generated: drawn
from the measured density of a calibration specimen; own output: the
detector's boxes and classes; features: ROI features on annotated boxes.}}}}
\\label{{tab:evidence}}
\\footnotesize
\\setlength{{\\tabcolsep}}{{5pt}}
\\begin{{tabular}}{{@{{}}lllllll@{{}}}}
\\toprule
Experiment & Section & Images & Volume & Mass & Fraud & Detector \\\\
\\midrule
Detector comparison & \\ref{{sec:exp:c4}} & Hybrid-v3.3 & -- & -- & -- & own output \\\\
Fusion heads and ablations & \\ref{{sec:exp:c2}}--\\ref{{sec:exp:modality}} & Hybrid-v3.3 & proxy & generated & generated & features \\\\
Noise grid, sensitivity & \\ref{{sec:exp:robust}}, \\ref{{sec:exp:density}} & Hybrid-v3.3 & proxy & generated & generated & features \\\\
Detector transfer & \\ref{{sec:exp:ood}} & BeachLitter & -- & -- & -- & own output \\\\
Frozen head, unseen source & \\ref{{sec:exp:ood}} & BeachLitter & proxy & generated & generated & features \\\\
Safety net & \\ref{{sec:exp:real}} & real495 & measured & measured & none (detector confusions) & own output \\\\
Measured masses & \\ref{{sec:exp:measured}} & real495 & measured & measured & real: {cond['heavy_fill']} filled, {cond['foreign_object']} foreign object & features \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table*}}
""")

(T / "macros_derived.tex").write_text("""% Numbers derived from tables/macros_numbers.tex. Expanded once with \\edef,
% so they read as plain digits inside \\hl.
% Detector confusions on real495 that pass the density check at the
% tolerance that rejects one correct recognition in ten.
\\edef\\nSnXPass{\\the\\numexpr\\nSnXConf-\\nSnXFlagTen\\relax}
\\edef\\nSnRPass{\\the\\numexpr\\nSnRConf-\\nSnRFlagTen\\relax}
""")
print("wrote tab0_densities, tab1_dataset, tab2_detectors, tab5_perclass, tab_evidence, macros_derived")

import re as _re
def _wrap_refs_in_revcap(text):
    out=[]; i=0
    while True:
        j=text.find("\\revcap{", i)
        if j<0: out.append(text[i:]); break
        a=j+len("\\revcap{"); depth=1; k=a
        while depth and k<len(text):
            depth += (text[k]=="{") - (text[k]=="}"); k+=1
        body=_re.sub(r'(?<!\\mbox\{)\\(ref|eqref|cite)\{([^{}]*)\}', r'\\mbox{\\\1{\2}}', text[a:k-1])
        out.append(text[i:a]+body+"}"); i=k
    return "".join(out)
for _p in T.glob("tab*.tex"):
    _p.write_text(_wrap_refs_in_revcap(_p.read_text()))
