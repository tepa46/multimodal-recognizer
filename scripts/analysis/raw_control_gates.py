"""Mean gate values of the raw- and log-input gated heads on the main-partition test tuples."""
import sys,torch,numpy as np
sys.path.insert(0,".")
from pathlib import Path
from torch.utils.data import DataLoader
from src.models import build_fusion_model
from scripts.fusion.train_fusion import TuplesDataset, collate_fn, expected_or_none
cache=torch.load("experiments/fusion/roi_cache_v33/epoch_17_test.pt",weights_only=True)
res={"raw":[], "log":[]}
for rep,root,tag in [("raw","experiments/fusion/v3_4_rawctl","gate_raw"),("log","experiments/fusion/v3_4_folds","gate")]:
  for k in range(4):
    dl=DataLoader(TuplesDataset(Path(f"datasets/multimodal/v3.4_folds/fold{k}/test/tuples.csv"),cache),batch_size=8192,collate_fn=collate_fn)
    for s in range(42,52):
      m=build_fusion_model("gated",density_yaml=f"configs/folds/fold{k}/calibration_dev.yaml",input_repr=rep,use_gate=True)
      m.load_state_dict(torch.load(f"{root}/fold{k}/{tag}/gated_lam0_mu0_s{s}/best.pth",weights_only=True)["state_dict"]); m.eval()
      G=[]
      with torch.no_grad():
        for b in dl:
          o=m(b["roi_features"],b["volume"],b["measured_weight"],b["reported_class"],expected_weight=expected_or_none(b,"cpu")); G.append(o["modality_gate"])
      res[rep].append(torch.cat(G).mean(0).tolist())
import json; out={rep:dict(zip(("visual","volume","mass"),np.array(v).mean(0).round(4).tolist())) for rep,v in res.items()}; json.dump(out,open("results/revision/raw_control_gates.json","w"),indent=1)
for rep,v in res.items():
  a=np.array(v); print(rep,"mean gate (visual, volume, mass):",a.mean(0).round(2),"min",a.min(0).round(2),"max",a.max(0).round(2))
