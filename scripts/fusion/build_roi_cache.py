import sys
sys.path.insert(0, ".")
from pathlib import Path
from scripts.fusion.train_fusion import load_or_build_roi_cache
cfg = Path("configs/detectors/htc_x101.py")
csv_path, out = Path(sys.argv[1]), Path(sys.argv[2])
ckpt = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("experiments/work_dirs/htc_x101_v33_noreal/epoch_17.pth")
c = load_or_build_roi_cache(csv_path, out, cfg, ckpt, "cuda")
print("n", len(c))
