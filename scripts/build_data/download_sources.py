"""
Download datasets for the hybrid waste COCO benchmark.

Supported datasets:
  taco           — TACO: Trash Annotations in Context (~1500 imgs, COCO masks)
  zerowaste      — ZeroWaste-f (4503 labelled imgs, COCO masks, Google Drive)
  drinking_waste — Drinking Waste Classification (UCL/Kaggle, ~10k imgs, bbox only)
  dwsd           — Dense Waste Segmentation Dataset (Mendeley, 784 imgs, COCO masks)
  beach_litter   — BeachLitter v2022 (GitHub release, ~3500 imgs, pixel masks)

Usage:
  python scripts/build_data/download_sources.py --datasets all --output datasets/raw/
  python scripts/build_data/download_sources.py --datasets taco zerowaste --output datasets/raw/
  python scripts/build_data/download_sources.py --list
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS: dict[str, dict] = {
    "taco": {
        "description": "TACO — Trash Annotations in Context (~1500 imgs, COCO masks)",
        "method": "git_taco",
        "repo_url": "https://github.com/pedropro/TACO.git",
    },
    "zerowaste": {
        "description": "ZeroWaste-f (4503 labelled imgs, COCO masks)",
        "method": "zenodo_zerowaste",
        # Zenodo record 6412647 — zerowaste-f-final.zip (7.5 GB, password-protected)
        "url": "https://zenodo.org/api/records/6412647/files/zerowaste-f-final.zip/content",
        "filename": "zerowaste-f-final.zip",
        # The archive password is published by the ZeroWaste authors on the
        # Zenodo record page itself. It is intentionally NOT hard-coded here:
        # this repository is meant for public release, and re-publishing another
        # project's access credential is theirs to do, not ours. Supply it via
        # the ZEROWASTE_ZIP_PASSWORD environment variable.
        "zip_password_env": "ZEROWASTE_ZIP_PASSWORD",
    },
    "drinking_waste": {
        "description": "Drinking Waste Classification — UCL/Kaggle (~10k imgs, bbox only; masks via SAM)",
        "method": "kaggle_drinking_waste",
        "kaggle_slug": "arkadiyhacks/drinking-waste-classification",
        "manual_url": "https://www.kaggle.com/datasets/arkadiyhacks/drinking-waste-classification",
    },
    "dwsd": {
        "description": "DWSD — Dense Waste Segmentation Dataset (784 imgs, COCO masks)",
        "method": "mendeley_dwsd",
        # Mendeley requires browser login to download — manual only
        "manual_url": "https://data.mendeley.com/datasets/gr99ny6b8p/1",
        "filename": "dwsd.zip",
    },
    "beach_litter": {
        "description": "BeachLitter v2022 (~3500 imgs, pixel-wise masks)",
        "method": "seanoe_beach_litter",
        # SEANOE: DOI 10.17882/85472, ~937 MB, CC BY-NC-SA 4.0
        "url": "https://www.seanoe.org/data/00743/85472/data/90569.zip",
        "filename": "beach_litter_v2022.zip",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], cwd: Path | None = None) -> int:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode


def check_skip(dest: Path, marker: str = ".downloaded") -> bool:
    """Return True if dataset already downloaded."""
    if (dest / marker).exists():
        print(f"  [skip] {dest} already downloaded.")
        return True
    return False


def mark_done(dest: Path, marker: str = ".downloaded") -> None:
    (dest / marker).touch()


def extract_zip(zip_path: Path, dest: Path, password: str | None = None) -> None:
    print(f"  Extracting {zip_path.name} → {dest}")
    pwd_bytes = password.encode() if password else None
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest, pwd=pwd_bytes)
    except RuntimeError:
        # AES-256 zips need pyzipper; ZipCrypto is handled by stdlib
        try:
            import pyzipper
            with pyzipper.AESZipFile(zip_path, "r") as zf:
                zf.extractall(dest, pwd=pwd_bytes)
        except ImportError:
            # Last resort: system unzip
            cmd = ["unzip", "-q"]
            if password:
                cmd += ["-P", password]
            cmd += [str(zip_path), "-d", str(dest)]
            rc = run(cmd)
            if rc != 0:
                print("  ERROR: extraction failed. Install pyzipper or system unzip.")
                return
    zip_path.unlink()


def _download_url(url: str, dest_file: Path, label: str = "") -> bool:
    """Download url → dest_file with a progress bar. Returns True on success."""
    try:
        import requests
    except ImportError:
        run([sys.executable, "-m", "pip", "install", "-q", "requests"])
        import requests

    desc = label or dest_file.name
    print(f"  Downloading {desc} …")
    try:
        with requests.get(url, stream=True, timeout=300,
                          headers={"User-Agent": "Mozilla/5.0"}) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb_done = downloaded >> 20
                        mb_total = total >> 20
                        print(f"\r  {pct:5.1f}%  {mb_done} / {mb_total} MB", end="", flush=True)
        print()
        return True
    except Exception as exc:
        print(f"\n  ERROR: {exc}")
        if dest_file.exists():
            dest_file.unlink()
        return False


def has_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()


def print_manual_download(url: str, dest: Path, hint: str = "") -> None:
    print()
    print("  *** MANUAL DOWNLOAD REQUIRED ***")
    print(f"  URL : {url}")
    print(f"  Save to: {dest}")
    if hint:
        print(f"  Hint: {hint}")
    print()


# ---------------------------------------------------------------------------
# Download methods
# ---------------------------------------------------------------------------

def download_taco(dest: Path) -> None:
    """Clone TACO repo, then run its download script to fetch the images."""
    repo_dir = dest / "taco"
    if check_skip(repo_dir):
        return
    repo_dir.mkdir(parents=True, exist_ok=True)

    print("  Cloning TACO repository...")
    rc = run(["git", "clone", "--depth", "1",
               "https://github.com/pedropro/TACO.git", str(repo_dir)])
    if rc != 0:
        print("  ERROR: git clone failed.")
        return

    print("  Installing TACO dependencies (requests, Pillow)...")
    run([sys.executable, "-m", "pip", "install", "-q", "requests", "Pillow"])

    print("  Downloading TACO images (this may take a few minutes)...")
    rc = run(
        [sys.executable, "download.py", "--dataset_path", "./data/annotations.json"],
        cwd=repo_dir,
    )
    if rc != 0:
        print("  WARNING: download.py exited with errors — some images may be missing.")

    mark_done(repo_dir)
    print("  TACO done.")


def download_zerowaste(dest: Path) -> None:
    """Download ZeroWaste-f from Zenodo (7.5 GB, password-protected zip)."""
    out_dir = dest / "zerowaste"
    if check_skip(out_dir):
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    url = DATASETS["zerowaste"]["url"]
    password = os.environ.get(DATASETS["zerowaste"]["zip_password_env"])
    if not password:
        print("  ZeroWaste-f: archive password not set. Take it from the Zenodo\n"
              "  record page (https://zenodo.org/records/6412647) and re-run with:\n"
              f"    {DATASETS['zerowaste']['zip_password_env']}='<password>' "
              "python scripts/build_data/download_sources.py --datasets zerowaste")
        return
    zip_path = out_dir / "zerowaste-f-final.zip"

    ok = _download_url(url, zip_path, "ZeroWaste-f from Zenodo (7.5 GB)")
    if ok:
        extract_zip(zip_path, out_dir, password=password)
        mark_done(out_dir)
        print("  ZeroWaste done.")
    else:
        print_manual_download(
            url="https://zenodo.org/records/6412647",
            dest=out_dir,
            hint=f"Download zerowaste-f-final.zip, extract with password: {password}",
        )


def download_drinking_waste(dest: Path) -> None:
    """Download Drinking Waste via Kaggle API or print manual instructions."""
    out_dir = dest / "drinking_waste"
    if check_skip(out_dir):
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    if has_kaggle_credentials():
        import importlib.util
        if importlib.util.find_spec("kaggle") is None:
            run([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

        print("  Downloading Drinking Waste via Kaggle API...")
        rc = run(
            ["kaggle", "datasets", "download",
             "-d", "arkadiyhacks/drinking-waste-classification",
             "-p", str(out_dir), "--unzip"],
        )
        if rc == 0:
            mark_done(out_dir)
            print("  Drinking Waste done.")
            return
        print("  Kaggle download failed — falling back to manual.")

    # No credentials or download failed
    print_manual_download(
        url=DATASETS["drinking_waste"]["manual_url"],
        dest=out_dir,
        hint=(
            "1. Go to the URL above, click 'Download'.\n"
            "  2. Unzip the archive into datasets/raw/drinking_waste/\n"
            "  3. The directory should contain class subdirectories: Aluminium, Glass, HDPE, PET.\n"
            "  4. To set up Kaggle API: https://www.kaggle.com/docs/api#authentication"
        ),
    )


def download_dwsd(dest: Path) -> None:
    """DWSD requires browser login on Mendeley — print manual instructions."""
    out_dir = dest / "dwsd"
    if check_skip(out_dir):
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    print_manual_download(
        url=DATASETS["dwsd"]["manual_url"],
        dest=out_dir,
        hint=(
            "1. Open the URL above in a browser.\n"
            "  2. Click 'Download All' (requires free Mendeley account).\n"
            "  3. Unzip into datasets/raw/dwsd/\n"
            "  4. Expected structure: images/ + annotations.json (or similar)."
        ),
    )


def download_beach_litter(dest: Path) -> None:
    """Download BeachLitter v2022 from SEANOE (~937 MB)."""
    out_dir = dest / "beach_litter"
    if check_skip(out_dir):
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    url = DATASETS["beach_litter"]["url"]
    zip_path = out_dir / "beach_litter_v2022.zip"
    ok = _download_url(url, zip_path, "BeachLitter v2022 from SEANOE (937 MB)")
    if ok:
        extract_zip(zip_path, out_dir)
        mark_done(out_dir)
        print("  BeachLitter done.")
    else:
        print_manual_download(
            url="https://doi.org/10.17882/85472",
            dest=out_dir,
            hint="Download the zip from SEANOE and extract into datasets/raw/beach_litter/",
        )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

DOWNLOAD_FN = {
    "taco": download_taco,
    "zerowaste": download_zerowaste,
    "drinking_waste": download_drinking_waste,
    "dwsd": download_dwsd,
    "beach_litter": download_beach_litter,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download public waste datasets for the hybrid COCO benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--datasets", nargs="+", metavar="NAME",
        default=["all"],
        help="Datasets to download. Use 'all' to download everything. "
             f"Available: {', '.join(DATASETS)}",
    )
    parser.add_argument(
        "--output", default="datasets/raw/", metavar="DIR",
        help="Root directory for raw dataset storage (default: datasets/raw/)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print("Available datasets:")
        for name, info in DATASETS.items():
            print(f"  {name:<20} {info['description']}")
        return

    names: list[str] = list(DATASETS.keys()) if args.datasets == ["all"] else args.datasets
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        parser.error(f"Unknown datasets: {', '.join(unknown)}. "
                     f"Available: {', '.join(DATASETS)}")

    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_root.resolve()}")
    print(f"Datasets to download: {', '.join(names)}\n")

    for name in names:
        print(f"=== {name} — {DATASETS[name]['description']} ===")
        DOWNLOAD_FN[name](out_root)
        print()

    print("All done. Check datasets/raw/ for the downloaded data.")
    print("Next step: run scripts/build_data/build_hybrid_coco.py to merge and convert.")


if __name__ == "__main__":
    main()
