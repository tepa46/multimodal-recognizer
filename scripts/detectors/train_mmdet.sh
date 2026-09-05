#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config_name> [mim train mmdet flags...]"
    echo "  config_name: filename without extension under configs/detectors/"
    echo "  e.g. htc_r50 -> configs/detectors/htc_r50.py"
    exit 1
fi

CONFIG_NAME="$1"; shift
CONFIG_PATH="configs/detectors/${CONFIG_NAME}.py"
WORK_DIR="experiments/work_dirs/${CONFIG_NAME}"

if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "Config not found: $CONFIG_PATH"
    exit 1
fi

# The caller may override --work-dir in "$@"; only create ours if it does not.
if [[ " $* " == *" --work-dir "* ]]; then WORK_DIR=""; else mkdir -p "$WORK_DIR"; fi
mkdir -p checkpoints

# Use the project venv's mim if present (uv-managed), otherwise fall back
# to the system-wide one.
MIM_BIN="${MIM_BIN:-}"
if [[ -z "$MIM_BIN" ]]; then
    if [[ -x ".venv/bin/mim" ]]; then
        MIM_BIN=".venv/bin/mim"
    elif command -v mim >/dev/null 2>&1; then
        MIM_BIN="mim"
    else
        echo "ERROR: 'mim' not found. Install via: .venv/bin/pip install openmim && .venv/bin/mim install mmengine 'mmcv>=2.1' 'mmdet>=3.3'"
        exit 1
    fi
fi

if [[ -n "$WORK_DIR" ]]; then
    "$MIM_BIN" train mmdet "$CONFIG_PATH" --work-dir "$WORK_DIR" "$@"
else
    "$MIM_BIN" train mmdet "$CONFIG_PATH" "$@"
fi
