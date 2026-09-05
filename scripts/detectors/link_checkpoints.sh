#!/bin/bash
set -eu
cd "$(dirname "$0")/../.."

SRC=models/detectors
DST=experiments/work_dirs

[ -d "$SRC" ] || { echo "no $SRC — download the checkpoints first (see models/README.md)"; exit 1; }
mkdir -p "$DST"

n=0
for d in "$SRC"/*/; do
    [ -d "$d" ] || continue
    name=$(basename "$d")
    target="$DST/$name"
    if [ -e "$target" ] && [ ! -L "$target" ]; then
        echo "  skip $name — real directory exists (trained locally?)"
        continue
    fi
    rm -f "$target"
    ln -s "$(cd "$d" && pwd)" "$target"
    echo "  $target -> $d"
    n=$((n + 1))
done

echo "linked $n checkpoint directories"
[ "$n" -gt 0 ] && ls -1 "$SRC"/*/*.pth 2>/dev/null | sed 's/^/  /' || true
