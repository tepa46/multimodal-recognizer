#!/bin/bash
set -euo pipefail

SRC=${1:-my_dataset}
DST=${2:-datasets/real495/images}

mkdir -p "$DST"

total=0
converted=0
skipped=0
for f in "$SRC"/*.HEIC; do
  name=$(basename "$f" .HEIC)
  out="$DST/$name.jpg"
  total=$((total + 1))
  if [[ -s "$out" ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  sips -s format jpeg -s formatOptions 92 -Z 3024 "$f" --out "$out" >/dev/null
  converted=$((converted + 1))
  if (( converted % 25 == 0 )); then
    echo "  ...$converted converted"
  fi
done

echo "HEIC total: $total, converted now: $converted, skipped (existing): $skipped"

n_jpg=$(find "$DST" -name '*.jpg' | wc -l | tr -d ' ')
echo "JPEG in $DST: $n_jpg"
if [[ "$n_jpg" -ne "$total" ]]; then
  echo "WARNING: JPEG count ($n_jpg) != HEIC count ($total)" >&2
fi

echo "--- size check (3 samples, expect long side 3024) ---"
find "$DST" -name '*.jpg' | head -3 | while read -r s; do
  sips -g pixelWidth -g pixelHeight "$s" | tr '\n' ' '
  echo
done

echo "--- total size ---"
du -sh "$DST"
