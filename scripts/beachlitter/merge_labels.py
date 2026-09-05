#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

PACKAGING = {
    "plastic_bottle", "glass_bottle", "glass_jar", "metal_can", "carton",
    "tetrapack", "paper", "plastic_bag", "plastic_container",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv-dir", required=True,
                    help="directory with page_*.csv exports from the galleries")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(Path(args.csv_dir).glob("*page_*.csv"))
    if not files:
        raise SystemExit(f"no page CSVs found in {args.csv_dir}")

    rows: dict[int, dict] = {}
    conflicts = 0
    for f in files:
        with open(f) as fh:
            for row in csv.DictReader(fh):
                aid = int(row["ann_id"])
                row["page"] = f.stem
                prev = rows.get(aid)
                if prev is None:
                    rows[aid] = row
                    continue
                # explicit decision beats auto_default; otherwise later wins
                if prev["flags"] == "clicked" and row["flags"] == "auto_default":
                    continue
                if prev["final_class"] != row["final_class"]:
                    conflicts += 1
                rows[aid] = row

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fields = ["ann_id", "fn", "final_class", "annotator", "flags",
              "sugg", "psim", "side", "page"]
    with open(out_p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for aid in sorted(rows):
            w.writerow(rows[aid])

    cls = Counter(r["final_class"] for r in rows.values())
    acc_by_page: dict[str, int] = defaultdict(int)
    for r in rows.values():
        if r["final_class"] in PACKAGING:
            acc_by_page[r["page"]] += 1

    print(f"merged {len(files)} files -> {out_p} ({len(rows)} objects, "
          f"{conflicts} overwritten conflicts)")
    print("\nfinal_class totals:")
    for name, n in cls.most_common():
        tag = "PACK" if name in PACKAGING else "    "
        print(f"  [{tag}] {name:<18} {n}")
    npack = sum(n for c, n in cls.items() if c in PACKAGING)
    print(f"\npackaging total: {npack}")
    print("\naccepted per page (stopping-rule profile):")
    for page in sorted(acc_by_page):
        print(f"  {page}: {acc_by_page[page]}")


if __name__ == "__main__":
    main()
