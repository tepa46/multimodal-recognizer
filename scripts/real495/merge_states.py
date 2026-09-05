#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def iou_xywh(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / (aw * ah + bw * bh - inter) if inter > 0 else 0.0


def match_boxes(boxes_a, boxes_b, thr=0.5):
    """Greedy IoU matching; returns (pairs, only_a, only_b)."""
    cand = []
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            v = iou_xywh(a["bbox"], b["bbox"])
            if v >= thr:
                cand.append((v, i, j))
    cand.sort(reverse=True)
    used_a, used_b, pairs = set(), set(), []
    for v, i, j in cand:
        if i in used_a or j in used_b:
            continue
        used_a.add(i); used_b.add(j)
        pairs.append((boxes_a[i], boxes_b[j], v))
    only_a = [b for i, b in enumerate(boxes_a) if i not in used_a]
    only_b = [b for j, b in enumerate(boxes_b) if j not in used_b]
    return pairs, only_a, only_b


def cohen_kappa(pairs_labels):
    """pairs_labels: list of (label_a, label_b)."""
    if not pairs_labels:
        return float("nan")
    labels = sorted({l for p in pairs_labels for l in p})
    idx = {l: i for i, l in enumerate(labels)}
    n = len(pairs_labels)
    po = sum(1 for a, b in pairs_labels if a == b) / n
    pa = [0.0] * len(labels)
    pb = [0.0] * len(labels)
    for a, b in pairs_labels:
        pa[idx[a]] += 1 / n
        pb[idx[b]] += 1 / n
    pe = sum(x * y for x, y in zip(pa, pb))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="datasets/real495/review_state.json")
    ap.add_argument("--other", required=True)
    ap.add_argument("--out",
                    default="datasets/real495/review_state.merged.json")
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    main_state = json.loads(Path(args.main).read_text())
    other_state = json.loads(Path(args.other).read_text())
    main_name = main_state.get("annotator") or "main"
    other_name = other_state.get("annotator") or "other"

    merged = {"version": 1, "annotator": main_name, "images": {}}
    pairs_all, only_m_all, only_o_all = [], 0, 0
    both_imgs = []

    all_fns = sorted(set(main_state["images"]) | set(other_state["images"]))
    for fn in all_fns:
        m = main_state["images"].get(fn)
        o = other_state["images"].get(fn)
        m_done = bool(m and m.get("exhaustive"))
        o_done = bool(o and o.get("exhaustive"))
        if m_done and o_done:
            both_imgs.append(fn)
            entry = dict(m)
            entry["annotator"] = m.get("annotator", main_name)
            entry["double_labeled_by"] = o.get("annotator", other_name)
            merged["images"][fn] = entry
            p, om, oo = match_boxes(m["boxes"], o["boxes"], args.iou)
            pairs_all.extend((a["class"], b["class"]) for a, b, _ in p)
            only_m_all += len(om)
            only_o_all += len(oo)
            dis = [(a["class"], b["class"]) for a, b, _ in p
                   if a["class"] != b["class"]]
            if dis or om or oo:
                print(f"  ✗ {fn}: class mismatch {dis if dis else ''}"
                      f"{'  only-' + main_name + ': ' + str(len(om)) if om else ''}"
                      f"{'  only-' + other_name + ': ' + str(len(oo)) if oo else ''}")
        elif o_done:
            entry = dict(o)
            entry["annotator"] = o.get("annotator", other_name)
            merged["images"][fn] = entry
        elif m:  # main's, possibly in_progress
            entry = dict(m)
            entry.setdefault("annotator", main_name)
            merged["images"][fn] = entry
        elif o:  # other's in_progress — keep, might be useful
            entry = dict(o)
            entry.setdefault("annotator", other_name)
            merged["images"][fn] = entry

    Path(args.out).write_text(json.dumps(merged, ensure_ascii=False))

    n_done = sum(1 for v in merged["images"].values() if v.get("exhaustive"))
    print(f"\n[merge] merged -> {args.out}")
    print(f"[merge] done images: {n_done} "
          f"(main={main_name}: "
          f"{sum(1 for v in main_state['images'].values() if v.get('exhaustive'))}, "
          f"{other_name}: "
          f"{sum(1 for v in other_state['images'].values() if v.get('exhaustive'))}, "
          f"double-labeled: {len(both_imgs)})")
    if both_imgs:
        agree = sum(1 for a, b in pairs_all if a == b)
        print(f"[kappa] matched boxes: {len(pairs_all)}, agreement: "
              f"{agree}/{len(pairs_all)}"
              f" ({agree / len(pairs_all):.1%})" if pairs_all else "")
        print(f"[kappa] unmatched: only-{main_name}={only_m_all}, "
              f"only-{other_name}={only_o_all}")
        print(f"[kappa] Cohen's kappa (classes, matched pairs): "
              f"{cohen_kappa(pairs_all):.3f}")
    print("\nЕсли отчёт ок:  cp", args.out, args.main)


if __name__ == "__main__":
    main()
