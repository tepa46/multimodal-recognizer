"""Per-source manifest of the Hybrid-v3.3 build: remapping, exclusions, split counts."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import scripts.build_data.build_hybrid_coco as B

SOURCES = ["taco", "zerowaste", "drinking_waste", "dwsd", "real495"]
RELEASED_NAME = {"real300": "real495"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=Path("datasets/raw"))
    ap.add_argument("--release", type=Path, default=Path("datasets/processed_v3.3/annotations"))
    ap.add_argument("--class-mapping", type=Path, default=Path("configs/class_mapping.yaml"))
    ap.add_argument("--cardboard-frac", type=float, default=0.33)
    ap.add_argument("--out", type=Path, default=Path("datasets/processed_v3.3/manifest.json"))
    a = ap.parse_args()

    unified, class_map = B.load_class_mapping(a.class_mapping)
    loaded, sub, splits = [], {}, {}

    load = B.load_coco_json
    def load_rec(path):
        d = load(path); loaded.append((Path(path), d)); return d
    B.load_coco_json = load_rec

    subsample = B.subsample_zerowaste_cardboard
    def subsample_rec(images, annotations, *args, **kw):
        sub["before"] = (list(images), list(annotations))
        out = subsample(images, annotations, *args, **kw)
        sub["after"] = out
        return out
    B.subsample_zerowaste_cardboard = subsample_rec

    def save_rec(split_name, img_id_set, all_images, all_annotations, unified_classes, **kw):
        imgs = [i for i in all_images if i["id"] in img_id_set]
        anns = [x for x in all_annotations if x["image_id"] in img_id_set]
        splits[split_name] = (imgs, anns)
        return Counter(unified_classes[x["category_id"]] for x in anns)
    B.save_split = save_rec

    B.build(raw_dir=a.raw_dir, output_dir=Path("/tmp/_manifest_unused"), class_mapping_path=a.class_mapping,
            val_ratio=0.15, test_ratio=0.15, image_size=None, sources=SOURCES,
            exclude=["beach_litter"], zerowaste_cardboard_frac=a.cardboard_frac)

    def source_of(path: Path) -> str:
        s = str(path)
        for name in ("taco", "zerowaste", "drinking_waste", "dwsd", "real495"):
            if f"/{name}/" in s or s.startswith(f"datasets/{name}/"):
                return name
        return "?"

    raw = defaultdict(lambda: {"images": 0, "annotations": 0, "files": []})
    remap = defaultdict(Counter)
    for path, d in loaded:
        src = source_of(path)
        raw[src]["images"] += len(d.get("images", [])); raw[src]["annotations"] += len(d.get("annotations", []))
        raw[src]["files"].append(str(path.relative_to(a.raw_dir)) if a.raw_dir in path.parents else str(path))
        names = {c["id"]: c["name"] for c in d.get("categories", [])}
        for x in d.get("annotations", []):
            n = names.get(x["category_id"], "")
            remap[src][(n, unified[B.remap_category_id(n, src, class_map, unified)])] += 1

    split_doc = json.loads(Path("datasets/real495/split_train_heldout.json").read_text())
    r495 = next(d for p, d in loaded if source_of(p) == "real495")
    held = set(split_doc["heldout"]); held_ids = {i["id"] for i in r495["images"] if i["file_name"] in held}

    def per_source(images, annotations):
        img_src = {i["id"]: i["source"] for i in images}
        ni = Counter(img_src.values()); na = Counter(img_src[x["image_id"]] for x in annotations if x["image_id"] in img_src)
        return ni, na

    bi, ba = per_source(*sub["before"]); ai, aa = per_source(*sub["after"])
    released = {s: json.loads((a.release / f"instances_{s}.json").read_text()) for s in ("train", "val", "test")}
    for d in released.values():
        for i in d["images"]:
            i["source"] = RELEASED_NAME.get(i["source"], i["source"])
    rebuilt_keys = {s: sorted((i["source"], i["original_id"]) for i in splits[s][0]) for s in splits}
    released_keys = {s: sorted((i["source"], i["original_id"]) for i in d["images"]) for s, d in released.items()}
    rebuilt_ann = {s: len(splits[s][1]) for s in splits}; released_ann = {s: len(d["annotations"]) for s, d in released.items()}

    out = {"build_command": "python scripts/build_data/build_hybrid_coco.py --raw-dir datasets/raw/ "
                            "--output-dir datasets/processed_v3.3/ --sources taco zerowaste drinking_waste dwsd real495 "
                            "--exclude beach_litter --zerowaste-cardboard-frac 0.33",
           "class_mapping": str(a.class_mapping), "unified_classes": unified,
           "released_source_names": {"real300": "real495 (earlier name of the same set; file names real300_*)"},
           "excluded_sources": {"beach_litter": "held out of training entirely; evaluation only"},
           "split": {"ratios": [0.70, 0.15, 0.15], "stratified_by": "most frequent class in the image",
                     "random_state": 42, "real495": "train frames forced into train; held-out frames withheld"},
           "sources": {}}
    for src in SOURCES:
        rec = {"raw_files": raw[src]["files"], "raw_images": raw[src]["images"], "raw_annotations": raw[src]["annotations"],
               "remapping": [{"original": n, "unified": u, "annotations": c} for (n, u), c in sorted(remap[src].items(), key=lambda t: (-t[1], t[0]))],
               "excluded": {"orphaned_annotations": raw[src]["annotations"] - ba[src] - (sum(1 for x in r495["annotations"] if x["image_id"] in held_ids) if src == "real495" else 0)}}
        if src == "zerowaste":
            rec["excluded"]["cardboard_subsample_annotations"] = ba[src] - aa[src]
            rec["excluded"]["images_left_without_annotations"] = bi[src] - ai[src]
            rec["excluded"]["cardboard_fraction_kept"] = a.cardboard_frac
        if src == "real495":
            rec["excluded"]["held_out_frames"] = len(held_ids)
            rec["excluded"]["held_out_annotations"] = sum(1 for x in r495["annotations"] if x["image_id"] in held_ids)
        rec["released"] = {s: {"images": sum(1 for i in released[s]["images"] if i["source"] == src),
                               "annotations": sum(1 for x in released[s]["annotations"]
                                                  if x["image_id"] in {i["id"] for i in released[s]["images"] if i["source"] == src})}
                           for s in released}
        rec["released"]["total"] = {k: sum(rec["released"][s][k] for s in ("train", "val", "test")) for k in ("images", "annotations")}
        out["sources"][src] = rec
    out["totals"] = {s: {"images": len(d["images"]), "annotations": len(d["annotations"])} for s, d in released.items()}
    out["rebuild_matches_release"] = {s: rebuilt_keys[s] == released_keys[s] and rebuilt_ann[s] == released_ann[s] for s in released}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("wrote", a.out, out["rebuild_matches_release"])


if __name__ == "__main__":
    main()
