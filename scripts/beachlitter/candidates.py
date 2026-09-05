#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

PACKAGING = [
    "plastic_bottle", "glass_bottle", "glass_jar", "metal_can", "carton",
    "tetrapack", "paper", "plastic_bag", "plastic_container",
]
FINAL_LABELS = PACKAGING + ["other", "ignore_merged"]

CROP_PAD = 0.15
CROP_MAX = 256          # px, thumbnail max side
CTX_MAX = 640           # px, context view max side
PAGE_SIZE = 100


# --------------------------------------------------------------------------- #
# Queue construction
# --------------------------------------------------------------------------- #

def load_zeroshot(path: Path, min_side: int) -> list[dict]:
    det = json.load(open(path))
    out = []
    for d in det:
        side = max(d["bbox"][2], d["bbox"][3])
        if side < min_side:
            continue
        d["side"] = side
        d["psim"] = max(d["sims"][c] for c in PACKAGING)
        d["pack_sugg"] = max(PACKAGING, key=lambda c: d["sims"][c])
        out.append(d)
    return out


def build_queue(det: list[dict], top_k: int) -> list[dict]:
    det = sorted(det, key=lambda d: -d["psim"])
    ranked = det[:top_k]
    seen = {d["ann_id"] for d in ranked}
    extra = [d for d in det[top_k:]
             if d["pred"] in PACKAGING and d["ann_id"] not in seen]
    if extra:
        print(f"  +{len(extra)} argmax-packaging candidates below rank {top_k}")
    return ranked + extra


def completeness_queue(det: list[dict], verified_csv: Path,
                       main_queue_json: Path | None = None) -> list[dict]:
    reviewed, accepted_imgs = set(), set()
    with open(verified_csv) as f:
        for row in csv.DictReader(f):
            reviewed.add(int(row["ann_id"]))
            if row["final_class"] in PACKAGING:
                accepted_imgs.add(row["fn"])
    in_main = set()
    if main_queue_json and main_queue_json.exists():
        in_main = {c["ann_id"] for c in json.load(open(main_queue_json))}
    queue = [d for d in det
             if d["fn"] in accepted_imgs
             and d["ann_id"] not in reviewed
             and d["ann_id"] not in in_main]
    queue.sort(key=lambda d: (d["fn"], -d["psim"]))
    print(f"  completeness: {len(accepted_imgs)} accepted images, "
          f"{len(queue)} beyond-queue unreviewed objects on them "
          f"(excluded {len(in_main)} main-queue ids)")
    return queue


# --------------------------------------------------------------------------- #
# Crop / context rendering
# --------------------------------------------------------------------------- #

def render_assets(queue: list[dict], image_root: Path, out_dir: Path) -> None:
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    by_fn = defaultdict(list)
    for d in queue:
        by_fn[d["fn"]].append(d)

    n_done = 0
    for fn, items in by_fn.items():
        try:
            im = Image.open(image_root / fn).convert("RGB")
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] cannot open {fn}: {exc}")
            continue
        for d in items:
            x, y, w, h = d["bbox"]
            dx, dy = w * CROP_PAD, h * CROP_PAD
            box = (max(0, int(x - dx)), max(0, int(y - dy)),
                   min(im.width, int(x + w + dx)), min(im.height, int(y + h + dy)))
            crop_p = crops_dir / f"{d['ann_id']}.jpg"
            if not crop_p.exists():
                crop = im.crop(box)
                crop.thumbnail((CROP_MAX, CROP_MAX))
                crop.save(crop_p, quality=88)
            ctx_p = crops_dir / f"ctx_{d['ann_id']}.jpg"
            if not ctx_p.exists():
                ctx = im.copy()
                dr = ImageDraw.Draw(ctx)
                lw = max(3, int(max(ctx.size) / 300))
                dr.rectangle((x, y, x + w, y + h), outline=(255, 40, 40), width=lw)
                ctx.thumbnail((CTX_MAX, CTX_MAX))
                ctx.save(ctx_p, quality=82)
            n_done += 1
            if n_done % 500 == 0:
                print(f"  assets: {n_done}/{len(queue)}")
    print(f"  assets done: {n_done}")


# --------------------------------------------------------------------------- #
# HTML generation
# --------------------------------------------------------------------------- #

PAGE_TMPL = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ background:#181818; color:#ddd; font-family:system-ui,sans-serif; margin:0; }}
 header {{ position:sticky; top:0; background:#222; padding:8px 16px; z-index:5;
          display:flex; gap:16px; align-items:center; border-bottom:1px solid #444; }}
 header b {{ color:#ffd24d; }}
 button {{ cursor:pointer; border:0; border-radius:4px; padding:5px 10px; }}
 .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(230px,1fr));
         gap:10px; padding:12px; }}
 .card {{ background:#242424; border:3px solid #555; border-radius:8px; padding:6px; }}
 .card.acc  {{ border-color:#3fbf5a; }}
 .card.rej  {{ border-color:#7a3038; }}
 .card.ign  {{ border-color:#e0a030; }}
 .card img.crop {{ width:100%; max-height:180px; object-fit:contain;
                  background:#111; border-radius:4px; cursor:zoom-in; }}
 .card img.ctx {{ width:100%; display:none; margin-top:4px; border-radius:4px; }}
 .meta {{ font-size:12px; color:#aaa; margin:4px 0; display:flex;
         justify-content:space-between; }}
 .zone {{ color:#e0a030; font-weight:600; }}
 .row {{ display:flex; gap:4px; margin-top:4px; flex-wrap:wrap; }}
 .accbtn {{ background:#2e7d32; color:#fff; flex:1; }}
 .othbtn {{ background:#5a3338; color:#eee; }}
 .ignbtn {{ background:#7a5a20; color:#eee; }}
 select {{ background:#333; color:#ddd; border-radius:4px; padding:4px; width:100%;
          margin-top:4px; }}
 .state {{ font-size:12px; margin-top:4px; min-height:15px; color:#9fdfaf; }}
</style></head><body>
<header>
 <b>{title}</b>
 <span>принято: <b id="nacc">0</b> / {n}</span>
 <button style="background:#3a6ea5;color:#fff" onclick="exportCSV()">Скачать CSV страницы</button>
 <button style="background:#444;color:#ccc" onclick="resetPage()">Сбросить страницу</button>
 <span style="font-size:12px;color:#888">клавиши при наведении: A=принять подсказку,
 O=other, M=ignore(слипшийся) &nbsp;|&nbsp; клик по кропу = контекст</span>
</header>
<div class="grid" id="grid"></div>
<script>
const ITEMS = {items_json};
const PAGE = "{page_key}";
const CLASSES = {classes_json};
let st = JSON.parse(localStorage.getItem("blv2_"+PAGE) || "{{}}");
let cur = null;

function setLabel(id, lbl, explicit) {{
  st[id] = {{l: lbl, e: explicit ? 1 : 0}};
  localStorage.setItem("blv2_"+PAGE, JSON.stringify(st));
  paint(id); count();
}}
function labelOf(id) {{ return (st[id] && st[id].l) || "other"; }}
function explicitOf(id) {{ return st[id] ? st[id].e : 0; }}

function paint(id) {{
  const card = document.getElementById("c"+id);
  const l = labelOf(id), e = explicitOf(id);
  card.className = "card " + (CLASSES.includes(l) ? "acc" : (l === "ignore_merged" ? "ign" : (e ? "rej" : "")));
  card.querySelector(".state").textContent =
     (CLASSES.includes(l) ? "✔ " + l : (l === "ignore_merged" ? "⊘ ignore (слипшийся)" : (e ? "✗ other" : "— other (по умолчанию)")));
  card.querySelector("select").value = l;
}}
function count() {{
  const n = ITEMS.filter(it => CLASSES.includes(labelOf(it.id))).length;
  document.getElementById("nacc").textContent = n;
}}
function render() {{
  const g = document.getElementById("grid");
  g.innerHTML = ITEMS.map(it => `
   <div class="card" id="c${{it.id}}" onmouseenter="cur=${{it.id}}">
    <img class="crop" loading="lazy" src="crops/${{it.id}}.jpg"
         onclick="const c=this.parentNode.querySelector('.ctx');
                  c.style.display = c.style.display==='block' ? 'none' : 'block';
                  if(!c.src) c.src='crops/ctx_${{it.id}}.jpg';">
    <img class="ctx">
    <div class="meta"><span>${{it.size}} px ${{it.zone ? '<span class="zone">40–60</span>' : ''}}</span>
      <span>SigLIP: ${{it.sugg}} ${{it.psim.toFixed(2)}}</span></div>
    <div class="row">
      <button class="accbtn" onclick="setLabel(${{it.id}}, '${{it.sugg}}', true)">✔ ${{it.sugg}}</button>
      <button class="othbtn" onclick="setLabel(${{it.id}}, 'other', true)">other</button>
      <button class="ignbtn" onclick="setLabel(${{it.id}}, 'ignore_merged', true)">⊘</button>
    </div>
    <select onchange="setLabel(${{it.id}}, this.value, true)">
      ${{["other","ignore_merged"].concat(CLASSES).map(c => `<option value="${{c}}">${{c}}</option>`).join("")}}
    </select>
    <div class="state"></div>
   </div>`).join("");
  ITEMS.forEach(it => paint(it.id)); count();
}}
document.addEventListener("keydown", ev => {{
  if (cur === null) return;
  const it = ITEMS.find(x => x.id === cur);
  if (!it) return;
  if (ev.key === "a" || ev.key === "A" || ev.key === "ф") setLabel(cur, it.sugg, true);
  if (ev.key === "o" || ev.key === "O" || ev.key === "щ") setLabel(cur, "other", true);
  if (ev.key === "m" || ev.key === "M" || ev.key === "ь") setLabel(cur, "ignore_merged", true);
}});
function exportCSV() {{
  let annot = localStorage.getItem("blv2_annotator");
  if (!annot) {{
    annot = prompt("Имя разметчика (для CSV):", "tepa") || "anon";
    localStorage.setItem("blv2_annotator", annot);
  }}
  const rows = [["ann_id","fn","final_class","annotator","flags","sugg","psim","side"]];
  ITEMS.forEach(it => rows.push([it.id, it.fn, labelOf(it.id), annot,
    explicitOf(it.id) ? "clicked" : "auto_default", it.sugg,
    it.psim.toFixed(4), it.side]));
  const blob = new Blob([rows.map(r => r.join(",")).join("\\n")], {{type:"text/csv"}});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "{page_key}.csv";
  a.click();
}}
function resetPage() {{
  if (confirm("Сбросить все решения на этой странице?")) {{
    st = {{}}; localStorage.removeItem("blv2_"+PAGE); render();
  }}
}}
render();
</script></body></html>
"""

INDEX_TMPL = """<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8"><title>BeachLitter-10 v2 — валидация</title>
<style>body{{background:#181818;color:#ddd;font-family:system-ui;max-width:860px;
margin:24px auto;padding:0 16px}} a{{color:#7ab8ff}} li{{margin:4px 0}}
code{{background:#2a2a2a;padding:1px 5px;border-radius:4px}}</style></head><body>
<h2>BeachLitter-10 v2 — очередь ручной валидации ({mode})</h2>
<p>Всего {n} объектов, {npages} страниц по {page_size}. Страницы отсортированы по
убыванию «упаковочности» (SigLIP-2). Протокол — <code>BEACHLITTER_V2_PLAN.md</code> §1:</p>
<ul>
<li><b>✔ класс</b> — только если уверен; <b>сомнение → other</b>;</li>
<li>жёсткая тара (канистры, вёдра, ящики) → <code>plastic_container</code>;
    плёнка/обрывки пакетов → <code>plastic_bag</code>;</li>
<li>в одном bbox несколько предметов и главный не выделить → <b>⊘ ignore_merged</b>;</li>
<li>бейдж <b style="color:#e0a030">40–60</b> — мелкая зона, там подсказки шумные;</li>
<li><b>правило остановки:</b> если на странице принято &lt;2 объектов —
    дальше страницы можно не проверять (зафиксируй номер);</li>
<li>после каждой страницы жми <b>«Скачать CSV страницы»</b>; прогресс также
    хранится в браузере (localStorage).</li>
</ul>
<p>Слить выгруженные CSV: <code>python scripts/beachlitter/merge_labels.py
--csv-dir &lt;папка с CSV&gt; --out verified_labels.csv</code></p>
<ol>
{links}
</ol></body></html>
"""


def write_pages(queue: list[dict], out_dir: Path, prefix: str, mode: str) -> None:
    pages = [queue[i:i + PAGE_SIZE] for i in range(0, len(queue), PAGE_SIZE)]
    links = []
    for pi, page in enumerate(pages, 1):
        page_key = f"{prefix}page_{pi:03d}"
        items = [{
            "id": d["ann_id"], "fn": d["fn"],
            "sugg": d["pack_sugg"], "psim": round(d["psim"], 4),
            "size": f"{int(d['bbox'][2])}×{int(d['bbox'][3])}",
            "side": d["side"], "zone": 1 if d["side"] < 60 else 0,
        } for d in page]
        html_text = PAGE_TMPL.format(
            title=f"{page_key} ({len(items)} объектов)",
            n=len(items), page_key=page_key,
            items_json=json.dumps(items, ensure_ascii=False),
            classes_json=json.dumps(PACKAGING),
        )
        (out_dir / f"{page_key}.html").write_text(html_text, encoding="utf-8")
        zone_n = sum(it["zone"] for it in items)
        links.append(f'<li><a href="{page_key}.html">{page_key}</a> — '
                     f'{len(items)} объектов (зона 40–60: {zone_n})</li>')
    (out_dir / f"{prefix}index.html").write_text(
        INDEX_TMPL.format(mode=mode, n=len(queue), npages=len(pages),
                          page_size=PAGE_SIZE, links="\n".join(links)),
        encoding="utf-8")
    print(f"  pages: {len(pages)} -> {out_dir}/{prefix}index.html")


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zeroshot", required=True,
                    help="beach_zeroshot40.json from the SigLIP pass")
    ap.add_argument("--coco", required=True)
    ap.add_argument("--image-root", required=True)
    ap.add_argument("--min-side", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=3000)
    ap.add_argument("--out", default="experiments/beachlitter_relabel/")
    ap.add_argument("--completeness-from", default=None,
                    help="verified_labels.csv -> generate supplementary pages")
    args = ap.parse_args()

    out_dir = Path(args.out) / "qa_html"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_root)

    det = load_zeroshot(Path(args.zeroshot), args.min_side)
    print(f"objects >= {args.min_side}px in zero-shot file: {len(det)}")

    if args.completeness_from:
        queue = completeness_queue(det, Path(args.completeness_from),
                                   Path(args.out) / "candidates.json")
        prefix, mode = "comp_", "completeness-проход"
    else:
        queue = build_queue(det, args.top_k)
        prefix, mode = "", "основная очередь"
    print(f"queue: {len(queue)} objects")

    render_assets(queue, image_root, out_dir)
    write_pages(queue, out_dir, prefix, mode)

    # machine-readable queue snapshot (for merge / audits)
    snap = [{k: d[k] for k in ("ann_id", "fn", "bbox", "side", "psim",
                               "pack_sugg", "pred")} for d in queue]
    (Path(args.out) / f"{prefix}candidates.json").write_text(json.dumps(snap))
    print(f"queue snapshot -> {args.out}/{prefix}candidates.json")


if __name__ == "__main__":
    main()
