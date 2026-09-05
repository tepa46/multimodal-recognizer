#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

CLASSES = ["plastic_bottle", "glass_bottle", "glass_jar", "metal_can",
           "carton", "tetrapack", "paper", "plastic_bag",
           "plastic_container", "other"]
CONDITIONS = ["normal", "heavy_fill", "foreign_object"]

FN_RE = re.compile(r"^[A-Za-z0-9_.-]+\.jpg$")

# --------------------------------------------------------------------------- #
# state container
# --------------------------------------------------------------------------- #

class Store:
    def __init__(self, root: Path):
        self.root = root
        self.state_path = root / "review_state.json"
        self.lock = threading.Lock()
        self.proposals = json.loads((root / "proposals.json").read_text())
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            ts = time.strftime("%Y%m%d-%H%M%S")
            shutil.copy2(self.state_path,
                         root / f"review_state.backup-{ts}.json")
        else:
            self.state = {"version": 1, "annotator": "", "images": {}}
        self.known = {im["file_name"] for im in self.proposals["images"]}

    def save_image(self, fn: str, image_state: dict, annotator: str) -> int:
        with self.lock:
            image_state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            self.state["images"][fn] = image_state
            if annotator:
                self.state["annotator"] = annotator
            fd, tmp = tempfile.mkstemp(dir=str(self.root), suffix=".tmp")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self.state, f, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self.state_path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            return sum(1 for v in self.state["images"].values()
                       if v.get("exhaustive"))


STORE = None  # type: Store

# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send(self, code, body, ctype="application/json", cache=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", "max-age=86400")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            self._send(200, PAGE.encode("utf-8"),
                       "text/html; charset=utf-8")
        elif self.path == "/api/bootstrap":
            payload = {
                "classes": CLASSES,
                "conditions": CONDITIONS,
                "proposals": STORE.proposals,
                "state": STORE.state,
            }
            self._send(200, json.dumps(payload).encode("utf-8"))
        elif self.path.startswith("/img/"):
            fn = os.path.basename(self.path[len("/img/"):])
            if not FN_RE.match(fn) or fn not in STORE.known:
                self._send(404, b"not found", "text/plain")
                return
            fp = STORE.root / "images" / fn
            try:
                data = fp.read_bytes()
            except OSError:
                self._send(404, b"missing", "text/plain")
                return
            self._send(200, data, "image/jpeg", cache=True)
        else:
            self._send(404, b"nope", "text/plain")

    def do_POST(self):
        if self.path != "/api/save":
            self._send(404, b"nope", "text/plain")
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
            fn = payload["fn"]
            if fn not in STORE.known:
                raise ValueError("unknown file " + fn)
            n_done = STORE.save_image(fn, payload["image"],
                                      payload.get("annotator", ""))
        except Exception as exc:  # noqa: BLE001
            self._send(400, json.dumps({"ok": False, "err": str(exc)})
                       .encode("utf-8"))
            return
        self._send(200, json.dumps({"ok": True, "n_done": n_done})
                   .encode("utf-8"))


# --------------------------------------------------------------------------- #
# the embedded single-page app
# --------------------------------------------------------------------------- #

PAGE = r"""<!DOCTYPE html>
<html lang="ru"><head><meta charset="utf-8">
<title>real495 review</title>
<style>
  :root { --bg:#14161a; --panel:#1e2128; --line:#30343d; --fg:#dfe3ea;
          --dim:#8a90a0; --acc:#4da3ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--fg);
         font:13px/1.45 -apple-system, "Segoe UI", sans-serif;
         display:flex; height:100vh; overflow:hidden; }
  #sidebar { width:230px; min-width:230px; background:var(--panel);
             border-right:1px solid var(--line); display:flex;
             flex-direction:column; }
  #head { padding:8px 10px; border-bottom:1px solid var(--line); }
  #progress { font-weight:600; }
  #filelist { flex:1; overflow-y:auto; }
  .frow { padding:3px 10px; cursor:pointer; display:flex; gap:6px;
          align-items:center; white-space:nowrap; }
  .frow:hover { background:#262a33; }
  .frow.cur { background:#2b3a52; }
  .dot { width:9px; height:9px; border-radius:50%; background:#555;
         flex:none; }
  .dot.done { background:#39c26d; }
  .dot.prog { background:#e8b93e; }
  .fmeta { color:var(--dim); margin-left:auto; font-size:11px; }
  #main { flex:1; display:flex; flex-direction:column; min-width:0; }
  #topbar { display:flex; gap:14px; align-items:center; padding:6px 12px;
            border-bottom:1px solid var(--line); background:var(--panel); }
  #topbar b { color:var(--acc); }
  #canvaswrap { flex:1; position:relative; overflow:hidden; }
  canvas { position:absolute; inset:0; cursor:default; }
  #side2 { width:270px; min-width:270px; background:var(--panel);
           border-left:1px solid var(--line); padding:10px;
           overflow-y:auto; }
  select, textarea, input[type=text] {
    width:100%; background:#12141a; color:var(--fg);
    border:1px solid var(--line); border-radius:4px; padding:4px 6px; }
  textarea { resize:vertical; min-height:52px; }
  .lbl { color:var(--dim); margin:10px 0 3px; font-size:11px;
         text-transform:uppercase; letter-spacing:.4px; }
  .hint { color:var(--dim); font-size:11px; margin-top:12px;
          border-top:1px solid var(--line); padding-top:8px; }
  .hint kbd { background:#12141a; border:1px solid var(--line);
              border-radius:3px; padding:0 4px; font-size:11px; }
  button { background:#2b3a52; color:var(--fg); border:1px solid var(--line);
           border-radius:4px; padding:5px 10px; cursor:pointer; }
  button:hover { background:#33456a; }
  #doneBtn { width:100%; margin-top:10px; font-weight:600; }
  #doneBtn.on { background:#1d5c36; }
  .sugrow { display:flex; gap:6px; flex-wrap:wrap; margin-top:4px; }
  .sug { background:#12141a; border:1px solid var(--line); border-radius:3px;
         padding:1px 6px; cursor:pointer; font-size:11px; }
  .sug:hover { border-color:var(--acc); }
  #savedot { width:8px; height:8px; border-radius:50%; background:#39c26d;
             display:inline-block; margin-left:6px; }
  #savedot.dirty { background:#e8b93e; }
</style></head><body>
<div id="sidebar">
  <div id="head">
    <div id="progress">…</div>
    <label style="font-size:11px;color:var(--dim)">annotator
      <input type="text" id="annotator" style="margin-top:2px"></label>
  </div>
  <div id="filelist"></div>
</div>
<div id="main">
  <div id="topbar">
    <b id="curfn">—</b>
    <span id="boxcount"></span>
    <span style="margin-left:auto"></span>
    <label><input type="checkbox" id="ghostToggle"> ghosts (T)</label>
    <span>сохранено<span id="savedot"></span></span>
  </div>
  <div id="canvaswrap"><canvas id="cv"></canvas></div>
</div>
<div id="side2">
  <div class="lbl">Выбранный бокс</div>
  <div id="boxinfo" style="color:var(--dim)">не выбран</div>
  <div class="lbl">Класс <span style="text-transform:none">(1-9, 0)</span></div>
  <select id="clsSel"><option value="">— выбери класс —</option></select>
  <div class="sugrow" id="sugRow"></div>
  <div class="lbl">Condition (C)</div>
  <select id="condSel"></select>
  <div class="lbl">Флаги</div>
  <label><input type="checkbox" id="unsureChk"> unsure (U)</label>
  <div class="lbl">Notes (;)</div>
  <textarea id="notesTa"></textarea>
  <button id="delBtn" style="margin-top:8px">Удалить бокс (X)</button>
  <button id="doneBtn">✔ Размечено исчерпывающе (E)</button>
  <div class="hint">
    <kbd>N</kbd>+drag новый бокс · <kbd>Tab</kbd> следующий бокс ·
    <kbd>A</kbd> принять ghost под курсором · <kbd>T</kbd> ghosts ·
    <kbd>E</kbd> done → next · <kbd>⇧E</kbd> снять done ·
    <kbd>←</kbd><kbd>→</kbd> кадры · <kbd>F</kbd> fit ·
    колесо zoom · drag по фону pan
  </div>
</div>
<script>
"use strict";
const COLORS = {plastic_bottle:"#4da3ff", glass_bottle:"#39c26d",
  glass_jar:"#20c5c5", metal_can:"#e8b93e", carton:"#c07840",
  tetrapack:"#b34dff", paper:"#e0e0e0", plastic_bag:"#ff6ea9",
  plastic_container:"#7dd649", other:"#9a9a9a"};
const GHOST_THR = 0.3;

let CLASSES=[], CONDITIONS=[], PROPS=null, STATE=null;
let files=[], cur=-1, sel=-1;          // current image idx, selected box idx
let img=new Image(), imgReady=false;
let scale=1, ox=0, oy=0;               // view transform
let drawMode=false, drag=null, mouse={x:0,y:0};
let showGhosts=false, dirty=false, saveTimer=null;

const cv=document.getElementById("cv"), ctx=cv.getContext("2d");
const wrap=document.getElementById("canvaswrap");

function el(id){ return document.getElementById(id); }
function curFn(){ return cur>=0 ? files[cur].file_name : null; }
function curImgState(){ return STATE.images[curFn()]; }

// ---------- bootstrap ----------
fetch("/api/bootstrap").then(r=>r.json()).then(b=>{
  CLASSES=b.classes; CONDITIONS=b.conditions; PROPS=b.proposals; STATE=b.state;
  files=PROPS.images.slice().sort((a,z)=>a.file_name<z.file_name?-1:1);
  const cs=el("clsSel");
  CLASSES.forEach(c=>{const o=document.createElement("option");
    o.value=c;o.textContent=c;cs.appendChild(o);});
  const cd=el("condSel");
  CONDITIONS.forEach(c=>{const o=document.createElement("option");
    o.value=c;o.textContent=c;cd.appendChild(o);});
  el("annotator").value=STATE.annotator||"";
  buildList();
  let start=files.findIndex(f=>!(STATE.images[f.file_name]||{}).exhaustive);
  selectImage(start<0?0:start);
});

function ensureImgState(fn){
  if(STATE.images[fn]) return STATE.images[fn];
  const meta=files.find(f=>f.file_name===fn);
  const boxes=[]; let bid=1;
  (meta.detections||[]).forEach(d=>{
    if(d.score>=GHOST_THR){
      boxes.push({bid:bid++, bbox:d.bbox.slice(), class:d.siglip_sugg||d.det_class,
        condition:"normal", unsure:false, notes:"", origin:"detector",
        det_id:d.det_id, det_score:d.score, det_class:d.det_class,
        siglip_sugg:d.siglip_sugg, siglip_pred:d.siglip_pred,
        suspicious:(d.siglip_pred||"").startsWith("_")});
    }});
  STATE.images[fn]={status:boxes.length?"in_progress":"todo",
    exhaustive:false, next_bid:bid, boxes:boxes};
  return STATE.images[fn];
}

// ---------- sidebar ----------
function buildList(){
  const fl=el("filelist"); fl.innerHTML="";
  files.forEach((f,i)=>{
    const st=STATE.images[f.file_name];
    const row=document.createElement("div");
    row.className="frow"+(i===cur?" cur":"");
    row.dataset.i=i;
    const dot=document.createElement("span");
    dot.className="dot"+(st?(st.exhaustive?" done":" prog"):"");
    const name=document.createElement("span");
    name.textContent=f.file_name.replace(".jpg","");
    const meta=document.createElement("span");
    meta.className="fmeta";
    const nb=st?st.boxes.length:(f.detections||[]).filter(d=>d.score>=GHOST_THR).length;
    meta.textContent=nb+(st&&st.boxes.some(b=>b.unsure)?" ?":"");
    row.append(dot,name,meta);
    row.onclick=()=>selectImage(i);
    fl.appendChild(row);
  });
  updateProgress();
}
function refreshRow(i){
  const row=document.querySelector('.frow[data-i="'+i+'"]');
  if(!row) return;
  const st=STATE.images[files[i].file_name];
  row.querySelector(".dot").className=
    "dot"+(st?(st.exhaustive?" done":" prog"):"");
  const nb=st?st.boxes.length:0;
  row.querySelector(".fmeta").textContent=
    nb+(st&&st.boxes.some(b=>b.unsure)?" ?":"");
  row.classList.toggle("cur", i===cur);
}
function updateProgress(){
  const done=Object.values(STATE.images).filter(v=>v.exhaustive).length;
  el("progress").textContent="done "+done+" / "+files.length;
}

// ---------- image switching ----------
function selectImage(i){
  if(cur>=0){ flushSave(); }
  const prev=cur; cur=i; sel=-1;
  ensureImgState(curFn());
  imgReady=false;
  img=new Image();
  img.onload=()=>{ imgReady=true; fit(); render(); };
  img.src="/img/"+curFn();
  el("curfn").textContent=curFn();
  if(prev>=0) refreshRow(prev);
  refreshRow(cur);
  const row=document.querySelector('.frow[data-i="'+i+'"]');
  if(row) row.scrollIntoView({block:"nearest"});
  syncPanel(); updateDoneBtn(); render();
}
function fit(){
  const st=curImgState(); const meta=files[cur];
  const cw=wrap.clientWidth, ch=wrap.clientHeight;
  scale=Math.min(cw/meta.width, ch/meta.height)*0.98;
  ox=(cw-meta.width*scale)/2; oy=(ch-meta.height*scale)/2;
}

// ---------- rendering ----------
function i2s(x,y){ return [x*scale+ox, y*scale+oy]; }
function s2i(x,y){ return [(x-ox)/scale, (y-oy)/scale]; }

function render(){
  const cw=wrap.clientWidth, ch=wrap.clientHeight;
  if(cv.width!==cw||cv.height!==ch){ cv.width=cw; cv.height=ch; }
  ctx.fillStyle="#101216"; ctx.fillRect(0,0,cw,ch);
  if(!imgReady){ requestAnimationFrame(render); return; }
  ctx.imageSmoothingEnabled=true;
  ctx.drawImage(img, ox, oy, img.naturalWidth*scale, img.naturalHeight*scale);
  const st=curImgState(); if(!st) return;

  if(showGhosts){
    ghostList().forEach(g=>{
      const [x,y]=i2s(g.bbox[0],g.bbox[1]);
      ctx.setLineDash([6,5]); ctx.lineWidth=1.4;
      ctx.strokeStyle="rgba(200,200,220,.65)";
      ctx.strokeRect(x,y,g.bbox[2]*scale,g.bbox[3]*scale);
      ctx.setLineDash([]);
      label(x,y-4,g.det_class+" "+g.score.toFixed(2)+" (ghost)","rgba(180,180,200,.8)","#222");
    });
  }
  st.boxes.forEach((b,i)=>{
    const [x,y]=i2s(b.bbox[0],b.bbox[1]);
    const w=b.bbox[2]*scale, h=b.bbox[3]*scale;
    const col=b.class?COLORS[b.class]:"#ff5252";
    ctx.lineWidth=i===sel?3:2;
    if(b.suspicious&&i!==sel){ ctx.setLineDash([8,4]); }
    ctx.strokeStyle=col; ctx.strokeRect(x,y,w,h);
    ctx.setLineDash([]);
    let t=(b.class||"??");
    if(b.unsure) t+=" ?";
    if(b.condition&&b.condition!=="normal") t+=" ["+b.condition+"]";
    label(x,y-4,t,col,"#000");
    if(b.origin==="detector"){
      label(x, y+h+12,
        "det:"+b.det_class+" "+(b.det_score||0).toFixed(2)+
        (b.siglip_sugg?"  sig:"+b.siglip_sugg:""),
        "rgba(255,255,255,.55)","#000");
    }
    if(i===sel){
      handles(b).forEach(hd=>{
        ctx.fillStyle="#fff";
        ctx.fillRect(hd.sx-4,hd.sy-4,8,8);
        ctx.strokeStyle="#000"; ctx.lineWidth=1;
        ctx.strokeRect(hd.sx-4,hd.sy-4,8,8);
      });
    }
  });
  if(drag&&drag.mode==="new"){
    ctx.strokeStyle="#ff5252"; ctx.lineWidth=2; ctx.setLineDash([4,4]);
    const r=newRect(); ctx.strokeRect(r[0]*scale+ox,r[1]*scale+oy,r[2]*scale,r[3]*scale);
    ctx.setLineDash([]);
  }
  el("boxcount").textContent=st.boxes.length+" боксов"+
    (st.exhaustive?"  ✔ done":"");
}
function label(x,y,text,bg,fg){
  ctx.font="12px -apple-system,sans-serif";
  const w=ctx.measureText(text).width+8;
  ctx.fillStyle=bg; ctx.fillRect(x,y-12,w,14);
  ctx.fillStyle=fg; ctx.fillText(text,x+4,y-1);
}
function handles(b){
  const [x,y]=i2s(b.bbox[0],b.bbox[1]);
  const w=b.bbox[2]*scale, h=b.bbox[3]*scale;
  const pts=[[0,0,"nw"],[.5,0,"n"],[1,0,"ne"],[0,.5,"w"],[1,.5,"e"],
             [0,1,"sw"],[.5,1,"s"],[1,1,"se"]];
  return pts.map(p=>({sx:x+w*p[0], sy:y+h*p[1], k:p[2]}));
}
function ghostList(){
  // ghost = любая детекция, не представленная боксом в стейте:
  // низкоскоровые предложения + удалённые детекторные боксы (natural undo)
  const st=curImgState(); const meta=files[cur];
  const used=new Set(st.boxes.filter(b=>b.det_id!==undefined&&b.det_id!==null)
                             .map(b=>b.det_id));
  return (meta.detections||[]).filter(d=>!used.has(d.det_id));
}

// ---------- mutations ----------
function touch(){
  const st=curImgState();
  if(!st.exhaustive) st.status="in_progress";
  dirty=true; el("savedot").className="dirty";
  clearTimeout(saveTimer); saveTimer=setTimeout(flushSave,400);
  render(); refreshRow(cur);
}
function flushSave(){
  if(!dirty||cur<0) return;
  dirty=false;
  fetch("/api/save",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({fn:curFn(), image:curImgState(),
                         annotator:el("annotator").value.trim()})})
    .then(r=>r.json()).then(r=>{
      if(r.ok){ el("savedot").className=""; updateProgress(); }
      else alert("save error: "+r.err);
    }).catch(e=>alert("save failed: "+e));
}
window.addEventListener("beforeunload", flushSave);

function addBox(bbox, extra){
  const st=curImgState();
  const b=Object.assign({bid:st.next_bid++, bbox:bbox, class:null,
    condition:"normal", unsure:false, notes:"", origin:"manual"}, extra||{});
  st.boxes.push(b); sel=st.boxes.length-1;
  syncPanel(); touch();
  return b;
}
function delBox(){
  const st=curImgState();
  if(sel<0||sel>=st.boxes.length) return;
  st.boxes.splice(sel,1); sel=-1; syncPanel(); touch();
}
function setClass(c){
  const st=curImgState();
  if(sel<0) return;
  st.boxes[sel].class=c; syncPanel(); touch();
}

// ---------- right panel ----------
function syncPanel(){
  const st=curImgState();
  const b=(st&&sel>=0)?st.boxes[sel]:null;
  el("boxinfo").textContent=b?("#"+b.bid+"  "+b.origin+
    (b.det_score?("  det:"+b.det_class+" "+b.det_score.toFixed(2)):"")+
    "  "+Math.round(b.bbox[2])+"×"+Math.round(b.bbox[3])):"не выбран";
  el("clsSel").value=b&&b.class?b.class:"";
  el("condSel").value=b?b.condition:"normal";
  el("unsureChk").checked=b?!!b.unsure:false;
  el("notesTa").value=b?b.notes:"";
  const sr=el("sugRow"); sr.innerHTML="";
  if(b){
    const meta=files[cur];
    const d=(meta.detections||[]).find(d=>d.det_id===b.det_id);
    (d&&d.siglip_top3?d.siglip_top3:[]).forEach(([c,s])=>{
      const sp=document.createElement("span");
      sp.className="sug"; sp.textContent=c+" "+s.toFixed(3);
      sp.onclick=()=>setClass(c);
      sr.appendChild(sp);
    });
  }
  updateDoneBtn();
}
function updateDoneBtn(){
  const st=curImgState();
  const btn=el("doneBtn");
  btn.className=st&&st.exhaustive?"on":"";
  btn.textContent=st&&st.exhaustive?
    "✔ done — снять (⇧E)":"✔ Размечено исчерпывающе (E)";
}
el("clsSel").onchange=e=>setClass(e.target.value||null);
el("condSel").onchange=e=>{const st=curImgState();
  if(sel>=0){st.boxes[sel].condition=e.target.value; touch();}};
el("unsureChk").onchange=e=>{const st=curImgState();
  if(sel>=0){st.boxes[sel].unsure=e.target.checked; touch();}};
el("notesTa").oninput=e=>{const st=curImgState();
  if(sel>=0){st.boxes[sel].notes=e.target.value; touch();}};
el("delBtn").onclick=delBox;
el("doneBtn").onclick=()=>toggleDone(false);
el("ghostToggle").onchange=e=>{showGhosts=e.target.checked; render();};
el("annotator").onchange=()=>{dirty=true; flushSave();};

function toggleDone(forceOff){
  const st=curImgState();
  if(st.exhaustive||forceOff){ st.exhaustive=false; st.status="in_progress"; }
  else{
    const noClass=st.boxes.filter(b=>!b.class).length;
    if(noClass){ alert(noClass+" бокс(а) без класса — назначь класс"); return; }
    st.exhaustive=true; st.status="done";
  }
  touch(); flushSave(); updateDoneBtn();
  if(st.exhaustive){
    let nxt=files.findIndex((f,i)=>i>cur&&!(STATE.images[f.file_name]||{}).exhaustive);
    if(nxt<0) nxt=files.findIndex(f=>!(STATE.images[f.file_name]||{}).exhaustive);
    if(nxt>=0) selectImage(nxt);
  }
}

// ---------- mouse ----------
function hitTest(sx,sy){
  const st=curImgState(); if(!st) return {type:"empty"};
  if(sel>=0&&sel<st.boxes.length){
    for(const hd of handles(st.boxes[sel]))
      if(Math.abs(sx-hd.sx)<=6&&Math.abs(sy-hd.sy)<=6)
        return {type:"handle",k:hd.k};
  }
  const [ix,iy]=s2i(sx,sy);
  let best=-1, bestA=1e18;
  st.boxes.forEach((b,i)=>{
    const [x,y,w,h]=b.bbox;
    if(ix>=x&&ix<=x+w&&iy>=y&&iy<=y+h&&w*h<bestA){best=i;bestA=w*h;}
  });
  if(best>=0) return {type:"box",i:best};
  if(showGhosts){
    for(const g of ghostList()){
      const [x,y,w,h]=g.bbox;
      if(ix>=x&&ix<=x+w&&iy>=y&&iy<=y+h) return {type:"ghost",g:g};
    }
  }
  return {type:"empty"};
}
function newRect(){
  const x1=Math.min(drag.ix0,drag.ix1), y1=Math.min(drag.iy0,drag.iy1);
  return [x1,y1,Math.abs(drag.ix1-drag.ix0),Math.abs(drag.iy1-drag.iy0)];
}
cv.addEventListener("mousedown",e=>{
  const sx=e.offsetX, sy=e.offsetY;
  const [ix,iy]=s2i(sx,sy);
  if(drawMode){
    drag={mode:"new",ix0:ix,iy0:iy,ix1:ix,iy1:iy}; return;
  }
  const h=hitTest(sx,sy);
  if(h.type==="handle"){
    const b=curImgState().boxes[sel];
    drag={mode:"resize",k:h.k,orig:b.bbox.slice()};
  }else if(h.type==="box"){
    sel=h.i; syncPanel();
    const b=curImgState().boxes[sel];
    drag={mode:"move",ix0:ix,iy0:iy,orig:b.bbox.slice()};
    render();
  }else if(h.type==="ghost"){
    acceptGhost(h.g);
  }else{
    sel=-1; syncPanel();
    drag={mode:"pan",sx0:sx,sy0:sy,ox0:ox,oy0:oy};
    render();
  }
});
cv.addEventListener("mousemove",e=>{
  mouse={x:e.offsetX,y:e.offsetY};
  if(!drag) return;
  const [ix,iy]=s2i(e.offsetX,e.offsetY);
  if(drag.mode==="pan"){
    ox=drag.ox0+(e.offsetX-drag.sx0); oy=drag.oy0+(e.offsetY-drag.sy0);
  }else if(drag.mode==="new"){
    drag.ix1=ix; drag.iy1=iy;
  }else if(drag.mode==="move"){
    const b=curImgState().boxes[sel];
    b.bbox[0]=drag.orig[0]+(ix-drag.ix0);
    b.bbox[1]=drag.orig[1]+(iy-drag.iy0);
  }else if(drag.mode==="resize"){
    const b=curImgState().boxes[sel];
    const o=drag.orig; let x1=o[0],y1=o[1],x2=o[0]+o[2],y2=o[1]+o[3];
    if(drag.k.includes("w")) x1=Math.min(ix,x2-4);
    if(drag.k.includes("e")) x2=Math.max(ix,x1+4);
    if(drag.k.includes("n")) y1=Math.min(iy,y2-4);
    if(drag.k.includes("s")) y2=Math.max(iy,y1+4);
    b.bbox=[x1,y1,x2-x1,y2-y1];
  }
  render();
});
window.addEventListener("mouseup",()=>{
  if(!drag) return;
  const was=drag; drag=null;
  if(was.mode==="new"){
    drawMode=false; cv.style.cursor="default";
    const r=newRect();
    if(r[2]>6&&r[3]>6) addBox(r);
    else render();
  }else if(was.mode==="move"||was.mode==="resize"){
    touch();
  }
});
cv.addEventListener("wheel",e=>{
  e.preventDefault();
  const f=Math.exp(-e.deltaY*0.0015);
  const ns=Math.min(8,Math.max(0.05,scale*f));
  const [ix,iy]=s2i(e.offsetX,e.offsetY);
  ox=e.offsetX-ix*ns; oy=e.offsetY-iy*ns; scale=ns;
  render();
},{passive:false});
cv.addEventListener("dblclick",e=>{
  const h=hitTest(e.offsetX,e.offsetY);
  if(h.type==="ghost") acceptGhost(h.g);
});
function acceptGhost(g){
  addBox(g.bbox.slice(),{origin:"detector",det_id:g.det_id,
    det_score:g.score,det_class:g.det_class,
    siglip_sugg:g.siglip_sugg,siglip_pred:g.siglip_pred,
    class:g.siglip_sugg||g.det_class,
    suspicious:(g.siglip_pred||"").startsWith("_")});
}

// ---------- keyboard ----------
window.addEventListener("keydown",e=>{
  const tag=(e.target.tagName||"").toLowerCase();
  if(tag==="input"||tag==="textarea"||tag==="select"){
    if(e.key==="Escape") e.target.blur();
    return;
  }
  const st=curImgState();
  if(e.key>="0"&&e.key<="9"&&sel>=0){
    const k=e.key==="0"?9:parseInt(e.key,10)-1;
    if(k<CLASSES.length) setClass(CLASSES[k]);
    e.preventDefault(); return;
  }
  switch(e.key){
    case "Tab": {
      if(!st||!st.boxes.length) break;
      sel=(sel+1)%st.boxes.length; syncPanel(); render();
      e.preventDefault(); break; }
    case "n": case "N": case "т": case "Т":
      drawMode=!drawMode; cv.style.cursor=drawMode?"crosshair":"default";
      break;
    case "x": case "X": case "ч": case "Ч":
    case "Delete": case "Backspace":
      delBox(); e.preventDefault(); break;
    case "u": case "U": case "г": case "Г":
      if(sel>=0){st.boxes[sel].unsure=!st.boxes[sel].unsure;
        syncPanel(); touch();} break;
    case "c": case "C": case "с": case "С":
      if(sel>=0){const i=CONDITIONS.indexOf(st.boxes[sel].condition);
        st.boxes[sel].condition=CONDITIONS[(i+1)%CONDITIONS.length];
        syncPanel(); touch();} break;
    case ";": case "ж": case "Ж":
      el("notesTa").focus(); e.preventDefault(); break;
    case "t": case "T": case "е": case "Е":
      showGhosts=!showGhosts; el("ghostToggle").checked=showGhosts;
      render(); break;
    case "a": case "A": case "ф": case "Ф": {
      const h=hitTest(mouse.x,mouse.y);
      if(h.type==="ghost") acceptGhost(h.g);
      break; }
    case "e": case "E": case "у": case "У":
      toggleDone(e.shiftKey); e.preventDefault(); break;
    case "f": case "F": case "а": case "А":
      fit(); render(); break;
    case "ArrowLeft":
      if(cur>0) selectImage(cur-1); e.preventDefault(); break;
    case "ArrowRight":
      if(cur<files.length-1) selectImage(cur+1); e.preventDefault(); break;
  }
});
new ResizeObserver(()=>{ if(imgReady){ render(); } }).observe(wrap);
</script></body></html>
"""


def main() -> None:
    global STORE
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/real495")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    root = Path(args.root)
    if not (root / "proposals.json").exists():
        raise SystemExit(f"{root}/proposals.json not found — run "
                         "real495_detect.py + real495_siglip.py first")
    STORE = Store(root)
    n_img = len(STORE.proposals["images"])
    n_done = sum(1 for v in STORE.state["images"].values()
                 if v.get("exhaustive"))
    print(f"[review] {n_img} images, {n_done} done")
    print(f"[review] state file: {STORE.state_path}")
    print(f"[review] open http://127.0.0.1:{args.port}")
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
