"""Local web UI for MIDI-AI Studio.

Run with:  python -m midi_ai.web

Serves a small single-page app: drag & drop a .mid file, tweak options, and
render + audition the result in the browser.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import re
import tempfile
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote

from .pipeline import RenderOptions, render_file

HOST = os.environ.get("MIDI_AI_HOST", "127.0.0.1")
PORT = int(os.environ.get("MIDI_AI_PORT", "8765"))

# Hard limits so a small server cannot be overwhelmed by concurrent renders.
MAX_UPLOAD = int(os.environ.get("MIDI_AI_MAX_UPLOAD", str(4 * 1024 * 1024)))
MAX_PENDING = int(os.environ.get("MIDI_AI_MAX_PENDING", "6"))
WORKERS = max(1, int(os.environ.get("MIDI_AI_WORKERS", "1")))
MAX_OUT_FILES = int(os.environ.get("MIDI_AI_MAX_OUT_FILES", "40"))
OUT_TTL = int(os.environ.get("MIDI_AI_OUT_TTL", "3600"))
MAX_DURATION = float(os.environ.get("MIDI_AI_MAX_DURATION", "300"))

_OUT_DIR = tempfile.mkdtemp(prefix="midiai_out_")
_JOBS: dict = {}
_JOBS_LOCK = threading.Lock()
_QUEUE: queue.Queue = queue.Queue(maxsize=MAX_PENDING)
_WORKERS_STARTED = False


def _cleanup_out_dir():
    """Drop old renders so the container's writable layer cannot fill up."""
    now = time.time()
    try:
        entries = [
            (os.path.join(_OUT_DIR, n), os.path.getmtime(os.path.join(_OUT_DIR, n)))
            for n in os.listdir(_OUT_DIR)
        ]
    except OSError:
        return
    for path, mtime in entries:
        if now - mtime > OUT_TTL:
            try:
                os.remove(path)
            except OSError:
                pass
    remaining = sorted(
        ((p, m) for p, m in entries if os.path.exists(p)), key=lambda x: x[1]
    )
    if len(remaining) > MAX_OUT_FILES:
        for path, _ in remaining[: len(remaining) - MAX_OUT_FILES]:
            try:
                os.remove(path)
            except OSError:
                pass


def _prune_jobs():
    now = time.time()
    with _JOBS_LOCK:
        stale = [jid for jid, j in _JOBS.items()
                 if j.get("status") in ("done", "error") and now - j.get("updated", now) > OUT_TTL]
        for jid in stale:
            _JOBS.pop(jid, None)


def _safe_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    name = re.sub(r"_+", "_", name)
    return (name or "output")[:80]


def _set_job(job_id: str, **fields):
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(fields)
            job["updated"] = time.time()


def _worker_loop():
    # Lower the render thread's priority so HTTP/health checks stay snappy
    # while a heavy render is running (POSIX only).
    try:
        os.nice(10)
    except (AttributeError, OSError):
        pass
    while True:
        job_id, in_path, out_path, opts, out_name = _QUEUE.get()
        try:
            _set_job(job_id, status="running", started=time.time())
            t0 = time.perf_counter()
            opts.progress = lambda frac, jid=job_id: _set_job(jid, progress=float(frac))
            result = render_file(in_path, out_path, opts)
            _set_job(
                job_id,
                status="done",
                progress=1.0,
                url=f"/out/{out_name}",
                outname=out_name,
                duration=round(result.duration, 2),
                bpm=result.bpm,
                model_used=result.model_used,
                elapsed=round(time.perf_counter() - t0, 2),
                tracks=result.tracks,
            )
            # The input file is no longer needed.
            try:
                os.remove(in_path)
            except OSError:
                pass
        except Exception as e:  # noqa: BLE001
            _set_job(job_id, status="error", error=str(e), progress=1.0)
        finally:
            _cleanup_out_dir()
            _prune_jobs()
            _QUEUE.task_done()


def _start_workers():
    global _WORKERS_STARTED
    if _WORKERS_STARTED:
        return
    _WORKERS_STARTED = True
    for _ in range(WORKERS):
        threading.Thread(target=_worker_loop, daemon=True, name="midi-render").start()


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MIDI-AI Studio</title>
<style>
:root{--bg:#0e1117;--panel:#161b25;--panel2:#1c2230;--text:#e6e9ef;--muted:#8b93a7;
--accent:#5b8cff;--accent2:#22d3a5;--border:#2a3142;--radius:14px}
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
background:radial-gradient(1200px 700px at 20% -10%,#16203a 0%,var(--bg) 60%);color:var(--text);min-height:100vh}
.wrap{max-width:980px;margin:0 auto;padding:32px 20px 80px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:.3px}
h1 .ai{color:var(--accent)}
.sub{color:var(--muted);margin:0 0 28px}
.grid{display:grid;grid-template-columns:1fr 340px;gap:20px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:20px}
.drop{border:2px dashed var(--border);border-radius:var(--radius);padding:44px 20px;text-align:center;
cursor:pointer;transition:.15s;background:var(--panel2)}
.drop.hover,.drop:hover{border-color:var(--accent);background:#1a2236}
.drop .big{font-size:44px;margin-bottom:8px}
.drop .hint{color:var(--muted)}
#fileinfo{margin-top:14px;font-size:14px;color:var(--accent2);display:none}
label{display:block;font-size:13px;color:var(--muted);margin:16px 0 6px}
.row{display:flex;justify-content:space-between;font-size:13px}
.row b{color:var(--text)}
input[type=range],select{width:100%;accent-color:var(--accent)}
button{width:100%;margin-top:20px;padding:13px;font-size:16px;font-weight:600;border:0;border-radius:10px;
background:linear-gradient(90deg,var(--accent),#7a5bff);color:#fff;cursor:pointer;transition:.15s}
button:hover{filter:brightness(1.1)}
button:disabled{opacity:.5;cursor:wait}
.status{margin-top:16px;font-size:14px;color:var(--muted);min-height:20px}
.status.ok{color:var(--accent2)}
.player{margin-top:18px;display:none}
audio{width:100%}
.dl{margin-top:14px;display:flex;gap:10px}
.dl a{flex:1;text-align:center;padding:10px;background:var(--panel2);border:1px solid var(--border);
border-radius:10px;color:var(--text);text-decoration:none;font-weight:600}
.dl a:hover{border-color:var(--accent)}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}
th,td{text-align:left;padding:7px 8px;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-weight:600}
.stats{display:flex;gap:14px;flex-wrap:wrap;margin-top:12px;font-size:13px;color:var(--muted)}
.stats b{color:var(--text);font-size:15px}
.progress{margin-top:12px;height:8px;background:var(--panel2);border-radius:4px;overflow:hidden;display:none}
.progress>div{height:100%;background:linear-gradient(90deg,var(--accent),#7a5bff);width:0%}
</style>
</head>
<body>
<div class="wrap">
  <h1>MIDI-AI <span class="ai">Studio</span></h1>
  <p class="sub">Reproduce tus archivos MIDI con un sonido casi real: interpretaci&oacute;n
  expresiva por IA + s&iacute;ntesis de alta calidad + mezcla profesional.</p>

  <div class="grid">
    <div>
      <div class="card">
        <div class="drop" id="drop">
          <div class="big">&#127926;</div>
          <div>Arrastra un archivo <b>.mid</b> aqu&iacute; o haz clic para elegirlo</div>
          <div class="hint">tambi&eacute;n .midi</div>
        </div>
        <input type="file" id="file" accept=".mid,.midi" hidden/>
        <div id="fileinfo"></div>

        <div class="player" id="player">
          <audio id="audio" controls></audio>
          <div class="dl">
            <a id="download" href="#" download>Descargar WAV</a>
          </div>
          <div class="stats" id="stats"></div>
          <div style="margin-top:14px">
            <div class="row"><span>Pistas</span></div>
            <table id="tracks"></table>
          </div>
        </div>
      </div>
    </div>

    <div>
      <div class="card">
        <label>Preset de g&eacute;nero</label>
        <select id="preset">
          <option value="">Personalizado</option>
          <option value="classical">Cl&aacute;sico</option>
          <option value="jazz">Jazz</option>
          <option value="pop">Pop</option>
          <option value="cinematic">Cinem&aacute;tico</option>
          <option value="neutral">Neutral</option>
        </select>

        <label>Intensidad de IA (expresi&oacute;n humana)</label>
        <div class="row"><span>0%</span><b id="v_ai">60%</b><span>100%</span></div>
        <input type="range" id="ai" min="0" max="1" step="0.05" value="0.6"/>

        <label>Reverberaci&oacute;n</label>
        <div class="row"><span>seca</span><b id="v_reverb">30%</b><span>sala</span></div>
        <input type="range" id="reverb" min="0" max="1" step="0.01" value="0.30"/>

        <label>Brillo</label>
        <div class="row"><span>-6 dB</span><b id="v_brightness">0 dB</b><span>+6 dB</span></div>
        <input type="range" id="brightness" min="-6" max="6" step="0.5" value="0"/>

        <label>Calidez</label>
        <div class="row"><span>-6 dB</span><b id="v_warmth">0 dB</b><span>+6 dB</span></div>
        <input type="range" id="warmth" min="-6" max="6" step="0.5" value="0"/>

        <label>Amplitud est&eacute;reo</label>
        <div class="row"><span>mono</span><b id="v_width">0%</b><span>amplio</span></div>
        <input type="range" id="width" min="0" max="1" step="0.05" value="0"/>

        <label>Volumen (RMS)</label>
        <div class="row"><span>suave</span><b id="v_loudness">-15 dB</b><span>fuerte</span></div>
        <input type="range" id="loudness" min="-24" max="-8" step="1" value="-15"/>

        <label>Saturaci&oacute;n (cinta)</label>
        <div class="row"><span>off</span><b id="v_saturation">0.5 dB</b><span>3 dB</span></div>
        <input type="range" id="saturation" min="0" max="3" step="0.5" value="0.5"/>

        <button id="renderBtn" disabled>Renderizar</button>
        <div class="progress" id="progress"><div id="pbar"></div></div>
        <div class="status" id="status"></div>
      </div>
    </div>
  </div>
</div>

<script>
const $=id=>document.getElementById(id);
const state={file:null,job:null};
const drop=$('drop'),fileEl=$('file');
drop.onclick=()=>fileEl.click();
fileEl.onchange=e=>{if(e.target.files[0])setFile(e.target.files[0])};
['dragover','dragenter'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.add('hover')}));
['dragleave','drop'].forEach(ev=>drop.addEventListener(ev,e=>{e.preventDefault();drop.classList.remove('hover')}));
drop.addEventListener('drop',e=>{const f=e.dataTransfer.files[0];if(f)setFile(f)});
function setFile(f){
  if(!/\.midi?$/i.test(f.name)){status('Formato no v&aacute;lido','err');return}
  state.file=f;
  $('fileinfo').style.display='block';
  $('fileinfo').innerHTML='&#128220; '+f.name+' &mdash; '+(f.size/1024).toFixed(1)+' KB';
  $('renderBtn').disabled=false;
  status('Listo para renderizar');
}
['ai','reverb','brightness','warmth','width','loudness','saturation'].forEach(id=>{
  const el=$(id);
  const fmt=id==='loudness'?v=>v+' dB':id==='brightness'||id==='warmth'||id==='saturation'?v=>(v>0?'+':'')+v+' dB':v=>Math.round(v*100)+'%';
  const upd=()=>$('v_'+id).textContent=fmt(el.value);
  el.addEventListener('input',upd);upd();
});
const PRESETS={classical:{ai:.7,reverb:.35,brightness:0,warmth:1,width:.15,loudness:-16,saturation:.2},
 jazz:{ai:.65,reverb:.25,brightness:0,warmth:2,width:.1,loudness:-14,saturation:.4},
 pop:{ai:.45,reverb:.2,brightness:2,warmth:0,width:.25,loudness:-12,saturation:.6},
 cinematic:{ai:.75,reverb:.45,brightness:0,warmth:1.5,width:.4,loudness:-16,saturation:.3},
 neutral:{ai:0,reverb:.2,brightness:0,warmth:0,width:0,loudness:-15,saturation:0}};
$('preset').onchange=e=>{
  const p=PRESETS[e.target.value]; if(!p) return;
  $('ai').value=p.ai; $('reverb').value=p.reverb; $('brightness').value=p.brightness;
  $('warmth').value=p.warmth; $('width').value=p.width; $('loudness').value=p.loudness;
  $('saturation').value=p.saturation;
  ['ai','reverb','brightness','warmth','width','loudness','saturation'].forEach(id=>{
    const el=$(id); const upd=()=>$('v_'+id).textContent=
      id==='loudness'?el.value+' dB':(id==='brightness'||id==='warmth'||id==='saturation')?(el.value>0?'+':'')+el.value+' dB':Math.round(el.value*100)+'%';
    upd();
  });
};
function status(msg,cls){const s=$('status');s.textContent=msg;s.className='status'+(cls?' '+cls:'');}
function opts(){return{
  ai_strength:parseFloat($('ai').value),
  reverb:parseFloat($('reverb').value),
  brightness:parseFloat($('brightness').value),
  warmth:parseFloat($('warmth').value),
  width:parseFloat($('width').value),
  loudness_db:parseFloat($('loudness').value),
  saturation:parseFloat($('saturation').value),
  sample_rate:44100,seed:42,use_model:parseFloat($('ai').value)>0
};}
$('renderBtn').onclick=async()=>{
  if(!state.file)return;
  const btn=$('renderBtn');btn.disabled=true;
  $('progress').style.display='block'; $('pbar').style.width='0%';
  status('Enviando archivo...');
  const buf=await state.file.arrayBuffer();
  const b64=btoa(String.fromCharCode(...new Uint8Array(buf)));
  try{
    const r=await fetch('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({name:state.file.name,data:b64,options:opts()})});
    const j=await r.json();
    if(!j.ok){status('Error: '+(j.error||'desconocido'),'err');return}
    state.job=j.job_id;
    poll();
  }catch(e){status('Error: '+e.message,'err');btn.disabled=false}
};
function poll(){
  if(!state.job)return;
  fetch('/api/status?job='+state.job).then(r=>r.json()).then(j=>{
    if(j.progress!=null){$('pbar').style.width=Math.round(j.progress*100)+'%';}
    if(j.status==='queued'){
      status('En cola ('+(j.queue_size||0)+' pendientes)...');
      setTimeout(poll,1000);
    } else if(j.status==='running'){
      status('Renderizando... '+Math.round((j.progress||0)*100)+'%');
      setTimeout(poll,800);
    } else if(j.status==='done'){
      const a=$('audio');a.src=j.url;a.onloadedmetadata=()=>a.play();
      $('player').style.display='block';
      $('download').href=j.url;
      $('download').setAttribute('download',j.outname||'output.wav');
      $('stats').innerHTML='<span>Duraci&oacute;n <b>'+j.duration+' s</b></span>'+
        '<span>BPM <b>'+j.bpm+'</b></span>'+
        '<span>IA <b>'+(j.model_used?'s&iacute;':'no')+'</b></span>'+
        '<span>Tiempo <b>'+j.elapsed+' s</b></span>';
      const tb=$('tracks');tb.innerHTML='<tr><th>Instrumento</th><th>Notas</th><th>Pista</th></tr>'+
        j.tracks.map(t=>'<tr><td>'+t.instrument+'</td><td>'+t.notes+'</td><td>'+t.name+'</td></tr>').join('');
      status('Renderizado correctamente','ok');
      $('renderBtn').disabled=false;
      state.job=null;
    } else {
      status('Error: '+(j.error||'desconocido'),'err');
      $('renderBtn').disabled=false;
      state.job=null;
    }
  }).catch(e=>{status('Error: '+e.message,'err');$('renderBtn').disabled=false;state.job=null});
}
</script>
</body>
</html>
"""
PAGE = PAGE


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send(200, PAGE.encode("utf-8"))
        elif self.path == "/healthz":
            body = json.dumps({
                "ok": True,
                "workers": WORKERS,
                "queue": _QUEUE.qsize(),
                "queue_capacity": MAX_PENDING,
            }).encode()
            self._send(200, body, "application/json")
        elif self.path.startswith("/out/"):
            name = os.path.basename(unquote(self.path))
            path = os.path.join(_OUT_DIR, name)
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self._send(200, f.read(), "audio/wav")
            else:
                self._send(404, b"not found", "text/plain")
        elif self.path.startswith("/api/status"):
            qs = parse_qs(self.path.split("?", 1)[-1])
            job_id = qs.get("job", [None])[0]
            with _JOBS_LOCK:
                job = dict(_JOBS[job_id]) if job_id in _JOBS else None
            if not job:
                self._send(404, json.dumps({"status": "unknown"}).encode(), "application/json")
                return
            job["queue_size"] = _QUEUE.qsize()
            self._send(200, json.dumps(job).encode(), "application/json",
                       {"Cache-Control": "no-store"})
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/render":
            self._send(404, b"not found", "text/plain")
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length <= 0 or length > MAX_UPLOAD:
                self._send(413, json.dumps({"ok": False, "error": "file too large"}).encode(),
                           "application/json")
                return
            raw = self.rfile.read(length)
            req = json.loads(raw)
            data = base64.b64decode(req["data"], validate=True)
            if len(data) > MAX_UPLOAD or data[:4] != b"MThd":
                self._send(400, json.dumps({"ok": False, "error": "not a MIDI file"}).encode(),
                           "application/json")
                return

            raw_name = os.path.splitext(os.path.basename(req.get("name", "input.mid")))[0]
            name = _safe_name(raw_name)
            token = uuid.uuid4().hex[:12]
            in_path = os.path.join(_OUT_DIR, f"{token}_in.mid")
            with open(in_path, "wb") as f:
                f.write(data)

            o = req.get("options", {})
            opts = RenderOptions(
                ai_strength=float(o.get("ai_strength", 0.6)),
                humanize=float(o.get("ai_strength", 0.6)) > 0,
                reverb=float(o.get("reverb", 0.3)),
                brightness=float(o.get("brightness", 0.0)),
                warmth=float(o.get("warmth", 0.0)),
                width=float(o.get("width", 0.0)),
                loudness_db=float(o.get("loudness_db", -15.0)),
                saturation=float(o.get("saturation", 0.5)),
                sample_rate=int(o.get("sample_rate", 44100)),
                seed=int(o.get("seed", 42)),
                use_model=bool(o.get("use_model", True)),
                max_duration=MAX_DURATION,
            )

            job_id = uuid.uuid4().hex
            out_name = f"{name}_{token}.wav"
            out_path = os.path.join(_OUT_DIR, out_name)
            with _JOBS_LOCK:
                _JOBS[job_id] = {"status": "queued", "progress": 0.0,
                                 "queued": time.time(), "updated": time.time()}
            try:
                _QUEUE.put_nowait((job_id, in_path, out_path, opts, out_name))
            except queue.Full:
                with _JOBS_LOCK:
                    _JOBS.pop(job_id, None)
                try:
                    os.remove(in_path)
                except OSError:
                    pass
                self._send(429, json.dumps({
                    "ok": False,
                    "error": "server busy, try again in a moment",
                }).encode(), "application/json")
                return

            self._send(200, json.dumps({"ok": True, "job_id": job_id,
                                        "queue": _QUEUE.qsize()}).encode(),
                       "application/json")
        except Exception as e:  # noqa: BLE001
            self._send(400, json.dumps({"ok": False, "error": str(e)}).encode(),
                       "application/json")


def main():
    _start_workers()
    _cleanup_out_dir()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    server.daemon_threads = True
    url = f"http://{HOST}:{PORT}"
    print(f"MIDI-AI Studio: {url}")
    print("Press Ctrl+C to stop.")
    if HOST in ("127.0.0.1", "localhost"):
        threading.Timer(0.7, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()