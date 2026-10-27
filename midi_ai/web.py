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
from pathlib import Path
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


PAGE = (Path(__file__).with_name("ui.html")).read_text(encoding="utf-8")


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
            self._send(200, PAGE.encode("utf-8"), extra={"Cache-Control": "no-store"})
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
                    self._send(200, f.read(), "audio/wav",
                               {"Cache-Control": "public, max-age=3600"})
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