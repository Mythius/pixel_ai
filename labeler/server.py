#!/usr/bin/env python3
"""Pixel art labeler + generator server."""

import base64
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA    = os.path.join(BASE_DIR, '..', 'training_data')
DESCRIPTIONS_FILE= os.path.join(BASE_DIR, 'descriptions.json')
INDEX_HTML       = os.path.join(BASE_DIR, 'index.html')
GENERATOR_HTML   = os.path.join(BASE_DIR, 'generator.html')
GENERATED_DIR    = os.path.join(BASE_DIR, 'generated')
GENERATED_META   = os.path.join(GENERATED_DIR, 'metadata.json')
GENERATE_SH      = os.path.join(BASE_DIR, '..', 'generateImage.sh')
OUTPUT_DIR       = os.path.join(BASE_DIR, '..', 'trainmodel', 'output')
PIXEL_REDUCE     = os.path.join(BASE_DIR, '..', 'pixel_reducer', 'pixelreduce.py')
REDUCER_HTML     = os.path.join(BASE_DIR, 'reducer.html')
TRAIN_HTML       = os.path.join(BASE_DIR, 'train.html')
REDUCER_TMP      = os.path.join(BASE_DIR, '..', 'tmp_reduce')

# Use the venv Python (has cv2/numpy/PIL) if available, else fall back to this process
_venv_python_local = os.path.join(BASE_DIR, 'venv', 'bin', 'python')
_venv_python_root  = os.path.join(BASE_DIR, '..', 'venv', 'bin', 'python')
PYTHON = (
    _venv_python_local if os.path.isfile(_venv_python_local) else
    _venv_python_root  if os.path.isfile(_venv_python_root)  else
    sys.executable
)

# ── Job tracking (thread-safe) ─────────────────────────────────────────────
jobs: dict = {}      # job_id → {status, images, error, prompt, started_at}
jobs_lock = threading.Lock()

# ── Logging ───────────────────────────────────────────────────────────────
def log(msg: str, level: str = "INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    label = level.ljust(5)
    print(f"[{ts}] {label}  {msg}", flush=True)

def log_req(method: str, path: str, status: int, extra: str = ""):
    color = "\033[32m" if status < 300 else "\033[33m" if status < 500 else "\033[31m"
    reset = "\033[0m"
    suffix = f"  {extra}" if extra else ""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {color}{method.ljust(4)} {str(status)}  {path}{suffix}{reset}", flush=True)

# ── Data helpers ──────────────────────────────────────────────────────────
def load_descriptions():
    if os.path.exists(DESCRIPTIONS_FILE):
        with open(DESCRIPTIONS_FILE) as f:
            return json.load(f)
    return {}

def save_descriptions(data):
    with open(DESCRIPTIONS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def list_images():
    return [f for f in sorted(os.listdir(TRAINING_DATA)) if f.lower().endswith('.png')]

def load_generated_meta():
    if os.path.exists(GENERATED_META):
        with open(GENERATED_META) as f:
            return json.load(f)
    return {}

def save_generated_meta(data):
    os.makedirs(GENERATED_DIR, exist_ok=True)
    with open(GENERATED_META, 'w') as f:
        json.dump(data, f, indent=2)

def list_generated_images():
    meta = load_generated_meta()
    records = [{"filename": k, **v} for k, v in meta.items()]
    records.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return records

# ── Generation (runs in background thread) ────────────────────────────────
def run_generate_thread(job_id: str, user_prompt: str, num: int):
    """Background thread: run generateImage.sh, move files, update job state."""
    full_prompt = f"pixel image: (32x32) {user_prompt}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    started = time.time()

    log(f'Job {job_id[:8]}  START  "{user_prompt}" ×{num}')

    with jobs_lock:
        jobs[job_id]["status"] = "running"

    try:
        os.makedirs(GENERATED_DIR, exist_ok=True)
        result = subprocess.run(
            ["/bin/bash", GENERATE_SH, full_prompt, str(num)],
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            stderr = result.stderr.strip() or "generateImage.sh returned non-zero exit code"
            raise RuntimeError(stderr)

        # Collect output files
        candidates = []
        if num == 1:
            p = os.path.join(OUTPUT_DIR, "generated.png")
            if os.path.exists(p):
                candidates.append((p, 1))
        for i in range(1, num + 1):
            p = os.path.join(OUTPUT_DIR, f"generated_{i}.png")
            if os.path.exists(p):
                candidates.append((p, i))

        if not candidates:
            raise RuntimeError("Generation finished but no output images were found in trainmodel/output/")

        # Run pixel reducer on each image: auto-detect pixel size, limit to 12 colors
        for src_path, idx in candidates:
            reduce = subprocess.run(
                [PYTHON, PIXEL_REDUCE, src_path, src_path, '--colors', '12'],
                capture_output=True, text=True, timeout=60,
            )
            if reduce.returncode == 0:
                log(f'Job {job_id[:8]}  REDUCE ok  {os.path.basename(src_path)}')
            else:
                log(f'Job {job_id[:8]}  REDUCE failed (keeping original): {reduce.stderr.strip()[:120]}', level="WARN")

        meta = load_generated_meta()
        new_filenames = []
        for src_path, idx in candidates:
            fname = f"gen_{ts}_{idx}.png"
            shutil.move(src_path, os.path.join(GENERATED_DIR, fname))
            meta[fname] = {"prompt": full_prompt, "user_prompt": user_prompt, "ts": ts}
            new_filenames.append(fname)

        save_generated_meta(meta)

        elapsed = time.time() - started
        log(f'Job {job_id[:8]}  DONE   {new_filenames}  ({elapsed:.1f}s)')

        with jobs_lock:
            jobs[job_id]["status"] = "done"
            jobs[job_id]["images"] = new_filenames

    except subprocess.TimeoutExpired:
        log(f'Job {job_id[:8]}  TIMEOUT  after 600s', level="ERROR")
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Generation timed out (>10 min)"

    except Exception as e:
        log(f'Job {job_id[:8]}  ERROR  {e}', level="ERROR")
        with jobs_lock:
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = str(e)


# ── Request handler ───────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        pass  # We handle logging ourselves

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(body))
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def send_html(self, path, status=200):
        try:
            with open(path, 'rb') as f:
                body = f.read()
            self.send_response(status)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def send_image(self, fpath):
        if not os.path.exists(fpath):
            self.send_response(404)
            self.end_headers()
            return
        try:
            with open(fpath, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', len(body))
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path in ('/', '/index.html'):
            self.send_html(INDEX_HTML)
            log_req("GET", path, 200)

        elif path == '/generator':
            self.send_html(GENERATOR_HTML)
            log_req("GET", path, 200)

        elif path == '/reducer':
            self.send_html(REDUCER_HTML)
            log_req("GET", path, 200)

        elif path == '/train':
            self.send_html(TRAIN_HTML)
            log_req("GET", path, 200)

        elif path == '/api/images':
            data = list_images()
            self.send_json(data)
            log_req("GET", path, 200, f"{len(data)} images")

        elif path == '/api/descriptions':
            self.send_json(load_descriptions())
            log_req("GET", path, 200)

        elif path == '/api/generated-images':
            data = list_generated_images()
            self.send_json(data)
            log_req("GET", path, 200, f"{len(data)} images")

        elif path.startswith('/api/generate/status/'):
            job_id = path[len('/api/generate/status/'):]
            with jobs_lock:
                job = jobs.get(job_id)
            if not job:
                self.send_json({"error": "Unknown job"}, status=404)
                log_req("GET", path, 404)
            else:
                self.send_json(job)
                log_req("GET", path, 200, job["status"])

        elif path.startswith('/images/'):
            fname = urllib.parse.unquote(path[len('/images/'):])
            if '/' in fname or '..' in fname:
                self.send_response(400); self.end_headers(); return
            self.send_image(os.path.join(TRAINING_DATA, fname))

        elif path.startswith('/generated/'):
            fname = urllib.parse.unquote(path[len('/generated/'):])
            if '/' in fname or '..' in fname:
                self.send_response(400); self.end_headers(); return
            self.send_image(os.path.join(GENERATED_DIR, fname))

        else:
            self.send_response(404)
            self.end_headers()
            log_req("GET", path, 404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        path = urllib.parse.urlparse(self.path).path

        if path == '/api/descriptions':
            try:
                data = json.loads(body)
                save_descriptions(data)
                self.send_json({'ok': True})
                log_req("POST", path, 200, f"{len(data)} descriptions saved")
            except Exception as e:
                self.send_json({'error': str(e)}, status=400)
                log_req("POST", path, 400, str(e))

        elif path == '/api/generate':
            try:
                data = json.loads(body)
                user_prompt = data.get('prompt', '').strip()
                num = max(1, min(4, int(data.get('num', 1))))

                if not user_prompt:
                    self.send_json({'ok': False, 'error': 'Prompt is empty'}, status=400)
                    log_req("POST", path, 400, "empty prompt")
                    return

                # Create job and start background thread immediately
                job_id = uuid.uuid4().hex
                with jobs_lock:
                    jobs[job_id] = {
                        "status": "pending",
                        "prompt": user_prompt,
                        "num": num,
                        "images": [],
                        "error": None,
                        "started_at": datetime.now().isoformat(),
                    }

                t = threading.Thread(
                    target=run_generate_thread,
                    args=(job_id, user_prompt, num),
                    daemon=True,
                )
                t.start()

                self.send_json({'ok': True, 'job_id': job_id})
                log_req("POST", path, 200, f"job {job_id[:8]} started  \"{user_prompt}\" ×{num}")

            except Exception as e:
                self.send_json({'ok': False, 'error': str(e)}, status=500)
                log_req("POST", path, 500, str(e))

        elif path == '/api/reduce':
            try:
                data = json.loads(body)
                img_b64 = data.get('image', '')
                width   = data.get('width')   # int or None (auto-detect)
                colors  = data.get('colors')  # int or None (no quantization)

                # Strip data URI prefix if present
                if ',' in img_b64:
                    img_b64 = img_b64.split(',', 1)[1]
                img_bytes = base64.b64decode(img_b64)

                os.makedirs(REDUCER_TMP, exist_ok=True)
                tmp_id   = uuid.uuid4().hex
                in_path  = os.path.join(REDUCER_TMP, f'{tmp_id}_in.png')
                out_path = os.path.join(REDUCER_TMP, f'{tmp_id}_out.png')

                with open(in_path, 'wb') as f:
                    f.write(img_bytes)

                try:
                    cmd = [PYTHON, PIXEL_REDUCE, in_path, out_path]
                    if width  is not None: cmd += ['--width',  str(int(width))]
                    if colors is not None: cmd += ['--colors', str(int(colors))]

                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                    log_req("POST", path, 200 if r.returncode == 0 else 500,
                            f"w={width} c={colors}")

                    if r.returncode != 0:
                        self.send_json({'ok': False, 'error': r.stderr.strip() or 'pixelreduce failed'}, status=500)
                        return

                    with open(out_path, 'rb') as f:
                        out_bytes = f.read()

                    out_b64 = base64.b64encode(out_bytes).decode()
                    self.send_json({'ok': True, 'image': out_b64})

                finally:
                    for p in (in_path, out_path):
                        if os.path.exists(p):
                            os.remove(p)

            except Exception as e:
                self.send_json({'ok': False, 'error': str(e)}, status=500)
                log_req("POST", path, 500, str(e))

        elif path == '/api/save-to-training':
            try:
                data = json.loads(body)
                img_b64 = data.get('image', '')
                if ',' in img_b64:
                    img_b64 = img_b64.split(',', 1)[1]
                img_bytes = base64.b64decode(img_b64)

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"reduced_{ts}_{uuid.uuid4().hex[:6]}.png"
                dest = os.path.join(TRAINING_DATA, fname)
                with open(dest, 'wb') as f:
                    f.write(img_bytes)

                self.send_json({'ok': True, 'filename': fname})
                log_req("POST", path, 200, fname)
            except Exception as e:
                self.send_json({'ok': False, 'error': str(e)}, status=500)
                log_req("POST", path, 500, str(e))

        else:
            self.send_response(404)
            self.end_headers()
            log_req("POST", path, 404)


# ── Server ────────────────────────────────────────────────────────────────
class PixelServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        """Suppress BrokenPipeError — browser closed connection early, harmless."""
        exc = sys.exc_info()[1]
        if isinstance(exc, BrokenPipeError):
            return
        super().handle_error(request, client_address)


if __name__ == '__main__':
    port = 8787
    os.makedirs(GENERATED_DIR, exist_ok=True)
    os.makedirs(REDUCER_TMP, exist_ok=True)

    server = PixelServer(('', port), Handler)

    log(f"Server started on http://localhost:{port}")
    log(f"  Labeler:   http://localhost:{port}/")
    log(f"  Generator: http://localhost:{port}/generator")
    log(f"  Images:    {os.path.abspath(TRAINING_DATA)}")
    log(f"  Generated: {os.path.abspath(GENERATED_DIR)}")
    log(f"  Python:    {PYTHON}")
    log("Press Ctrl+C to stop")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("Stopped.")
