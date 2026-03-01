#!/usr/bin/env python3
"""Simple image labeling server - no dependencies beyond stdlib."""

import json
import os
import shutil
import subprocess
import urllib.parse
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

BASE_DIR         = os.path.dirname(__file__)
TRAINING_DATA    = os.path.join(BASE_DIR, '..', 'training_data')
DESCRIPTIONS_FILE= os.path.join(BASE_DIR, 'descriptions.json')
INDEX_HTML       = os.path.join(BASE_DIR, 'index.html')
GENERATOR_HTML   = os.path.join(BASE_DIR, 'generator.html')
GENERATED_DIR    = os.path.join(BASE_DIR, 'generated')
GENERATED_META   = os.path.join(GENERATED_DIR, 'metadata.json')
GENERATE_SH      = os.path.join(BASE_DIR, '..', 'generateImage.sh')
OUTPUT_DIR       = os.path.join(BASE_DIR, '..', 'trainmodel', 'output')


def load_descriptions():
    if os.path.exists(DESCRIPTIONS_FILE):
        with open(DESCRIPTIONS_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_descriptions(data):
    with open(DESCRIPTIONS_FILE, 'w') as f:
        json.dump(data, f, indent=2)


def list_images():
    images = []
    for fname in sorted(os.listdir(TRAINING_DATA)):
        if fname.lower().endswith('.png'):
            images.append(fname)
    return images


def load_generated_meta():
    if os.path.exists(GENERATED_META):
        with open(GENERATED_META, 'r') as f:
            return json.load(f)
    return {}


def save_generated_meta(data):
    os.makedirs(GENERATED_DIR, exist_ok=True)
    with open(GENERATED_META, 'w') as f:
        json.dump(data, f, indent=2)


def list_generated_images():
    """Return list of generated image records sorted newest first."""
    meta = load_generated_meta()
    records = [{"filename": k, **v} for k, v in meta.items()]
    records.sort(key=lambda r: r.get("ts", ""), reverse=True)
    return records


def run_generate(user_prompt, num):
    """
    Run generateImage.sh, move output files to labeler/generated/,
    update metadata.json, and return list of new filenames.
    """
    os.makedirs(GENERATED_DIR, exist_ok=True)

    full_prompt = f"pixel image: (32x32) {user_prompt}"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    result = subprocess.run(
        ["/bin/bash", GENERATE_SH, full_prompt, str(num)],
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "generateImage.sh failed")

    # Collect output files (trainmodel.py names them generated.png or generated_N.png)
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
        raise RuntimeError("Generation finished but no output images were found")

    meta = load_generated_meta()
    new_filenames = []

    for src_path, idx in candidates:
        fname = f"gen_{ts}_{idx}.png"
        dst_path = os.path.join(GENERATED_DIR, fname)
        shutil.move(src_path, dst_path)
        meta[fname] = {"prompt": full_prompt, "user_prompt": user_prompt, "ts": ts}
        new_filenames.append(fname)

    save_generated_meta(meta)
    return new_filenames


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def send_html(self, path):
        with open(path, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == '/' or path == '/index.html':
            self.send_html(INDEX_HTML)

        elif path == '/generator':
            self.send_html(GENERATOR_HTML)

        elif path == '/api/images':
            self.send_json(list_images())

        elif path == '/api/descriptions':
            self.send_json(load_descriptions())

        elif path == '/api/generated-images':
            self.send_json(list_generated_images())

        elif path.startswith('/images/'):
            fname = urllib.parse.unquote(path[len('/images/'):])
            if '/' in fname or '..' in fname:
                self.send_response(400); self.end_headers(); return
            fpath = os.path.join(TRAINING_DATA, fname)
            self._serve_image(fpath)

        elif path.startswith('/generated/'):
            fname = urllib.parse.unquote(path[len('/generated/'):])
            if '/' in fname or '..' in fname:
                self.send_response(400); self.end_headers(); return
            fpath = os.path.join(GENERATED_DIR, fname)
            self._serve_image(fpath)

        else:
            self.send_response(404)
            self.end_headers()

    def _serve_image(self, fpath):
        if not os.path.exists(fpath):
            self.send_response(404); self.end_headers(); return
        with open(fpath, 'rb') as f:
            body = f.read()
        self.send_response(200)
        self.send_header('Content-Type', 'image/png')
        self.send_header('Content-Length', len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)

        if self.path == '/api/descriptions':
            try:
                data = json.loads(body)
                save_descriptions(data)
                self.send_json({'ok': True})
                print(f"  Saved {len(data)} descriptions.")
            except Exception as e:
                self.send_json({'error': str(e)}, status=400)

        elif self.path == '/api/generate':
            try:
                data = json.loads(body)
                user_prompt = data.get('prompt', '').strip()
                num = max(1, min(4, int(data.get('num', 1))))
                if not user_prompt:
                    self.send_json({'ok': False, 'error': 'Prompt is empty'}, status=400)
                    return
                print(f"  Generating: \"{user_prompt}\" ×{num}")
                new_files = run_generate(user_prompt, num)
                print(f"  Done: {new_files}")
                self.send_json({'ok': True, 'images': new_files})
            except subprocess.TimeoutExpired:
                self.send_json({'ok': False, 'error': 'Generation timed out (>10 min)'}, status=500)
            except Exception as e:
                print(f"  Generate error: {e}")
                self.send_json({'ok': False, 'error': str(e)}, status=500)

        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    port = 8787
    os.makedirs(GENERATED_DIR, exist_ok=True)
    server = HTTPServer(('', port), Handler)
    print(f"Server running at http://localhost:{port}")
    print(f"  Labeler:   http://localhost:{port}/")
    print(f"  Generator: http://localhost:{port}/generator")
    print(f"Loading images from: {os.path.abspath(TRAINING_DATA)}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
