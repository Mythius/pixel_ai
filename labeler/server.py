#!/usr/bin/env python3
"""Simple image labeling server - no dependencies beyond stdlib."""

import json
import os
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

TRAINING_DATA = os.path.join(os.path.dirname(__file__), '..', 'training_data')
DESCRIPTIONS_FILE = os.path.join(os.path.dirname(__file__), 'descriptions.json')
INDEX_HTML = os.path.join(os.path.dirname(__file__), 'index.html')


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

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path

        if path == '/' or path == '/index.html':
            with open(INDEX_HTML, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

        elif path == '/api/images':
            self.send_json(list_images())

        elif path == '/api/descriptions':
            self.send_json(load_descriptions())

        elif path.startswith('/images/'):
            fname = urllib.parse.unquote(path[len('/images/'):])
            # Sanitize: no path traversal
            if '/' in fname or '..' in fname:
                self.send_response(400)
                self.end_headers()
                return
            fpath = os.path.join(TRAINING_DATA, fname)
            if not os.path.exists(fpath):
                self.send_response(404)
                self.end_headers()
                return
            with open(fpath, 'rb') as f:
                body = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'image/png')
            self.send_header('Content-Length', len(body))
            self.end_headers()
            self.wfile.write(body)

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == '/api/descriptions':
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
                save_descriptions(data)
                self.send_json({'ok': True})
                print(f"  Saved {len(data)} descriptions.")
            except Exception as e:
                self.send_json({'error': str(e)}, status=400)
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    port = 8787
    server = HTTPServer(('', port), Handler)
    print(f"Image Labeler running at http://localhost:{port}")
    print(f"Loading images from: {os.path.abspath(TRAINING_DATA)}")
    print(f"Saving descriptions to: {os.path.abspath(DESCRIPTIONS_FILE)}")
    print("Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
