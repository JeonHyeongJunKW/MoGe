"""Read-only, dependency-free server for validation PLY comparisons."""
import argparse
import json
import mimetypes
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

STATIC = Path(__file__).resolve().parent
DEFAULT_ROOT = STATIC.parents[1] / 'workspace'
ASSETS = {'gt.ply', 'pred.ply', 'image.png', 'info.json'}


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, **kwargs):
        self.root = root
        super().__init__(*args, **kwargs)

    def do_GET(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == '/api/samples':
                samples = []
                if self.root.is_dir():
                    for gt in sorted(self.root.rglob('gt.ply')):
                        try:
                            folder = gt.parent.relative_to(self.root).as_posix()
                            if all(inside(self.root, f'{folder}/{name}').is_file()
                                   for name in ('gt.ply', 'pred.ply')):
                                samples.append(folder)
                        except (ValueError, OSError):
                            continue
                self.send_bytes(json.dumps({'samples': samples}).encode(), 'application/json')
                return
            if url.path == '/api/asset':
                name = query.get('name', [''])[0]
                if name not in ASSETS:
                    self.send_error(404)
                    return
                path = inside(self.root, query.get('sample', [''])[0]) / name
                path = inside(self.root, path)
            else:
                name = url.path.removeprefix('/') or 'index.html'
                if name not in {'index.html', 'app.js', 'ply.js', 'trackball.js', 'style.css'}:
                    self.send_error(404)
                    return
                path = STATIC / name
            with path.open('rb') as source:
                self.send_response(200)
                self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
                self.send_header('Content-Length', str(path.stat().st_size))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                while chunk := source.read(1024 * 1024):
                    self.wfile.write(chunk)
        except (ValueError, FileNotFoundError, IsADirectoryError, PermissionError):
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_bytes(self, data, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT, help='Directory to scan recursively for gt.ply + pred.ply')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root = args.root.resolve()
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, root=root))
    print(f'PLY viewer: http://{args.host}:{server.server_port}\nResults root: {root}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
