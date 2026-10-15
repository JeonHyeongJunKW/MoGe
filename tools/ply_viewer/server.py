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
    def __init__(self, *args, root, data_roots=(), **kwargs):
        self.root = root
        self.data_roots = tuple(Path(p).resolve() for p in data_roots)
        super().__init__(*args, **kwargs)

    def do_GET(self):
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        try:
            if url.path == '/api/data-roots':
                self.send_bytes(json.dumps({'roots': [str(p) for p in self.data_roots]}).encode(), 'application/json')
                return
            if url.path == '/api/reports':
                reports = []
                if self.root.is_dir():
                    for candidate in sorted(self.root.rglob('per_image_step_*.json')):
                        try:
                            relative = candidate.relative_to(self.root).as_posix()
                            inside(self.root, relative)
                            reports.append(relative)
                        except ValueError:
                            continue
                self.send_bytes(json.dumps({'reports': reports}).encode(), 'application/json')
                return
            if url.path == '/api/dataset-image':
                root = Path(query.get('root', [''])[0]).resolve()
                if root not in self.data_roots:
                    raise ValueError('Dataset root not registered with --data-root')
                path = inside(root, query.get('filename', [''])[0])
                if path.name != 'image.jpg':
                    raise ValueError('Only dataset RGB images are served')
            elif url.path in ('/api/report', '/api/report-image'):
                report_path = inside(self.root, query.get('report', [''])[0])
                if not report_path.name.startswith('per_image_step_') or report_path.suffix != '.json':
                    raise ValueError('Invalid report')
                if url.path == '/api/report':
                    data = json.loads(report_path.read_text())
                    if data.get('schema_version') != 1 or not isinstance(data.get('samples'), list):
                        raise ValueError('Unsupported report')
                    self.send_bytes(json.dumps(data, allow_nan=False).encode(), 'application/json')
                    return
                report = json.loads(report_path.read_text())
                index = int(query.get('index', ['-1'])[0])
                if index < 0 or index >= len(report['samples']):
                    raise ValueError('Invalid sample')
                preview = report['samples'][index].get('image', '')
                path = inside(report_path.parent.resolve(), preview)
                inside(self.root, path)
                if path.suffix.lower() != '.jpg' or path.parent != report_path.parent.resolve() / 'sample_images':
                    raise ValueError('Invalid preview')
            elif url.path == '/api/samples':
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
            elif url.path == '/api/asset':
                name = query.get('name', [''])[0]
                if name not in ASSETS:
                    self.send_error(404)
                    return
                path = inside(self.root, query.get('sample', [''])[0]) / name
                path = inside(self.root, path)
            else:
                name = url.path.removeprefix('/') or 'index.html'
                if name not in {'index.html', 'app.js', 'ply.js', 'trackball.js', 'style.css', 'analysis.html', 'analysis.js', 'analysis.css', 'tags.js'}:
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
        except (ValueError, KeyError, TypeError, FileNotFoundError, IsADirectoryError, PermissionError):
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
    parser.add_argument('--data-root', type=Path, action='append', default=[], help='Dataset RGB root; repeat for multiple datasets')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root = args.root.resolve()
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, root=root, data_roots=args.data_root))
    print(f'PLY viewer: http://{args.host}:{server.server_port}\nResults root: {root}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
