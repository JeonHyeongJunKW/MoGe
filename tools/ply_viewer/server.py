"""Read-only, dependency-free server for validation PLY comparisons."""
import argparse
import json
import mimetypes
import hashlib
import os
import subprocess
import sys
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

STATIC = Path(__file__).resolve().parent
DEFAULT_ROOT = STATIC.parents[1] / 'workspace'
ASSETS = {'gt.ply', 'pred.ply', 'image.png', 'info.json'}
INFERENCE_LOCK = threading.Lock()


def inside(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root)
    return path


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, root, data_roots=(), checkpoint=None, inference_python=None, **kwargs):
        self.root = root
        self.data_roots = tuple(Path(p).resolve() for p in data_roots)
        self.checkpoint = Path(checkpoint).resolve() if checkpoint else None
        self.inference_python = str(inference_python or sys.executable)
        super().__init__(*args, **kwargs)

    def do_POST(self):
        if self.path != '/api/pointcloud':
            self.send_error(404)
            return
        # Only same-origin UI requests can start local inference.
        if self.headers.get('Origin') not in (None, 'http://' + self.headers.get('Host', '')):
            self.send_error(403)
            return
        acquired = False
        try:
            if not self.checkpoint:
                raise ValueError('서버 실행 시 --checkpoint와 --inference-python을 지정하세요.')
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 < size <= 16384:
                raise ValueError('Invalid request size')
            request = json.loads(self.rfile.read(size))
            report_path = inside(self.root, request['report'])
            if not report_path.name.startswith('per_image_step_') or report_path.suffix != '.json':
                raise ValueError('서버에 저장된 이미지별 보고서를 선택하세요.')
            report = json.loads(report_path.read_text())
            index = request['index']
            if type(index) is not int or not 0 <= index < len(report['samples']):
                raise ValueError('Invalid sample index')
            sample = report['samples'][index]
            root = Path(request.get('data_root') or sample.get('dataset_root', '')).resolve()
            if root not in self.data_roots:
                raise ValueError('--data-root로 등록한 데이터셋 경로를 선택하세요.')
            folder = inside(root, sample['filename'])
            metadata = report.get('metadata', {})
            if 'config' not in metadata and (report_path.parent / 'summary.json').is_file():
                metadata = json.loads((report_path.parent / 'summary.json').read_text())
            if 'config' not in metadata:
                raise ValueError('평가 config가 없습니다. analyze_validation으로 만든 보고서를 사용하세요.')
            expected = metadata.get('checkpoint')
            if expected and Path(expected).is_file() and Path(expected).resolve() != self.checkpoint:
                raise ValueError('보고서와 서버의 checkpoint가 다릅니다. 해당 모델로 서버를 실행하세요.')
            def stamp(path):
                stat = path.stat()
                return [str(path), stat.st_size, stat.st_mtime_ns]
            identity = [stamp(report_path), stamp(self.checkpoint), index, str(root),
                        *[stamp(inside(root, folder / name)) for name in ('image.jpg', 'depth.png', 'meta.json')]]
            digest = hashlib.sha256(json.dumps(identity).encode()).hexdigest()
            cache = inside(self.root, 'pointcloud_cache/' + digest)
            marker = cache / 'complete.json'
            if not marker.is_file():
                acquired = INFERENCE_LOCK.acquire(blocking=False)
                if not acquired:
                    raise ValueError('다른 이미지의 3D 생성 중입니다. 잠시 후 다시 눌러주세요.')
                cache.mkdir(parents=True, exist_ok=True)
                cmd = [self.inference_python, '-m', 'moge.scripts.export_sample_pointcloud',
                       '--report', str(report_path), '--index', str(index), '--checkpoint', str(self.checkpoint),
                       '--data-root', str(root), '--output', str(cache)]
                env = {**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                       'MOGE_NUM_LOAD_WORKERS': '2', 'MOGE_NUM_PROCESS_WORKERS': '2'}
                with (cache / 'inference.log').open('w') as log:
                    result = subprocess.run(cmd, cwd=STATIC.parents[1], env=env, stdout=log,
                                            stderr=subprocess.STDOUT, timeout=120)
                if result.returncode or not marker.is_file():
                    raise ValueError(f'3D 생성 실패. 로그: {cache / "inference.log"}')
            destination = inside(self.root, cache / json.loads(marker.read_text())['sample'])
            self.send_bytes(json.dumps({'sample': destination.relative_to(self.root).as_posix()}).encode(), 'application/json')
        except (ValueError, KeyError, IndexError, TypeError, OSError, subprocess.TimeoutExpired) as error:
            data = json.dumps({'error': str(error)}, ensure_ascii=False).encode()
            self.send_response(400)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            if acquired:
                INFERENCE_LOCK.release()

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
    parser.add_argument('--checkpoint', type=Path, help='Model for on-demand point cloud generation')
    parser.add_argument('--inference-python', type=Path, help='Python interpreter with MoGe evaluation dependencies')
    parser.add_argument('--data-root', type=Path, action='append', default=[], help='Dataset RGB root; repeat for multiple datasets')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    root = args.root.resolve()
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, root=root, data_roots=args.data_root,
                                                             checkpoint=args.checkpoint, inference_python=args.inference_python))
    print(f'PLY viewer: http://{args.host}:{server.server_port}\nResults root: {root}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
