"""HTTP contract and read-only file boundary for the standalone viewer."""
import json
import tempfile
import threading
import unittest
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.request import Request
from unittest.mock import patch
from types import SimpleNamespace

from tools.ply_viewer.server import Handler


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / 'results'
        self.sample = self.root / 'validation' / 'step_00001499' / 'sample'
        self.sample.mkdir(parents=True)
        for name in ('gt.ply', 'pred.ply'):
            (self.sample / name).write_bytes(b'ply\nfixture')
        (self.base / 'secret').write_text('private')
        (self.sample / 'info.json').symlink_to(self.base / 'secret')
        self.data_root = self.base / 'dataset'
        (self.data_root / 'sample').mkdir(parents=True)
        (self.data_root / 'sample' / 'image.jpg').write_bytes(b'original RGB')
        self.checkpoint = self.base / 'model.pt'
        self.checkpoint.write_bytes(b'fixture')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, root=self.root, data_roots=[self.data_root], checkpoint=self.checkpoint))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f'http://127.0.0.1:{self.server.server_port}'

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def get(self, path):
        with urlopen(self.url + path) as response:
            return response.read()

    def test_discovery_and_asset(self):
        samples = json.loads(self.get('/api/samples'))['samples']
        self.assertEqual(samples, ['validation/step_00001499/sample'])
        asset = '/api/asset?' + urlencode({'sample': samples[0], 'name': 'gt.ply'})
        self.assertEqual(self.get(asset), b'ply\nfixture')
        self.assertIn(b'GT & Prediction', self.get('/'))
        self.assertIn(b'export function trackball', self.get('/trackball.js'))

    def test_dataset_images_require_registered_root_and_stay_inside_it(self):
        self.assertEqual(json.loads(self.get('/api/data-roots'))['roots'], [str(self.data_root)])
        route = '/api/dataset-image?'
        self.assertEqual(self.get(route + urlencode({'root': str(self.data_root), 'filename': 'sample/image.jpg'})), b'original RGB')
        (self.data_root / 'escape').symlink_to(self.base, target_is_directory=True)
        (self.base / 'image.jpg').write_bytes(b'outside')
        for root, filename in [(self.base, 'dataset/sample/image.jpg'),
                               (self.data_root, '../secret'),
                               (self.data_root, 'sample/meta.json'),
                               (self.data_root, 'escape/image.jpg')]:
            with self.assertRaises(HTTPError):
                self.get(route + urlencode({'root': str(root), 'filename': filename}))

    def test_pointcloud_cache_and_origin_boundary(self):
        for name in ('depth.png', 'meta.json'):
            (self.data_root / 'sample' / name).write_bytes(b'fixture')
        report = self.root / 'per_image_step_00000000.json'
        report.write_text(json.dumps({'metadata': {'config': {}, 'checkpoint': str(self.checkpoint)},
                                     'samples': [{'dataset': 'val', 'filename': 'sample', 'dataset_root': str(self.data_root)}]}))
        payload = json.dumps({'report': report.name, 'index': 0}).encode()

        def infer(cmd, **kwargs):
            cache = Path(cmd[cmd.index('--output') + 1])
            (cache / 'sample').mkdir()
            (cache / 'complete.json').write_text(json.dumps({'sample': 'sample'}))
            return SimpleNamespace(returncode=0)

        with patch('tools.ply_viewer.server.subprocess.run', side_effect=infer) as run:
            def post():
                with urlopen(Request(self.url + '/api/pointcloud', data=payload)) as response:
                    return json.load(response)
            self.assertEqual(post(), post())
            self.assertEqual(run.call_count, 1)
        with self.assertRaises(HTTPError) as error:
            urlopen(Request(self.url + '/api/pointcloud', data=payload, headers={'Origin': 'http://foreign.example'}))
        self.assertEqual(error.exception.code, 403)

    def test_image_reports_and_preview_boundary(self):
        directory = self.root / 'validation'
        images = directory / 'sample_images'
        images.mkdir()
        (images / 'rgb.jpg').write_bytes(b'jpeg fixture')
        report = directory / 'per_image_step_00000001.json'
        payload = {'schema_version': 1, 'samples': [{'image': 'sample_images/rgb.jpg'}]}
        report.write_text(json.dumps(payload))
        relative = report.relative_to(self.root).as_posix()
        self.assertEqual(json.loads(self.get('/api/reports'))['reports'], [relative])
        self.assertEqual(json.loads(self.get('/api/report?' + urlencode({'report': relative}))), payload)
        route = '/api/report-image?' + urlencode({'report': relative, 'index': 0})
        self.assertEqual(self.get(route), b'jpeg fixture')
        (images / 'rgb.jpg').unlink()
        (images / 'rgb.jpg').symlink_to(self.base / 'secret')
        with self.assertRaises(HTTPError):
            self.get(route)
        with self.assertRaises(HTTPError):
            self.get('/api/report?' + urlencode({'report': '../per_image_step_1.json'}))
        self.assertIn(b'analysis.js', self.get('/analysis.html'))

    def test_no_traversal_arbitrary_files_or_symlink_escape(self):
        for params in [
            {'sample': '..', 'name': 'gt.ply'},
            {'sample': str(self.base), 'name': 'gt.ply'},
            {'sample': 'validation/step_00001499/sample', 'name': 'info.json'},
            {'sample': '.', 'name': '../secret'},
        ]:
            with self.subTest(params=params), self.assertRaises(HTTPError) as error:
                self.get('/api/asset?' + urlencode(params))
            self.assertEqual(error.exception.code, 404)
        with self.assertRaises(HTTPError):
            self.get('/server.py')


if __name__ == '__main__':
    unittest.main()
