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
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), partial(Handler, root=self.root))
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
