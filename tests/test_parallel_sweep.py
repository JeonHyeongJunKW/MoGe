import json
from pathlib import Path
import sys
import tempfile
import unittest
from run_script.parallel_sweep import run_parallel


class ParallelSweepTests(unittest.TestCase):
    def test_concurrent_visibility_and_failure_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = root / 'worker.py'
            worker.write_text('''import os,sys,time,json
from pathlib import Path
root=Path(sys.argv[1]); name=sys.argv[2]
(root/(name+'.ready')).write_text(os.environ['CUDA_VISIBLE_DEVICES'])
deadline=time.monotonic()+5
while len(list(root.glob('*.ready'))) < 2:
    if time.monotonic()>deadline: raise RuntimeError('workers were not concurrent')
    time.sleep(.02)
print(name, flush=True)
raise SystemExit(int(sys.argv[3]))
''')
            jobs = [dict(name=n, rank=8, device=d, port=29600+i,
                         command=[sys.executable, str(worker), str(root), n, str(code)])
                    for i,(n,d,code) in enumerate([('r8','MIG-first',3),('direct','MIG-second',0)])]
            runs = run_parallel(jobs, root, root, 'ok', lambda *args: None)
            self.assertEqual({n:s for n,_,s in runs}, {'r8':'failed(3)', 'direct':'ok'})
            for job in jobs:
                self.assertEqual((root/(job['name']+'.ready')).read_text(), job['device'])
                self.assertIn(job['name'], (root/(job['name']+'.launch.log')).read_text())
            self.assertEqual(json.loads((root/'parallel_jobs.json').read_text()),jobs)

if __name__ == '__main__':
    unittest.main()
