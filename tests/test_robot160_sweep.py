import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from run_script.sweep_5000 import REPO, resolve_experiment


class RobotSweepTests(unittest.TestCase):
    def test_fixed_tokens_aug_both_datasets_and_lr(self):
        manifest=json.loads((REPO/'configs/experiments/gaemi_robot160_motion_sweep.json').read_text())
        base=json.loads((REPO/manifest['base_config']).read_text())
        self.assertEqual(base['model']['num_tokens_range'],[160,160])
        self.assertEqual(base['data']['image_sizes'],[[640,360]])
        self.assertEqual((round((160*640/360)**.5),round((160/(640/360))**.5)),(17,9))
        self.assertEqual([d['weight'] for d in base['data']['datasets']],[20817,9351])
        self.assertEqual(manifest['num_iterations'],3000)
        self.assertEqual(len(manifest['experiments']),4)
        self.assertEqual([e['flip_augmentation'] for e in manifest['experiments']],[False,True,True,True])
        for e in manifest['experiments']:
            cfg,b,a=resolve_experiment(base,e,200,32,1)
            self.assertEqual((b,a),(8,4))
            self.assertEqual(cfg['train_backbone_last_n_blocks'],4)
            self.assertEqual(e['learning_rates']['backbone'],1e-6)
            for ds in cfg['data']['datasets']:
                self.assertEqual(ds['image_augmentation'],e['image_augmentation'])
                self.assertEqual(ds['flip_augmentation'],e['flip_augmentation'])
                self.assertTrue(ds['index'].startswith('.review-versions/'))

    def test_relocating_parent_preserves_separate_roots(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'output'
            subprocess.run([sys.executable,str(REPO/'run_script/sweep_5000.py'),
                            '--manifest',str(REPO/'configs/experiments/gaemi_robot160_motion_sweep.json'),
                            '--root',str(root),'--data-root','/tmp/robot-data','--dry-run'],
                           check=True,capture_output=True)
            cfg=json.loads((root/'robot160_flip_motion_mild_3000/train_config.json').read_text())
            self.assertEqual([d['path'] for d in cfg['data']['datasets']],
                             ['/tmp/robot-data/20261017_train','/tmp/robot-data/near_obstacles_train'])
            val=json.loads((root/'validation_config.json').read_text())
            ds=next(iter(val['datasets'].values()))
            self.assertEqual(ds['path'],'/tmp/robot-data/20261017_train')
            self.assertTrue(ds['split'].endswith('reviewed_20261018_v1/.val.txt'))
            self.assertIsNone(val['depth_bins'])


if __name__ == '__main__':
    unittest.main()
