import unittest
from run_script.run_v4l2_distillation import build_config
from moge.model.v2 import MoGeModel
from moge.train.finetuning import configure_moge2_heads

class FreezeSweepTests(unittest.TestCase):
    def test_only_lora_and_scale_scope(self):
        for selected, expected in [([], {'backbone'}), (['scale_head'], {'backbone','scale_head'})]:
            config, _, _, _ = build_config('/tmp/carla', '/tmp/pseudo', lora_rank=4,
                multitask_labels=True, mask_weight=.1, normal_weight=.1,
                trainable_components=selected)
            self.assertEqual({g['name'] for g in config['optimizer']['params']}, expected)
            self.assertEqual(len(config['lr_scheduler']['params']['lr_lambda']),len(expected))
            self.assertEqual({k for k,v in config['trainable_heads'].items() if v},set(selected))
            self.assertEqual(config['loss']['PSEUDO']['mask']['teacher_mask']['weight'],.025)
            self.assertEqual(config['loss']['PSEUDO']['normal']['teacher_normal']['weight'],.025)
            model=MoGeModel(**config['model'])
            configure_moge2_heads(model,config['trainable_heads'],4)
            names=[n for n,p in model.named_parameters() if p.requires_grad]
            self.assertTrue(any('.lora_' in n for n in names))
            self.assertTrue(all('.lora_' in n or ('scale_head.' in n and selected) for n in names))
            self.assertEqual(any('scale_head.' in n for n in names),bool(selected))
        with self.assertRaises(ValueError):
            build_config('/tmp/carla','/tmp/pseudo',lora_rank=4,trainable_components=['unknown'])

if __name__=='__main__':unittest.main()
