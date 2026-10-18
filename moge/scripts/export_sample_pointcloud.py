"""Re-infer one reported validation sample with identical preprocessing for 3D inspection."""
import argparse
import copy
import json
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if not 0 <= args.index < len(report['samples']):
        parser.error('Invalid sample index')
    sample = report['samples'][args.index]
    metadata = report.get('metadata', {})
    if 'config' not in metadata:
        summary = args.report.parent / 'summary.json'
        if summary.is_file():
            metadata = json.loads(summary.read_text())
    config = copy.deepcopy(metadata['config'])
    dataset = config['datasets'][sample['dataset']]
    root = args.data_root.resolve()
    (root / sample['filename']).resolve().relative_to(root)
    if '\n' in sample['filename'] or '\r' in sample['filename']:
        parser.error('Invalid sample filename')
    if not args.checkpoint.is_file():
        parser.error('Checkpoint does not exist')
    import torch
    from moge.model.v2 import MoGeModel
    from moge.train.validation import ValidationRunner
    with tempfile.TemporaryDirectory(prefix='moge-pointcloud-') as temporary:
        split = Path(temporary) / 'sample.txt'
        split.write_text(sample['filename'] + '\n')
        dataset.update(path=str(root), split=str(split))
        dataset.pop('subset', None)
        config.update(datasets={sample['dataset']: dataset}, batch_size=1, max_samples=1,
                      metric_groups='depth_metric', depth_bins=None, per_image_metrics=False,
                      visualization={'num_samples': 1, 'max_points': 50000})
        runner = ValidationRunner(config, args.output)
        model = MoGeModel.from_pretrained(str(args.checkpoint)).to('cuda').eval()
        model.enable_mixed_precision()
        metrics = runner.evaluate(model, torch.device('cuda'), step=0)
        folder = args.output / 'validation' / 'step_00000000' / sample['dataset'] / '000000'
        info_path = folder / 'info.json'
        info = json.loads(info_path.read_text())
        info.update(checkpoint=str(args.checkpoint.resolve()), source_report=str(args.report.resolve()),
                    source_index=args.index, source_metrics=sample['metrics'],
                    regenerated_metrics=metrics['mean'], note='Re-inferred on demand; not stored predictions from original evaluation')
        info_path.write_text(json.dumps(info, indent=2, allow_nan=False))
        (args.output / 'complete.json').write_text(json.dumps({'sample': str(folder.relative_to(args.output))}))
        print(folder)


if __name__ == '__main__':
    main()
