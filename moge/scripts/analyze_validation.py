"""Evaluate an existing MoGe-2 checkpoint and export the interactive viewer's reports."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New empty directory under the viewer root')
    parser.add_argument('--config', type=Path, default=Path('configs/validation/gaemi_moge2_vits_sweep_5000.json'))
    parser.add_argument('--data-root', '--dataset-path', type=Path)
    parser.add_argument('--split', help='Split file relative to dataset root, e.g. .val.txt or .train.txt')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-samples', type=int, help='Optional first N samples per dataset for a quick check')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        parser.error('Checkpoint does not exist')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('Output directory must be empty; choose a new directory')
    config = json.loads(args.config.read_text())
    config.update(per_image_metrics=True, batch_size=args.batch_size, visualization={'num_samples': 0})
    if args.max_samples is not None:
        config['max_samples'] = args.max_samples
    if args.data_root:
        for dataset in config['datasets'].values():
            dataset['path'] = str(args.data_root.resolve())
    if args.split:
        for dataset in config['datasets'].values():
            dataset['split'] = args.split
    import torch
    from moge.model.v2 import MoGeModel
    from moge.train.validation import ValidationRunner
    from moge.train.per_image_validation import write_image_report
    from moge.train.depth_bins import write_depth_bin_report
    runner = ValidationRunner(config, args.output)
    model = MoGeModel.from_pretrained(str(args.checkpoint)).to(args.device)
    if torch.device(args.device).type == 'cuda':
        model.enable_mixed_precision()
    results = runner.evaluate(model, torch.device(args.device), step=0)
    directory = args.output / 'validation'
    directory.mkdir(parents=True, exist_ok=True)
    report = write_image_report(directory, 0, runner.signature, runner.image_records, metadata={
        'checkpoint': str(args.checkpoint.resolve()), 'config': config, 'metrics': results,
    })
    (directory / 'summary.json').write_text(json.dumps({
        'checkpoint': str(args.checkpoint.resolve()), 'config': config, 'metrics': results,
    }, indent=2, allow_nan=False))
    if runner.depth_bin_results is not None:
        write_depth_bin_report(directory / 'depth_bins_step_00000000', {'step': 0, **runner.depth_bin_results})
    print(f'Per-image report: {report}')
    print(json.dumps(results['mean'], indent=2))


if __name__ == '__main__':
    main()
