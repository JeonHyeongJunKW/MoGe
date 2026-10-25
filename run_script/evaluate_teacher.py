"""Evaluate a MoGe2 teacher without creating an optimizer or training loader."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained', required=True, help='Local checkpoint or Hugging Face repository')
    parser.add_argument('--validation-config', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--num-tokens', type=int, default=1200)
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    if args.num_tokens < 1 or args.batch_size < 1:
        parser.error('Tokens and batch size must be positive')
    import torch
    from huggingface_hub import hf_hub_download
    from moge.model.v2 import MoGeModel
    from moge.train.validation import ValidationRunner

    source = Path(args.pretrained)
    if not source.is_file():
        if source.is_absolute() or args.pretrained.endswith('.pt'):
            raise FileNotFoundError(source)
        source = Path(hf_hub_download(repo_id=args.pretrained, filename='model.pt'))
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    model_config = copy.deepcopy(checkpoint['model_config'])
    model_config['num_tokens_range'] = [args.num_tokens, args.num_tokens]
    model = MoGeModel(**model_config)
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    model.requires_grad_(False).to('cuda').eval()
    config = json.loads(args.validation_config.read_text())
    config.update(batch_size=args.batch_size, per_image_metrics=False, depth_bins=None)
    args.workspace.mkdir(parents=True, exist_ok=True)
    (args.workspace / 'teacher_config.json').write_text(json.dumps({
        'pretrained': args.pretrained, 'resolved_checkpoint': str(source),
        'model': model_config, 'validation': config, 'precision': 'float32',
    }, indent=2))
    runner = ValidationRunner(config, args.workspace)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        results = runner.evaluate(model, torch.device('cuda'), step=0)
    info = {'step': 0, 'monitor': runner.monitor, 'mode': runner.mode,
            'score': results['mean'][runner.monitor], 'signature': runner.signature,
            'metrics': results, 'pretrained': args.pretrained, 'num_tokens': args.num_tokens}
    directory = args.workspace / 'validation'
    directory.mkdir(exist_ok=True)
    for filename in ['step_00000000.json', 'best.json']:
        (directory / filename).write_text(json.dumps(info, indent=2, allow_nan=False))
    (args.workspace / 'evaluation_memory.json').write_text(json.dumps({
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
    }, indent=2))
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
