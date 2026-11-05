"""Export LoRA + decoder weights as an ordinary MoGe2 checkpoint for deployment."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output exists; use a new filename')
    import torch
    from moge.model.modules.lora import merged_checkpoint
    from moge.model.v2 import MoGeModel
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    # Strictly check architecture/adapter completeness before merging.
    model = MoGeModel(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model'], strict=True)
    del model
    merged = merged_checkpoint(checkpoint)
    model = MoGeModel(**merged['model_config'])
    model.load_state_dict(merged['model'], strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, args.output)
    print(f'Merged model-only checkpoint: {args.output}')


if __name__ == '__main__':
    main()
