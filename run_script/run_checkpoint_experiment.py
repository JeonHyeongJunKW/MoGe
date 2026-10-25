"""Resolve checkpoint-owned architecture before launching a sweep training command."""
import argparse
import copy
import json
import subprocess
from pathlib import Path


def configure_model(config, checkpoint_model_config):
    result = copy.deepcopy(config)
    experiment = result['sweep_experiment']
    result['model'] = copy.deepcopy(checkpoint_model_config)
    result['model']['num_tokens_range'] = [experiment['num_tokens']] * 2
    if 'rgb_detail' in experiment:
        result['model']['rgb_detail'] = copy.deepcopy(experiment['rgb_detail'])
    for component in experiment['learning_rates']:
        if component != 'backbone' and result['model'].get(component) is None:
            raise ValueError(f'Checkpoint has no requested component: {component}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-source', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command or '--config' not in command or '--initial_checkpoint' not in command:
        parser.error('Expected training command with --config and --initial_checkpoint')
    import torch
    source = Path(args.checkpoint_source)
    if not source.is_file():
        if source.is_absolute() or args.checkpoint_source.endswith('.pt'):
            raise FileNotFoundError(source)
        from huggingface_hub import hf_hub_download
        source = Path(hf_hub_download(repo_id=args.checkpoint_source, filename='model.pt'))
    checkpoint = torch.load(source, map_location='cpu', weights_only=True)
    path = Path(command[command.index('--config') + 1])
    config = configure_model(json.loads(path.read_text()), checkpoint['model_config'])
    del checkpoint
    config['resolved_initial_checkpoint'] = str(source.resolve())
    path.write_text(json.dumps(config, indent=4) + '\n')
    command[command.index('--initial_checkpoint') + 1] = str(source.resolve())
    (path.parent / 'resolved_command.json').write_text(json.dumps(command, indent=2) + '\n')
    print(f"Checkpoint architecture: {config['model']['encoder']['backbone']}; tokens={config['model']['num_tokens_range']}", flush=True)
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
