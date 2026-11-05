"""Explicit, checkpointable LoRA for trailing DINOv2 attention projections.

Base weight/bias names are unchanged, so a plain checkpoint initializes a LoRA
model. No implicit train/eval merging: EMA and saved weights remain unambiguous.
"""
import math
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Linear):
    def __init__(self, base, rank, alpha):
        super().__init__(base.in_features, base.out_features, bias=base.bias is not None,
                         device=base.weight.device, dtype=base.weight.dtype)
        self.weight = base.weight
        self.bias = base.bias
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + self.scaling * F.linear(F.linear(x, self.lora_A), self.lora_B)


def validate_lora(config):
    if set(config) != {'rank', 'alpha', 'last_n_blocks', 'targets'}:
        raise ValueError('LoRA requires rank, alpha, last_n_blocks, targets')
    if type(config['rank']) is not int or config['rank'] < 1:
        raise ValueError('LoRA rank must be a positive integer')
    if not math.isfinite(config['alpha']) or config['alpha'] <= 0:
        raise ValueError('LoRA alpha must be positive and finite')
    if type(config['last_n_blocks']) is not int or config['last_n_blocks'] < 1:
        raise ValueError('LoRA last_n_blocks must be positive')
    if (not config['targets'] or len(set(config['targets'])) != len(config['targets']) or
            set(config['targets']) - {'qkv', 'proj'}):
        raise ValueError('LoRA targets must be distinct qkv/proj names')


def inject_lora(backbone, config):
    validate_lora(config)
    if getattr(backbone, 'chunked_blocks', False):
        raise ValueError('LoRA currently supports unchunked DINOv2 blocks only')
    count = config['last_n_blocks']
    if count > len(backbone.blocks):
        raise ValueError('More LoRA blocks requested than exist')
    # Keep the surrounding RNG stream unchanged for controlled comparisons.
    with torch.random.fork_rng(devices=[]):
        for block in list(backbone.blocks)[-count:]:
            for name in config['targets']:
                base = getattr(block.attn, name)
                if type(base) is not nn.Linear:
                    raise ValueError(f'Expected ordinary Linear at attention.{name}')
                setattr(block.attn, name, LoRALinear(base, config['rank'], config['alpha']))


def merged_checkpoint(checkpoint):
    """Merge a full adapter checkpoint into portable, ordinary model weights."""
    config = deepcopy(checkpoint['model_config'])
    lora = config.pop('lora', None)
    if not lora:
        raise ValueError('Checkpoint has no LoRA configuration')
    validate_lora(lora)
    state = dict(checkpoint['model'])
    prefixes = [k.removesuffix('lora_A') for k in state if k.endswith('.lora_A')]
    expected = lora['last_n_blocks'] * len(lora['targets'])
    if len(prefixes) != expected:
        raise ValueError(f'Expected {expected} LoRA modules, found {len(prefixes)}')
    for prefix in prefixes:
        a, b = state.pop(prefix + 'lora_A'), state.pop(prefix + 'lora_B')
        weight = state[prefix + 'weight']
        if a.shape != (lora['rank'], weight.shape[1]) or b.shape != (weight.shape[0], lora['rank']):
            raise ValueError(f'Invalid adapter shape at {prefix}')
        state[prefix + 'weight'] = (weight.float() + (lora['alpha'] / lora['rank']) * (b.float() @ a.float())).to(weight.dtype)
    if any('.lora_' in k for k in state):
        raise ValueError('Unmerged adapter tensors remain')
    # Do not carry optimizer/step state into a model-only deployment checkpoint.
    result = dict(model_config=config, model=state, merged_lora=lora)
    if 'validation' in checkpoint:
        result['source_validation'] = checkpoint['validation']
    return result
