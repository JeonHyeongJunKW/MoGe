"""Train selected MoGe-2 decoder components and optional trailing DINOv2 blocks."""

from torch import nn


def _selected_backbone_blocks(model, count):
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('train_backbone_last_n_blocks must be a non-negative integer')
    if count == 0:
        return []
    backbone = getattr(getattr(model, 'encoder', None), 'backbone', None)
    if backbone is None or not hasattr(backbone, 'blocks'):
        raise ValueError('Cannot unfreeze DINOv2 blocks: encoder.backbone.blocks is absent')
    if getattr(backbone, 'chunked_blocks', False):
        # DINO chunks prepend Identity placeholders to preserve block indices.
        blocks = [block for chunk in backbone.blocks for block in chunk
                  if not isinstance(block, nn.Identity)]
    else:
        blocks = list(backbone.blocks)
    if count > len(blocks):
        raise ValueError(f'Requested {count} backbone blocks, but only {len(blocks)} exist')
    return list(enumerate(blocks))[-count:]



def configure_moge2_heads(model, trainable_heads, train_backbone_last_n_blocks=0):
    selected_blocks = _selected_backbone_blocks(model, train_backbone_last_n_blocks)
    allowed = {'neck', 'scale_head', 'points_head', 'normal_head', 'mask_head'}
    if set(trainable_heads) - allowed:
        raise ValueError(f'Unsupported trainable components: {set(trainable_heads) - allowed}')
    if not any(trainable_heads.values()) and not selected_blocks:
        raise ValueError('MoGe-2 requires at least one trainable decoder component or backbone block')
    for name, enabled in trainable_heads.items():
        if enabled and getattr(model, name, None) is None:
            raise ValueError(f'Cannot train {name}: it is absent from the model config')
    model.requires_grad_(False)
    for name, enabled in trainable_heads.items():
        head = getattr(model, name, None)
        if head is not None:
            head.requires_grad_(enabled)
        print(f'{name}: {"trainable" if enabled else "frozen"}')

    for index, block in selected_blocks:
        block.requires_grad_(True)
        print(f'backbone block {index}: trainable')
    if not selected_blocks:
        print('backbone: frozen')


def set_moge2_head_training_modes(model, trainable_heads, train_backbone_last_n_blocks=0):
    # Frozen feature extractors must also retain their running statistics and
    # deterministic behavior (e.g. batch norm/dropout in alternative configs).
    model.eval()
    for name, enabled in trainable_heads.items():
        if enabled:
            getattr(model, name).train()
    for _, block in _selected_backbone_blocks(model, train_backbone_last_n_blocks):
        block.train()
