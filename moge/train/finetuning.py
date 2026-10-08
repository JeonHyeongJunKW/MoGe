"""Train only selected MoGe-2 output heads."""


def configure_moge2_heads(model, trainable_heads):
    if not any(trainable_heads.values()):
        raise ValueError('MoGe-2 requires at least one of --train_scale_head or --train_points_head to be True')
    for name, enabled in trainable_heads.items():
        if enabled and getattr(model, name, None) is None:
            raise ValueError(f'Cannot train {name}: it is absent from the model config')
    model.requires_grad_(False)
    for name, enabled in trainable_heads.items():
        head = getattr(model, name, None)
        if head is not None:
            head.requires_grad_(enabled)
        print(f'{name}: {"trainable" if enabled else "frozen"}')


def set_moge2_head_training_modes(model, trainable_heads):
    # Frozen feature extractors must also retain their running statistics and
    # deterministic behavior (e.g. batch norm/dropout in alternative configs).
    model.eval()
    for name, enabled in trainable_heads.items():
        if enabled:
            getattr(model, name).train()
