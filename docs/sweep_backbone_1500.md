# DINOv2 partial-backbone sweep (1500 steps)

Run from the repository root in the training virtual environment:

```bash
bash run_script/run_gaemi_backbone_sweep_1500.sh --smoke
bash run_script/run_gaemi_backbone_sweep_1500.sh
```

The smoke invocation runs 10 steps per experiment without validation or checkpoint
saving. Each invocation creates a separate output directory. Training is not
started by preparing the configs.

Defaults: original checkpoint `/home/coder/MoGe/pretrained/moge-2-vits.pt`,
1500 optimizer steps, one GPU, forward batch 8 and accumulation 4 (effective 32).
Every experiment starts independently from that checkpoint, with a fresh optimizer.
This is not continuation of the previous 5000-step run.
Use `--pretrained /path/to/model.pt` to override the initial model for all runs.

Seven training experiments:

| Last trainable DINOv2 blocks | Backbone LR |
| --- | --- |
| 0 (frozen control) | none |
| 2 | 1e-7 |
| 2 | 3e-7 |
| 2 | 1e-6 |
| 4 | 1e-7 |
| 4 | 3e-7 |
| 4 | 1e-6 |

Decoder rates stay fixed: scale head 1e-5, points/normal/mask heads 3e-6,
neck 1e-6. All optimizer groups use 200-step linear warmup followed by constant LR.
The ViT-S selections are blocks 10–11 or 8–11 (zero-based); other backbone
parameters and encoder output projections remain frozen.

The data composition, split paths, losses, resolution, and no-augmentation policy
match the existing neck batch128 sweep. No glass/fence samples are removed.
Validation uses the existing complete `.val.txt`, batch 8, every 250 steps
(249, 499, 749, 999, 1249, 1499), with per-image metrics
and PLY output for eight samples. Depth-bin calculation, CSV/JSON export, and terminal tables are disabled by default
(`depth_bins: null`). An initial pretrained evaluation runs once
in addition to the seven training experiments; `--skip-baseline` skips it.

Useful commands:

```bash
# Generate configs and commands only.
bash run_script/run_gaemi_backbone_sweep_1500.sh --dry-run

# One selected experiment.
bash run_script/run_gaemi_backbone_sweep_1500.sh \
  --experiment backbone_last2_lr3e-7_1500

# If forward batch 8 runs out of GPU memory, preserve effective batch 32 as 4 x 8.
bash run_script/run_gaemi_backbone_sweep_1500.sh --batch-size 4
```

The script inherits the active Python environment; `PYTHON_BIN` can select one
explicitly. `--data-root` relocates train and validation together.
Results live under `workspace/gaemi_backbone_sweep_1500/<timestamp>/`, including
resolved train configs, command.json, run.log, checkpoints, validation metrics,
TensorBoard logs and summary.csv/summary.md. The summary includes selected block
count, component learning rates and peak GPU memory. Existing sweep manifests
remain backward-compatible and keep the backbone frozen.

The launcher and freeze/gradient behavior have CPU regression tests. Actual GPU
memory and full-run quality must be checked on the training machine.
