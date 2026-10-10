
# Training 

This document provides instructions for training and finetuning the MoGe model.

## Additional Requirements

Training needs the `train` extra declared in [`pyproject.toml`](../pyproject.toml):

```bash
uv sync --extra train           # or: pip install -e ".[train]"
```

It adds `accelerate` (used for distributed training), `sympy`, and all three
logging backends: `tensorboard`, `wandb` and `mlflow`.

## Data preparation

### Dataset format

Each dataset should be organized as follows:

```
somedataset
├── .index.txt          # A list of instance paths
├── folder1 
│   ├── instance1       # Each instance is in a folder
│   │   ├── image.jpg   # RGB image.
│   │   ├── depth.png   # 16-bit depth. See moge/utils/io.py for details
│   │   ├── meta.json   # Stores "intrinsics" as a 3x3 matrix
│   │   └── ...         # Other componests such as segmentation mask, normal map etc.
...
```

* `.index.txt` is placed at top directory to store a list of instance paths in this dataset. The dataloader will look for instances in this list. You may also use a custom split, e.g. `.train.txt`, `.val.txt` and specify it in the configuration file.

* For depth images, it is recommended to use `read_depth()` and `write_depth()` in [`moge/utils/io.py`](../moge/utils/io.py) to read and write depth images. The depth is stored in logarithmic scale in 16-bit PNG format, offering a balanced precision, dynamic range and compression ratio compared to 16-bit and 32-bit EXR and linear depth formats. It also encodes `NaN` and `Inf` values for invalid depth values.

* The `meta.json` should be a dictionary containing the key `intrinsics`, which are **normalized** camera parameters. You may put more metadata.

* We also support reading and storing segementation masks for evaluation data (see paper evaluation of local points), which are saved in PNG format with semantic labels stored in png metadata as JSON strings. See `read_segmentation()` and `write_segmentation()` in [`moge/utils/io.py`](../moge/utils/io.py) for details.


### Visual inspection

We provide a script to visualize the data and check the data quality. It will export the instance as a PLY file for visualization of point cloud.

```bash
python moge/scripts/vis_data.py PATH_TO_INSTANCE --ply [-o SOMEWHERE_ELSE_TO_SAVE_VIS]
```

### DataLoader

Our training dataloaders is customized to handle loading data, performing perspective crop, and augmentation in a multithreading pipeline. Please refer to [`moge/train/dataloader.py`](../moge/train/dataloader.py) if you have any concern.


## Configuration

See [`configs/train/v1.json`](../configs/train/v1.json) for an example configuration file. The configuration file defines the hyperparameters for training the MoGe model. 
Here is a commented configuration for reference:

```json
{
    "data": {
        "aspect_ratio_range": [0.5, 2.0],               # Range of aspect ratio of sampled images
        "area_range": [250000, 1000000],                # Range of sampled image area in pixels
        "clamp_max_depth": 1000.0,                      # Maximum far/near
        "flip_augmentation": true,                      # Random horizontal flip
        "perspective_warp": true,                       # Random FOV/center perspective transform
        "resize": true,                                 # Resize samples to the selected training image size
        "center_augmentation": 0.5,                     # Ratio of center crop augmentation
        "fov_range_absolute": [1, 179],                 # Absolute range of FOV in degrees
        "fov_range_relative": [0.01, 1.0],              # Relative range of FOV to the original FOV
        "image_augmentation": ["jittering", "jpeg_loss", "blurring"],       # List of image augmentation techniques
        "datasets": [ 
            {
                "name": "TartanAir",                    # Name of the dataset. Name it as you like.
                "path": "data/TartanAir",               # Path to the dataset
                "label_type": "synthetic",              # Label type for this dataset. Losses will be applied accordingly. see "loss" config
                "weight": 4.8,                          # Probability of sampling this dataset
                "index": ".index.txt",                  # File name of the index file.  Defaults to .index.txt
                "depth": "depth.png",                   # File name of depth images. Defaults to depth.png
                "center_augmentation": 0.25,            # Below are dataset-specific hyperparameters. Overriding the global ones above.
                "fov_range_absolute": [30, 150],
                "fov_range_relative": [0.5, 1.0],
                "image_augmentation": ["jittering", "jpeg_loss", "blurring", "shot_noise"]
            }
        ]
    },
    "model_version": "v1",                 # Model version. If you have multiple model variants, you can use this to switch between them.
    "model": {                             # Model hyperparameters. Will be passed to Model __init__() as kwargs.
        "encoder": "dinov2_vitl14",
        "remap_output": "exp",
        "intermediate_layers": 4,
        "dim_upsample": [256, 128, 64],
        "dim_times_res_block_hidden": 2,
        "num_res_blocks": 2,
        "num_tokens_range": [1200, 2500],
        "last_conv_channels": 32,
        "last_conv_size": 1
    },
    "optimizer": {                          # Reflection-like optimizer configurations. See moge.train.utils.py build_optimizer() for details.
        "params": [                         # One entry per parameter group. "name" is only used for logging.
            {"name": "head", "type": "AdamW", "params": {"include": ["*"], "exclude": ["*backbone.*"]}, "lr": 1e-4},
            {"name": "backbone", "type": "AdamW", "params": {"include": ["*backbone.*"]}, "lr": 1e-5}
        ]
    },
    "lr_scheduler": {                       # Reflection-like lr_scheduler configurations. See moge.train.utils.py build_lr_scheduler() for details.
        "type": "SequentialLR",
        "params": {
            "schedulers": [
                {"type": "LambdaLR", "params": {"lr_lambda": ["1.0", "max(0.0, min(1.0, (epoch - 1000) / 1000))"]}},
                {"type": "StepLR", "params": {"step_size": 25000, "gamma": 0.5}}
            ],
            "milestones": [2000]
        }
    },
    "low_resolution_training_steps": 50000, # Total number of low-resolution training steps. It makes the early stage training faster. Later stage training on varying size images will be slower.
    "loss": {                               # Losses are keyed by label type, then grouped by the prediction they supervise
        "invalid": {},                      # invalid instance due to runtime error when loading data
        "synthetic": {                      # Below are loss hyperparameters
            "points": {                     # Terms supervising the predicted point map
                "global": {"function": "affine_invariant_global_loss", "weight": 1.0, "params": {"align_resolution": 32}},
                "patch_4": {"function": "affine_invariant_local_loss", "weight": 1.0, "params": {"level": 4, "align_resolution": 16, "num_patches": 16}},
                "patch_16": {"function": "affine_invariant_local_loss", "weight": 1.0, "params": {"level": 16, "align_resolution": 8, "num_patches": 256}},
                "patch_64": {"function": "affine_invariant_local_loss", "weight": 1.0, "params": {"level": 64, "align_resolution": 4, "num_patches": 4096}},
                "normal": {"function": "normal_loss", "weight": 1.0}
            },
            "mask": {                       # Terms supervising the predicted infinity mask
                "mask": {"function": "mask_l2_loss", "weight": 1.0}
            }
        },
        "sfm": {
            "points": {
                "global": {"function": "affine_invariant_global_loss", "weight": 1.0, "params": {"align_resolution": 32}},
                "patch_4": {"function": "affine_invariant_local_loss", "weight": 1.0, "params": {"level": 4, "align_resolution": 16, "num_patches": 16}},
                "patch_16": {"function": "affine_invariant_local_loss", "weight": 1.0, "params": {"level": 16, "align_resolution": 8, "num_patches": 256}}
            },
            "mask": {
                "mask": {"function": "mask_l2_loss", "weight": 1.0}
            }
        },
        "lidar": {
            "points": {
                "global": {"function": "affine_invariant_global_loss", "weight": 1.0, "params": {"align_resolution": 32}},
                "patch_4": {"function": "affine_invariant_local_loss", "weight": 1.0, "params": {"level": 4, "align_resolution": 16, "num_patches": 16}}
            },
            "mask": {
                "mask": {"function": "mask_l2_loss", "weight": 1.0}
            }
        }
    }
}
```

`flip_augmentation`, `perspective_warp`, and `resize` can be overridden per
dataset like the other augmentation settings. Setting both `perspective_warp`
and `resize` to `false` preserves the source image geometry, depth samples, and
intrinsics without remapping; `image_augmentation` remains independent. In that
mode, samples grouped into one batch must have matching source dimensions so
they can be stacked.

## Run Training 

Launch the training script [`moge/train/train_moge12.py`](../moge/train/train_moge12.py) as a module from the repository root. It trains both MoGe-1 and MoGe-2; which one you get is decided by `model_version` in the config. Note that we use [`accelerate`](https://github.com/huggingface/accelerate) for distributed training. 

```bash
uv run accelerate launch \
    --num_processes 8 \
    --module moge.train.train_moge12 \
    --config configs/train/v2.json \
    --name train_moge2 \
    --workspace workspace/moge2 \
    --batch_size_forward 2 \
    --gradient_accumulation_steps 2 \
    --enable_gradient_checkpointing True \
    --precision mixed_bf16 \
    --enable_ema True \
    --log_every 100 \
    --log_type tensorboard \
    --log_type wandb \
    --vis_every 1000 
```


## Finetuning

To finetune the pre-trained MoGe model, first download the model checkpoint and put it in a local directory, e.g. `pretrained/moge-vitl.pt`, then pass that checkpoint with `--initial_checkpoint`.

> NOTE: when finetuning pretrained MoGe model, a much lower learning rate is required. 
The suggested learning rate for finetuning is not greater than 1e-5 for the head and 1e-6 for the backbone. 
And the batch size is recommended to be 32 at least. 
The settings in default configuration are not optimal for specific datasets and may require further tuning.

```bash
uv run accelerate launch \
    --num_processes 8 \
    --module moge.train.train_moge12 \
    --config configs/train/v2.json \
    --name finetune_moge2 \
    --workspace workspace/finetune_moge2 \
    --batch_size_forward 2 \
    --gradient_accumulation_steps 2 \
    --initial_checkpoint pretrained/moge-2-vitl.pt \
    --enable_gradient_checkpointing True \
    --precision mixed_bf16 \
    --log_every 100 \
    --log_type tensorboard \
    --log_type wandb \
    --vis_every 1000 
```


### Choosing which MoGe-2 heads to train

The `train_moge12` entry point accepts two independent boolean options:

| Option | Default | Controlled module |
| --- | --- | --- |
| `--train_scale_head` | `True` | `scale_head`: metric scale factor MLP |
| `--train_points_head` | `True` | `points_head`: affine-invariant point map Conv head |

For example, add `--train_scale_head True --train_points_head False` to the
finetuning command above to train the scale head while freezing the point head.
One head can be frozen, but at least one must remain trainable. These are command-line options; the effective
values are recorded in the workspace config under `trainable_heads`.

For MoGe-2, all other modules are frozen by default: the entire encoder (including
feature projections), neck, normal head and mask head. Only the selected scale
and point heads are included in the optimizer. Frozen modules stay in eval mode
while the selected heads use train mode. Losses are still computed for logging;
only paths connected to the selected heads contribute gradients. Both options
set to `False` are rejected because there would be nothing to train. MoGe-1
training behavior is unchanged.

When resuming optimizer state, use the same head settings as the original run.
Checkpoints from the earlier full-model training behavior have different optimizer
groups and must instead be used as model-only initialization in a new workspace.
To change which heads are trained, start in a new workspace with
`--checkpoint none --initial_checkpoint PATH_TO_MODEL.pt`, using a model-only
checkpoint (for example, a pretrained checkpoint or the saved `00001000.pt` model
shard), rather than a checkpoint containing optimizer state.

### Validation during MoGe-2 training

The separate training-sample visualization controlled by `--vis_every` writes
point maps as EXR when OpenCV supports it. If the EXR writer is unavailable, both
GT and prediction point maps are saved as `points*.npy` instead, preserving XYZ
channel order and floating-point values (load with `numpy.load`). RGB/depth/normal
previews remain images. This does not affect validation's PLY exports below.

Validation is optional and disabled when no `validation` section or `--val_config`
is supplied. Copy and edit [`configs/validation/moge2.json`](../configs/validation/moge2.json),
then add this option to the training/finetuning command:

```bash
--val_config configs/validation/moge2.json
```

Add `--validate_before_training True` to evaluate the loaded checkpoint once
before the first optimizer update. This is also useful with
`--num_iterations 0` for a validation-only pretrained baseline run.

The file contains the validation settings themselves (not a surrounding
`validation` key). Alternatively, place the same object under `validation` in
your training JSON; `--val_config` overrides that section.

```json
{
    "every": 1000,
    "monitor": "depth_metric/rel",
    "mode": "min",
    "metric_groups": "global,metric",
    "resolution_level": 9,
    "max_samples": null,
    "seed": 0,
    "datasets": {
        "simulation_val": {
            "path": "/path/to/validation_dataset",
            "split": ".val.txt",
            "width": 640,
            "height": 480,
            "depth": "depth.png",
            "depth_unit": 1.0
        }
    }
}
```

Each `datasets` entry uses the evaluation loader options documented in
[`docs/eval.md`](eval.md). `split` lists sample directories relative to `path`;
the training loader's corresponding option is named `index`. Keep validation
samples out of the training index. For simulation data, split by scene/route
rather than neighboring frames to avoid train/validation overlap.

- `every`: evaluate after this many optimizer iterations and at the final step.
  Saved step numbers are zero-based, so `every: 1000` first evaluates step 999.
- `metric_groups`: existing evaluation suites/categories/groups, e.g.
  `global,metric` or just `depth_metric`. No validation training loss is computed.
- `monitor`: metric path in the sample-weighted aggregate used to select the best
  model. Default `depth_metric/rel` is absolute relative depth error (lower is
  better). For `depth_metric/delta1`, set `mode: "max"`.
- `max_samples`: first N indexed samples **per dataset** after any `subset`
  stride; `null` evaluates all samples. Ordering and preprocessing are fixed,
  with no training augmentation. `resolution_level` controls inference tokens.
- `depth_unit`: conversion to meters; use `1.0` for meter-valued depth. Relative
  datasets may omit it and use an invariant monitor such as
  `depth_affine_invariant/rel`. Validation also applies `metric_scale` or
  `depth_scale` from each sample's `meta.json`, matching training. The optional
  dataset flag `metric_from_meta: true` marks only annotated samples as metric.

Validation evaluates the **current model**, not EMA, using inferred camera FOV
(no ground-truth FOV input). All ranks synchronize; rank zero runs the unwrapped
model in eval/no-grad mode and restores training modes and random states afterward.
Samples without valid positive depth are skipped and counted. Empty datasets,
missing monitor metrics, and non-finite predictions/metrics fail explicitly.
For mixed metric/non-metric datasets, each metric averages only eligible samples.

Outputs are saved independently of `--log_every` and `--checkpoint_every`:

- `validation/step_XXXXXXXX.json`: per-dataset and aggregate metrics, plus counts.
- `val/<dataset-or-mean>/<metric>`: TensorBoard, W&B and/or MLflow scalars using
  whichever `--log_type` backends are enabled.
- `checkpoint/best.pt`: model weights, `model_config`, and validation metadata;
  overwritten atomically only on strict improvement.
- `validation/best.json`: best score and step; restored when continuing in the
  same workspace. Changing validation settings or split contents requires a new
  workspace, so incomparable scores are not silently reused.

`best.pt` is a **model-only** checkpoint, loadable with
`MoGeModel.from_pretrained(...)` or `--initial_checkpoint`. It does not contain
optimizer/scheduler state and does not replace `latest.pt`. Use the normal
`--checkpoint latest` flow for full training-state resumption.

#### Validation 3D comparisons

Both example validation configs enable point-cloud exports. Add or adjust:

```json
"visualization": {
    "num_samples": 8,
    "max_points": 50000
}
```

At each validation step, up to `num_samples` fixed indices per dataset are
selected evenly across the evaluated split (after `subset` and `max_samples`).
Invalid selected samples are skipped rather than replaced, so valid selections
stay comparable between steps. Set `num_samples: 0` to disable exports. Changing
visualization settings is allowed when resuming an existing workspace.

Files are saved in
`validation/step_XXXXXXXX/<dataset>/<sample-index>/`:

- `gt.ply`, `pred.ply`: RGB-colored GT and prediction point clouds.
- `overlay.ply`: both clouds in one file; GT is cyan and prediction is orange.
- `image.png`: the processed validation RGB image.
- `info.json`: original sample path, intrinsics, units, coordinate convention,
  point counts and filtering details.

Open the PLY files in a point-cloud viewer such as CloudCompare or MeshLab.
Coordinates retain the OpenCV camera frame (+X right, +Y down, +Z forward).
No scale/shift alignment or display offset is applied, so metric-scale errors
remain visible. Relative-depth GT is explicitly marked in `info.json` and should
not be interpreted as a metric-aligned overlay.

`max_points` caps each cloud by deterministically subsampling GT-valid pixels;
predictions use those same pixel locations. Non-finite or non-positive-Z points
are omitted. The model's predicted mask is not applied, matching metric
evaluation. Each overlay contains at most twice `max_points` points. These
exports reuse validation inference and do not change the full-resolution metrics.

## Training MoGe-3

MoGe-3 is trained upon a pretrained MoGe-2 checkpoint. 

```bash
uv run accelerate launch \
    --num_processes 8 \
    --module moge.train.train_moge3 \
    --config configs/train/v3.json \
    --name train_moge3 \
    --workspace workspace/moge3 \
    --initial_checkpoint pretrained/moge-2-vitl.pt \
    --checkpoint latest \
    --batch_size_forward 1 \
    --gradient_accumulation_steps 6 \
    --precision mixed_bf16 \
    --enable_gradient_checkpointing True \
    --log_every 100 \
    --log_type tensorboard \
    --log_type wandb \
    --vis_every 1000 
```

The MoGe-3 training config carries a few extra keys:

| Key | Meaning |
| --- | --- |
| `refine_steps` | Number of refinement iterations per forward pass |
| `refine_ratio` | Fraction of accumulation micro-batches drawn from `refine_data` rather than `norefine_data` |
| `refiner_detach_backbone_until` | Step until which the refiner trains on detached encoder features |

It also splits the dataset list into two pipelines, `norefine_data` and `refine_data`, instead of the single `data` key used by v1/v2. Both take the same fields as `data` above.

Its `loss` section follows the same label type → group → term layout, with one addition: every term in the `points` group lists the refiner iterations it applies to via `apply_steps` (`[0]` = the base prediction only, `[0, 1, 2, 3]` = base plus all three refine steps). Datasets used only for refinement carry label `D` in the shipped config.
