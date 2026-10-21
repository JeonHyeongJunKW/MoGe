# Robot 640x360 / num_tokens=160 sweep

Run in the training environment, from /home/coder/MoGe:

```bash
bash run_script/run_gaemi_robot160_motion_sweep.sh --smoke
bash run_script/run_gaemi_robot160_motion_sweep.sh
```

Four independent 3000-step training experiments, all initialized from the original
/home/coder/MoGe/pretrained/moge-2-vits.pt (override with --pretrained).
The previous best finetuning *hyperparameters* are reused; the previous best
checkpoint is not automatically loaded.

| Experiment | Flip probability | RGB motion blur probability | Kernel sizes at 640x360 |
| --- | --- | --- | --- |
| robot160_clean_3000 | 0 | 0 | none |
| robot160_flip_3000 | 0.5 | 0 | none |
| robot160_flip_motion_mild_3000 | 0.5 | 0.3 | 3, 5, 7 px |
| robot160_flip_motion_strong_3000 | 0.5 | 0.5 | 5, 9, 13, 17 px |

Angles are uniformly sampled from -180 to 180 degrees. Blur uses a normalized,
supersampled straight-line kernel and reflected borders. It is applied only to
RGB after geometric preprocessing, with the existing per-sample RNG.
Blur leaves GT depth, normals, intrinsics and masks unchanged. Horizontal flips
reflect RGB/depth/normal maps, negate the normal X component, and reflect the
normalized principal point (cx -> 1-cx) and skew. Masks/points derived downstream
use the transformed depth and intrinsics. This approximates linear
motion smear, not depth-dependent camera motion, rolling shutter or temporal
occlusion. The probabilities keep clean examples in both augmented runs.
Legacy string augmentations remain supported; configurable motion blur is a dict
with type, probability, kernel_sizes and angle_range.

All runs use:
- image_sizes [[640,360]], num_tokens_range [160,160];
- last four DINOv2 blocks trainable, backbone LR 1e-6;
- neck LR 1e-6, scale head 1e-5, points/normal/mask heads 3e-6;
- forward batch 8, accumulation 4 (effective batch 32 on one GPU);
- 200-step warmup then constant LR, bf16, gradient checkpointing, EMA;
- horizontal flips with probability 0.5 in the three augmented runs; no other augmentations;
- full clean validation every 250 steps; 1m depth bins disabled;
- per-image validation metrics and eight sample point-cloud exports.

Data entries use:
1. /home/coder/dataset/20261017_train with
   .review-versions/reviewed_20261018_v1/.train.txt, weight 20817.
2. /home/coder/dataset/near_obstacles_train with
   .review-versions/wall_transparent/.train.txt, weight 9351.

The weights sample the two sources in proportion to their reviewed counts
(approximately 69% / 31%), rather than 50/50. Sampling is with replacement;
3000 steps are not a whole-number epoch count. Validation uses the original
reviewed_20261018_v1/.val.txt under 20261017_train. No new validation split is made.
--data-root relocates the common dataset parent while retaining both child paths.

At aspect ratio 640/360, MoGe rounds sqrt(160/aspect) and sqrt(160*aspect) to
9 and 17: 153 spatial tokens (the backbone also has a CLS token).
The public input and depth output are 640x360. Internally DINOv2 resizes to
238x126 pixels for its 14px patches; it does not encode all 640x360 pixels at
native patch resolution. Smaller tokens can reduce thin-obstacle detail.

Validation calls infer with resolution_level=0; [160,160] makes the token count
160 for any resolution level. Saved model configs retain this range.
On the robot pass the token argument explicitly:

```python
# images: float tensor [3, 3, 360, 640], RGB in [0,1],
# ordered left, front, right, on the same device as the model.
model.eval()
with torch.inference_mode():
    result = model.infer(images, num_tokens=160, apply_mask=False)
depth = result["depth"]  # [3, 360, 640]
```

RGB order, image resize/crop and associated camera intrinsics must match deployment.
Training batch 32 and robot inference batch 3 are separate settings.
The current dataset's front_left camera is not renamed or treated as front
by this sweep; deployment camera/viewpoint differences remain to be evaluated.

Outputs: workspace/gaemi_robot160_motion_sweep/<timestamp>/.
Use --dry-run to prepare configs only, --experiment NAME to select a run.
The pretrained baseline is evaluated once at the same 160-token setting.
Clean validation alone does not establish motion-blur robustness: also compare
the selected checkpoints on the same real robot blurry sequences and clean
frames before selecting a deployed model.
