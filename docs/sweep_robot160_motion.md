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

## Best flip에서 160 / 400토큰 추가 학습

설정: `configs/experiments/gaemi_flip_continue_tokens_total10000.json`
실행: `run_script/run_gaemi_flip_continue_tokens_total10000.sh`

초기 가중치는 CNN 없는 `robot160_flip_3000`의 step 1999이다. 로컬 `/home/jeon/Downloads/best_flip.pt`의 체크포인트 내 validation 메타데이터에서 REL 0.3482493114, delta1 0.6467669138, Points REL 0.2968272942를 확인했다. 학습 서버에는 이 파일을 `/home/coder/MoGe/pretrained/best_flip.pt`에 복사하거나 `--pretrained`로 실제 위치를 지정한다. 기존 backbone sweep의 best나 CNN 모델과 혼동하지 않는다.

두 실험은 같은 초기 가중치에서 각각 8,000 iteration을 추가 학습한다. 하나는 160토큰, 다른 하나는 400토큰이며, validation도 각 모델의 고정 토큰 수를 사용한다. 640×360 입력에서 각각 17×9 / 27×15 패치 격자(153 / 405개)다. 초기 checkpoint는 2,000번 업데이트한 모델이므로 총 이력은 10,000번이지만, 새 실험 로그 step은 0~7999이다. Optimizer·scheduler·EMA는 새로 시작한다. 과거 optimizer 상태를 그대로 이어가는 resume 실험은 아니다.

공통: CNN 없음, 좌우 반전 적용, mild 모션블러(확률 30%, 커널 3/5/7, 각도 -180~180도) 적용, 색상·원근 augmentation 없음, 배치 8×4=32, seed 0, warmup 200 후 constant LR. 기존 데이터 구성과 마지막 DINOv2 4개 블록·neck·모든 head 학습 범위를 유지한다. 후반 학습을 위한 후보로 기존 LR을 모두 절반으로 낮춘다: backbone/neck 5e-7, scale 5e-6, points/normal/mask 1.5e-6. 따라서 이번 160토큰 실험은 낮은 LR에서의 추가 학습 효과를 측정하며, 과거 학습 대비 시간만 달라지는 실험은 아니다. 두 새 실험 간에는 토큰 수만 다르다.

```bash
bash run_script/run_gaemi_flip_continue_tokens_total10000.sh --dry-run
bash run_script/run_gaemi_flip_continue_tokens_total10000.sh --smoke
bash run_script/run_gaemi_flip_continue_tokens_total10000.sh
# 다른 경로의 동일 checkpoint 사용
bash run_script/run_gaemi_flip_continue_tokens_total10000.sh --pretrained /path/to/best_flip.pt
```

학습 실험은 2개다. 별도로 각 토큰 수의 학습 전 validation을 한 번씩 수행해, 토큰만 바꾼 효과와 추가 학습·블러 augmentation의 결합 효과를 구분한다. summary의 `_initial` 행 2개는 학습 실험이 아니다. baseline을 포함해 비용을 확인하려면 기본 실행을 사용하고, 생략하려면 `--skip-baseline`을 지정한다. 메모리가 부족하면 `--batch-size 4 --validation-batch-size 4`로 두 실험을 함께 조정한다(학습 누적 8회로 유효 배치 32 유지).

250 step마다 평가하고, 각 실험의 최적 결과를 같은 토큰 수의 initial 결과와 비교한다. 400토큰 학습 이후 로봇 속도가 유지되는지는 별도 측정해야 한다. 학습량은 같아도 연산량과 학습 시간은 다르다.


### 동시 학습

기본 실행은 두 실험을 순차 처리한다. `--num-processes 2`는 실험 2개를 병렬로 실행하는 옵션이 아니라, 한 실험을 두 프로세스로 분산 학습하는 옵션이다.

GPU가 두 개라면 별도 터미널에서 각 GPU에 실험 하나씩 배정할 수 있다. 각 프로세스에는 GPU 하나만 노출하고 `--num-processes 1`을 유지한다. workspace와 로그도 분리한다. 아래 GPU 번호는 학습 PC에서 실제 확인한 번호로 바꾼다.

```bash
# 터미널 A
CUDA_VISIBLE_DEVICES=0 SWEEP_ROOT=workspace/continue_total10000_t160_run1 \
  bash run_script/run_gaemi_flip_continue_tokens_total10000.sh \
  --num-processes 1 --experiment flip_mild_continue_t160_add8000

# 터미널 B
CUDA_VISIBLE_DEVICES=1 SWEEP_ROOT=workspace/continue_total10000_t400_run1 \
  bash run_script/run_gaemi_flip_continue_tokens_total10000.sh \
  --num-processes 1 --experiment flip_mild_continue_t400_add8000
```

동일 GPU에 두 실험을 배정하는 것도 기술적으로 가능하지만 두 모델·optimizer·EMA·활성값·CUDA context가 동시에 메모리를 사용한다. 학습뿐 아니라 validation peak도 포함해 여유가 있어야 한다. GPU·CPU·스토리지를 공유하므로 총 완료 시간이 줄어든다고 보장할 수 없다. 각각 `--smoke`로 실행 가능 여부를 확인하되 smoke는 validation을 실행하지 않으므로 충분조건은 아니다. 현재 노트북은 RTX 3050 Laptop 4GB 한 대로 확인됐으며 동시 학습 검증은 수행하지 않았다.

기존 `_3000.sh` 실행 경로와 manifest도 추가 8,000회로 갱신했다. 기존에 실행 중인 프로세스나 이미 생성한 workspace config에는 이 변경이 적용되지 않는다. 같은 best_flip.pt(기존 2,000회)에서 새 workspace로 시작할 때 총 10,000회다. 다른 초기 모델이나 `--num-iterations` override를 사용하면 총 이력이 달라진다.

현재 학습률 스케줄은 warmup 200 후 constant를 유지한다. 장기 학습용 cosine decay는 아직 적용하지 않았다. 현재 validation은 정상 영상 평가이며, 고정 합성 블러 평가와 실제 로봇 블러 평가는 아직 추가되지 않았다. 따라서 현재 best는 정상 validation REL 기준이며 블러 강건성 최적 모델임을 의미하지 않는다.
