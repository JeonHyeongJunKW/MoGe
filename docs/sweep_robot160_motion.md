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

### 누적 30,000회 continuation (400 → 160토큰)

`run_script/run_gaemi_flip_continue_tokens_total30000.sh`는 각 토큰의 누적
10,000회 best 모델에서 20,000회를 추가한다. 기존 best.pt는 모델 가중치만
불러오므로 optimizer/scheduler/EMA는 새로 초기화된다. 입력 640×360,
배치 8×4, 마지막 DINOv2 4개 블록, flip + mild motion blur 조건을 유지한다.

원격 PC에 시작 모델을 각각 준비한다:
- `/home/coder/MoGe/pretrained/best_flip_mild_t400_total10000.pt`
- `/home/coder/MoGe/pretrained/best_flip_mild_t160_total10000.pt`

```bash
# 명령 및 설정만 생성
bash run_script/run_gaemi_flip_continue_tokens_total30000.sh --dry-run
# 두 모델의 초기 평가 후, 400토큰 학습 → 160토큰 학습
bash run_script/run_gaemi_flip_continue_tokens_total30000.sh
# 400토큰만 실행
bash run_script/run_gaemi_flip_continue_tokens_total30000.sh \
  --experiment flip_mild_continue_t400_add20000
```

추가 학습 기준 200회 warmup, 5,000회까지 기존 LR 유지,
5,000~20,000회 cosine 감소로 최종 LR을 시작 LR의 20%로 만든다.
시작 LR은 backbone/neck 5e-7, points/normal/mask 1.5e-6, scale 5e-6이다.
`--num-iterations`만 바꾸면 cosine 종료 시점은 바뀌지 않는다.
이미지별 validation 보고서는 비활성화한다. 전체 지표와 best 모델은 저장한다.
`--pretrained`는 선택한 모든 실험의 시작 모델을 동일 경로로 덮어쓰므로,
기본 실행에서는 생략한다. 초기 평가는 별도 `_initial` 폴더에 저장되므로
후속 best가 초기 모델보다 좋은지는 summary에서 함께 비교해야 한다.

### Large teacher 사전 평가

누적 30,000회 sweep은 이제 `teacher_moge2_vitl_normal_t1200` 평가를 가장 먼저
실행한다. 공개 `Ruicheng/moge-2-vitl-normal`의 `model.pt`를 Hugging Face에서
자동 다운로드/캐시하며 체크포인트의 모델 구조를 그대로 사용한다.
640×360 validation, 1,200토큰, batch 1, FP32, 학습 없음이다.
Large와 Small의 비교에는 모델 크기뿐 아니라 토큰 수의 차이도 포함된다.

```bash
# Teacher 평가만 실행 (Small 체크포인트 불필요)
bash run_script/run_gaemi_flip_continue_tokens_total30000.sh --teacher-only
# Teacher → Small 초기 평가 → 400 추가 학습 → 160 추가 학습
bash run_script/run_gaemi_flip_continue_tokens_total30000.sh
# 이미 teacher 평가를 마쳤다면 생략
bash run_script/run_gaemi_flip_continue_tokens_total30000.sh --skip-teacher
```

Teacher 결과는 sweep의 summary.csv/summary.md와
`teacher_moge2_vitl_normal_t1200/validation/{best,step_00000000}.json`에 저장된다.
`teacher_config.json`은 실제 모델 구조/평가 조건, `evaluation_memory.json`은
추론 메모리 사용량을 기록한다. 학습 peak와 혼동하지 않도록 summary의
training memory 칸에는 teacher 추론 메모리를 채우지 않는다.
모델 가중치를 중복 저장하지 않으며 per-image JSON/CSV도 생성하지 않는다.
Teacher 평가 실패 시 후속 학습을 시작하지 않는다. `--smoke`는 기존처럼
학습 메모리 점검만 수행하므로 teacher 평가를 생략한다.
오프라인 실행은 manifest의 teacher_evaluations[0].pretrained를 다운로드한
Large 체크포인트의 절대 경로로 변경하면 된다. `--pretrained`는 Small 실험에만 적용된다.

### Large teacher 학습 추가

현재 전체 순서는 공개 Large 사전 평가 → Small 초기 평가 →
Large 1,200토큰 2,000회 학습 → Small 400토큰 추가 20,000회 →
Small 160토큰 추가 20,000회이다. Large는 공개 가중치에서 시작하므로
이 데이터에서 2,000회, Small은 기존 학습을 합쳐 30,000회다.
Large best 모델을 Small에 자동 증류하거나 학습 데이터로 전달하지는 않는다.

Large 실험 이름은 `teacher_vitl_t1200_train2000`이다.
DINOv2 마지막 4개 블록과 neck, scale/points/normal/mask head를 학습한다.
시작 LR, flip + mild blur와 데이터는 Small continuation과 동일하다.
Large는 200회 warmup, 500회까지 LR 유지, 500~2,000회 cosine 감소로
최종 LR을 시작 LR의 20%로 낮춘다. Small의 20,000회 스케줄은 유지한다.
Large 기본 forward batch는 1, accumulation은 32로 유효 배치 32를 유지한다.
Large validation batch도 1이며, Small은 기존 train/validation batch 8을 유지한다.
실제 VRAM은 아직 검증하지 않았으므로 다음 smoke로 확인할 수 있다:

```bash
bash run_script/run_gaemi_flip_continue_tokens_total30000.sh \
  --experiment teacher_vitl_t1200_train2000 --smoke
# 사전 평가와 Large 학습만 실행
bash run_script/run_gaemi_flip_continue_tokens_total30000.sh \
  --experiment teacher_vitl_t1200_train2000
```

`--teacher-only`는 여전히 공개 teacher의 평가만 수행한다.
`--skip-teacher`는 사전 평가만 생략하며 Large 학습을 생략하지 않는다.
Small만 학습하려면 `--experiment`로 Small 실험을 선택한다.
`run_checkpoint_experiment.py`가 HF 캐시/체크포인트에서 실제 Large 모델
구조를 읽어 train_config.json의 model을 교체한 뒤 학습 프로세스를 실행한다.
따라서 dry-run의 model 필드는 임시 Small 템플릿이며 실제 Large 구조는
실행 시 확정된다. 최종 명령은 resolved_command.json에 저장된다.
`--pretrained`는 이 Large 실험의 checkpoint_model_source를 덮어쓰지 않는다.
오프라인에서는 checkpoint_model_source와 teacher_evaluations의 pretrained를
같은 로컬 Large 가중치 경로로 지정한다.

### Small 400토큰 backbone 범위 비교 (2026-10-29)

`run_script/run_gaemi_t400_backbone_sweep_3000.sh`는
`/home/coder/MoGe/pretrained/best_flip_mild_t400_total18500.pt`에서 시작한다.
이 파일은 추가 학습 step 8499, validation REL 0.2626725414의 Small best이며,
이전 학습 포함 18,500회를 거친 모델이다. 각 실험은 독립적으로 3,000회
추가 학습해 누적 21,500회에 해당한다. optimizer/scheduler/EMA는 초기화한다.

| 실험 | 마지막 학습 블록 수 | backbone LR |
| --- | --- | --- |
| t400_last4_lr5e-7_add3000 | 4 | 5e-7 |
| t400_last8_lr5e-7_add3000 | 8 | 5e-7 |
| t400_last12_lr5e-7_add3000 | 12 | 5e-7 |
| t400_last12_lr1e-7_add3000 | 12 | 1e-7 |

모두 400토큰, 640×360, forward 8 × accumulation 4, 기존 데이터와
flip + mild blur를 유지한다. scale LR 5e-6, points/normal/mask 1.5e-6,
neck 5e-7이며, 200회 warmup → 1,000회까지 유지 → 3,000회까지 cosine으로
시작 LR의 20%까지 낮춘다. 최초 공통 모델을 한 번 평가한 뒤 네 실험을
표 순서대로 실행한다. Large 평가/학습은 이 sweep에 포함하지 않는다.
12블록을 열어도 patch embedding, 최종 norm, encoder output projection은
기존 코드대로 동결된다. 블록 수 증가에 따른 VRAM은 실제 smoke로 확인한다.
전체 validation은 250회마다 수행하며 이미지별 JSON/CSV는 저장하지 않는다.
고정 블러 별도 평가는 이 sweep에 자동 포함되어 있지 않다.

```bash
bash run_script/run_gaemi_t400_backbone_sweep_3000.sh --dry-run
bash run_script/run_gaemi_t400_backbone_sweep_3000.sh --smoke
bash run_script/run_gaemi_t400_backbone_sweep_3000.sh
```
