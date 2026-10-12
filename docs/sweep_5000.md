# E 설정 5000-step / conv neck 학습 비교

설정: `configs/experiments/gaemi_finetune_sweep_5000.json`
실행: `run_script/run_gaemi_finetune_sweep_5000.sh`

두 실험 모두 **같은 원본 pretrained에서 총 5000 optimizer step** 학습합니다. 이전 E의 1499-step 모델을 이어서 학습하거나 5000 step을 추가하는 실행이 아닙니다. 새 timestamp workspace를 사용합니다.

| 실험 | scale LR | points LR | neck LR | normal / mask LR | forward × accumulation (GPU 1개) |
| --- | --- | --- | --- | --- | --- |
| `e_heads_5000` | 1e-5 | 3e-6 | 동결 | 동결 | 8 × 4 |
| `e_neck_all_heads_5000` | 1e-5 | 3e-6 | 1e-6 | 각각 3e-6 | 8 × 4 |

공통: encoder 전체 동결, augmentation 없음, warmup 200 후 일정 LR, effective batch 32, seed 0, mixed BF16, gradient checkpointing, EMA 사용. 손실 가중치, 입력 데이터와 모델 구조는 기존 E와 동일합니다. 현재 config의 `low_resolution_training_steps=10000`이므로 5000 step 전체가 1200-token 구간입니다. Scheduler multiplier 개수는 실험별 optimizer group 개수와 일치하도록 생성합니다.

## Normal / mask 검토

실제 MoGe-2 forward 경로는 다음과 같습니다.

- Encoder feature → shared conv neck → points / normal / mask heads.
- Encoder CLS token → scale MLP head (neck을 거치지 않음).

neck을 업데이트하면 frozen normal / mask head를 거쳐도 출력이 바뀝니다. 따라서 neck 비교 실험은 두 head도 함께 업데이트하도록 설정했습니다. 기존의 `normal_map_loss` 0.1과 `mask_bce_loss` 0.1을 유지합니다. GT normal은 깊이와 intrinsics에서 계산되며, mask는 유효 깊이와 알려진 무한 깊이 영역으로 감독합니다. 각 loss가 없는 설정은 sweep 준비 단계에서 거부합니다.

출력 텐서 형식은 변경하지 않습니다. normal은 이미 단위 벡터로 normalize되며, mask는 sigmoid 확률입니다. 추론 시 mask는 0.5로 이진화되고 focal / shift 복원에도 사용됩니다. `apply_mask=False`는 최종 points/depth를 mask로 지우지 않는 옵션이며, 복원 과정에서 mask를 무시하는 옵션이 아닙니다.

기존 E에서 frozen normal / mask 분기의 loss는 두 학습 head에 gradient를 주지 않았지만, neck을 풀면 해당 loss도 shared neck에 gradient를 전달합니다. 따라서 이번 비교는 **neck과 모든 출력 head의 공동 적응 효과**를 평가합니다. neck 하나만의 효과를 분리하는 실험은 아닙니다.

Best checkpoint 선택 기준은 계속 `depth_metric/rel`입니다. 현재 validation은 normal 각도와 mask IoU를 따로 보고하지 않으므로, 깊이 성능 개선을 normal / mask 품질 개선으로 단정하면 안 됩니다. 두 head의 학습 loss도 TensorBoard에서 함께 확인하세요.

## 실행

원격 PC의 MoGe 저장소에서 가상환경을 활성화한 뒤 먼저 설정을 확인합니다.

```bash
source .venv-moge/bin/activate
bash run_script/run_gaemi_finetune_sweep_5000.sh --dry-run
```

학습 메모리 점검은 두 조건을 각각 10 step 실행합니다. 실제 loss, accumulation, EMA를 사용하지만 validation과 checkpoint/PLY 저장은 생략합니다. 별도 smoke workspace에 저장되며 본 실험은 새로 시작합니다.

```bash
bash run_script/run_gaemi_finetune_sweep_5000.sh --smoke
```

본 실행은 공통 pretrained baseline을 한 번 평가하고 두 실험을 순서대로 수행합니다.

```bash
bash run_script/run_gaemi_finetune_sweep_5000.sh
```

기본 pretrained: `/home/coder/MoGe/pretrained/moge-2-vits.pt`
기본 데이터: `/home/coder/dataset`

경로를 바꾸려면 `--pretrained /path/model.pt --data-root /path/dataset`을 사용합니다. `PYTHON_BIN` 환경변수로 Python 실행 파일을 지정할 수 있습니다. `--help`로 옵션을 확인하세요.

## 배치와 메모리

대상 GPU는 이전에 확인한 약 23.6 GiB MIG입니다. 두 실험 모두 기본 `8×4`입니다. 사용자가 해당 GPU에서 수행한 10-step smoke에서 E는 peak allocated / reserved 4.69 / 6.27 GiB, neck 포함 모델은 6.86 / 9.01 GiB로 정상 완료했습니다. encoder는 여전히 동결되지만 neck과 normal / mask 분기의 역전파 활성값, gradient, Adam 상태가 추가됩니다. mixed BF16 옵션에서도 현재 구현은 encoder만 혼합 정밀도로 동작하고 neck / heads는 FP32이므로, trainable parameter 수만으로 메모리 여유를 추정하면 안 됩니다. Gradient checkpointing은 neck과 conv heads에도 적용됩니다.

위 수치는 사용자 제공 smoke 로그 기준이며, 전체 학습 및 validation의 최대 메모리를 보장하는 값은 아닙니다. 각 workspace의 `training_memory.json`과 `summary.csv`에 실제 PyTorch training peak allocated / reserved GiB를 기록합니다. 이 값은 CUDA 전체 프로세스 사용량과 같지 않으며 validation 피크도 포함하지 않습니다. Reserved 값에는 이전 validation에서 확보한 allocator cache가 남을 수 있습니다.

neck 조건에서 OOM이면 forward를 1로 낮춥니다. accumulation은 effective batch 32에 맞춰 자동으로 32가 됩니다.

```bash
bash run_script/run_gaemi_finetune_sweep_5000.sh \
  --smoke --experiment e_neck_all_heads_5000 --batch-size 1

bash run_script/run_gaemi_finetune_sweep_5000.sh \
  --experiment e_neck_all_heads_5000 --batch-size 1
```

기본적으로 두 실험 모두 같은 microbatch 8을 사용합니다. 메모리를 줄이려면 `--batch-size 2`로 모두 `2×16`으로 바꿀 수 있습니다. Effective batch가 같아도 microbatch 차이로 연산 순서와 확률적 loss 샘플링이 달라질 수 있습니다. Gradient accumulation 횟수를 줄이는 것만으로 속도가 빨라진다고 보장하지 않습니다.

Validation은 별도 batch 8, 매 250 step, 전체 `.val.txt`, `resolution_level=0`, 고정 8개 샘플의 PLY를 사용합니다. 두 실험과 baseline에 동일 설정을 적용합니다. 이전 batch 16보다 메모리 여유를 두었습니다. `--validation-batch-size 4` 등으로 조정할 수 있습니다. Smoke에는 validation이 없으므로 실제 validation 메모리는 본 실행의 baseline 평가에서 별도로 확인됩니다.

## 저장 결과

`workspace/gaemi_finetune_sweep_5000/<timestamp>_train/` 아래에 저장합니다.

- `manifest.json`, `invocation.json`, `validation_config.json`: 실험 입력과 실행 옵션.
- 실험별 `train_config.json`, `command.json`, `run.log`, `training_memory.json`.
- 실험별 `checkpoint/best.pt`, validation 결과 JSON 및 GT/pred PLY.
- 매 1000 step 영구 checkpoint, 매 500 step rolling checkpoint.
- 공통 `tensorboard/`, `summary.csv`, `summary.md`.

실험 실패 시 다음 실험을 계속 실행하고 최종 종료 코드는 실패를 반영합니다. Dry-run 결과는 `dry_run`으로 표기하며 성공한 학습 결과로 표시하지 않습니다. 기존 결과를 덮어쓰지 않도록 `--root`로 지정한 폴더는 비어 있어야 합니다. 이 스크립트는 자동 resume를 하지 않습니다.
