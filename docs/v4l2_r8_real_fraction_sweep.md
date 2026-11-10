# r8 V4L2 비율 sweep (2026-11-10)

## 직전 실험 결과

모든 모델은 `best_flip_mild_t400_total18500.pt`에서 시작했다. 기존 r8의 배치 샘플링은 CARLA 75%, V4L2 25%였고 총 3,000 step, 유효 배치 32, LoRA rank 8, Teacher depth·mask·normal 손실을 사용했다.

| r8 시점 | CARLA depth metric REL ↓ | CARLA δ1 ↑ |
|---|---:|---:|
| 시작 | 0.262673 | 0.705911 |
| 249 step (CARLA best) | 0.261232 | 0.710709 |
| 2999 step (final) | 0.267126 | 0.707393 |

별도 촬영 `2026_11_06_17_07_41`의 `front_left` 57장에서 r8 final의 Large Teacher 대비 depth REL은 0.234407, δ1은 0.678535, normal 평균 각도 차이는 11.292°, Teacher 무효 mask 검출률은 0.072124였다. 이 데이터에는 카메라에 정렬된 실제 깊이 정답이 없다. 첫 V4L2 252장은 학습 데이터이며 최근 촬영 57장은 test 전용이다.

| 모델 | CARLA final REL ↓ | 새 촬영 Teacher 대비 depth REL ↓ | normal 각도 차이 ↓ | Teacher 무효 mask 검출률 ↑ |
|---|---:|---:|---:|---:|
| LoRA r4 final | 0.263843 | 0.245854 | 11.314° | 8.07% |
| LoRA r8 final | 0.267126 | 0.234407 | 11.292° | 7.21% |
| direct_4 final | 0.268744 | 0.235361 | 11.289° | 7.55% |
| LoRA r4_only final | 0.267900 | 0.223021 | 11.560° | 0% |
| LoRA r4_scale final | 0.264910 | 0.245748 | 11.598° | 0% |

r4_only는 CARLA best가 249 step의 0.254859였지만 final은 0.267900이었다. 소수 실영상 샘플의 반복 노출, 25% 혼합 비율, 학습 길이의 영향을 분리하기 위해 이번 sweep은 r8의 혼합 비율만 바꾼다. 표의 실영상 수치는 전부 Teacher 일치도이며 실제 깊이 정확도가 아니다.

## 새 sweep

기존 25% 조건을 반복하지 않는다. 동일한 시작 Student와 나머지 설정으로 V4L2 샘플링 비율 10%, 20%를 각각 실행한다. 두 조건 모두 CARLA GT `depth_metric/rel`로 best를 고른다. 0 step 및 매 250 step의 같은 validation에서 최근 `front_left` 57장에 대한 Teacher 대비 depth·normal·mask 지표를 별도로 기록한다. `mean`은 CARLA GT만 집계하며 실영상 지표는 `metrics.v4l2_front_left_teacher`에 둔다.

같은 시점에 V4L2 학습셋 252장 전체의 Teacher 깊이 대비 **metric Abs Rel**(`mean(|prediction_m - teacher_m| / teacher_m)`)과 MAE(m)도 계산한다. 배치별 학습 손실과 구분되는 고정 학습셋 측정값이다. `metrics.v4l2_train_teacher`와 `validation/v4l2_train_curve.csv`에 저장하며 로그에는 `V4L2 TRAIN Teacher metric Abs Rel=...`이 출력된다. REL은 미터 단위 깊이에 대해 계산한 **무단위 상대 오차**이고, MAE만 미터 단위다. Teacher pseudo label과의 차이이므로 실제 정확도는 아니다. 평가에는 학습 영상의 원본 이미지를 사용하고 학습 중 증강을 적용하지 않는다.

비교 기준으로 기존 r8 final을 같은 고정 학습셋 252장에 평가하면 REL **0.126420**, MAE **1.902 m**이다. 새 sweep의 0 step 및 후속 step은 각 실험의 고정 학습셋 곡선으로 비교한다.

## 평가 데이터 준비

이 저장소에서 이미 만들어 둔 `workspace/evaluation/v4l2_holdout_20261106_front_left_reference`는 Teacher 체크포인트 SHA-256 `7e3db564696c17abeed2e4b7570291b97ffc2c8906c0220582024165e1d4228e`로 생성한 57장 캐시다. 재생성이 필요하면 다음 명령을 사용한다.

```bash
.venv-eval/bin/python tools/prepare_v4l2_teacher_reference.py \
  --rgb-root workspace/evaluation/holdout_20261106_front_left/rgb \
  --teacher /home/jeon/Downloads/best.pt \
  --output workspace/evaluation/v4l2_holdout_20261106_front_left_reference
```

원격 학습 서버의 `/home/coder/dataset/v4l2_holdout_20261106_front_left`에 이 디렉터리의 **내용 전체**를 복사한다. 학습 데이터는 `/home/coder/dataset/v4l2_pseudo_v3`를 사용한다. 두 세션이 겹치면 실행 전 검사가 중단된다.

## 실행

서버의 `~/MoGe`에서:

```bash
source .venv-moge/bin/activate
python run_script/run_v4l2_r8_real_fraction_sweep.py \
  --devices 'MIG-7a18dd2c-6193-55ed-b630-3ca8839362d7' 'MIG-7623a7a2-4d3a-5bfa-adf1-8edb3e8f1fbd' \
  --data-root /home/coder/dataset \
  --pseudo-root /home/coder/dataset/v4l2_pseudo_v3 \
  --real-validation-root /home/coder/dataset/v4l2_holdout_20261106_front_left \
  --student ~/MoGe/pretrained/best_flip_mild_t400_total18500.pt \
  --batch-size 8 \
  --port-base 29600
```

각 실험의 `validation/step_XXXXXXXX.json`에서 `metrics.gaemi_sim_val`은 CARLA GT, `metrics.v4l2_train_teacher`는 학습 영상의 Teacher 대비 오차, `metrics.v4l2_front_left_teacher`는 독립 촬영의 Teacher 대비 오차다. `validation/v4l2_train_curve.csv`로 세 곡선을 step별로 나란히 확인한다. `validation/best.json`과 `checkpoint/best.pt`는 CARLA GT REL로 선정된다. 전체 `summary.csv`에도 시작·CARLA best·마지막 시점의 V4L2 train REL이 기록된다. 실영상 Teacher 일치도는 실제 정확도가 아니다.
