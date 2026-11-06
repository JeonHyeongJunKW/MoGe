# v4l2 실제 영상으로 Student 추가 학습

Teacher가 실제 RGB의 metric depth를 미리 생성하고, Student는 CARLA GT와
실제 pseudo depth를 섞어 학습한다. Teacher는 고정이며 학습 GPU에 함께 올라가지 않는다.
기존 CARLA 데이터나 체크포인트는 덮어쓰지 않는다.

## 사용할 체크포인트

저장소 루트 `pretrained/`에 아래 파일을 둔다.

| 역할 | 기본 파일 | 추론/학습 토큰 |
| --- | --- | --- |
| Student 시작 가중치 | `best_flip_mild_t400_total18500.pt` | 400 |
| 고정 Teacher | `best_teacher_vitl_t1200_last12_total5000.pt` | 1,200 |

Student는 18,500회 학습 best에서 **3,000회 추가 학습**한다. 최종 실행은
기존 optimizer/scheduler/EMA를 이어받는 resume가 아니라 가중치만 불러오는 새 실험이다.
checkpoint에 저장된 모델 구조를 읽으며 ViT-Small과 model-only 형식을 확인한다.
`--student`로 다른 Small best를 사용하면 이전 학습량은 그 파일의 이력에 따른다.

## 이미 만든 데이터셋과 서버 배치 위치

이번 로컬 작업에서 만든 파일:

- 원본: `/home/jeon/real_dataset/2026_11_03_19_35_51/`
- 보정 RGB: `/home/jeon/real_dataset/v4l2_rectified_v1/`
- Teacher pseudo depth 포함: `/home/jeon/real_dataset/v4l2_pseudo_v1/`
- 전달용 압축 파일: `/home/jeon/real_dataset/v4l2_pseudo_v1.tar.gz`

`left`, `front_left`, `right`를 각 카메라의 CameraInfo로 보정하고, 타임스탬프
0.5초 간격으로 추출했다. 각 84장, 합계 252장이다. 전부 train이다.
Teacher는 로컬에 있는 `vitl_t1200_last12_lr5e-7_add3000/best.pt`를 사용했다.
이는 위 서버 파일명의 근거가 된 누적 5,000회 Large best이며 기록된 REL은
0.2183992478이다. 실제 파일 SHA-256은 데이터셋 `generation.json`에 저장했다.
서버 파일과 동일한 바이트인지 확인하려면 `sha256sum pretrained/best_teacher_vitl_t1200_last12_total5000.pt`
결과를 `generation.json`의 `settings.teacher_sha256`과 비교한다.
로컬 생성은 검토했던 실제 영상 추론과 동일하게 `--fp16`, 1,200토큰,
FOV override 없이 수행했다. 카메라 K는 depth의 학습용 3D 복원에 사용한다.

학습 서버에서는 **완성된 pseudo 데이터셋 폴더 전체**를 다음처럼 둔다.
`samples`만 옮기면 안 된다. 숨김 index 파일과 `dataset.json`도 필요하다.

```text
/home/coder/dataset/
├── 20261017_train/             # 기존 CARLA, 그대로 유지
├── near_obstacles_train/       # 기존 CARLA, 그대로 유지
└── v4l2_pseudo_v1/             # 새 실제 영상 데이터셋
    ├── .train.txt
    ├── .val.txt                # 현재 비어 있음
    ├── .test.txt               # 현재 비어 있음
    ├── .index.txt
    ├── dataset.json
    ├── generation.json
    └── samples/<session>/<camera>/<timestamp>/
        ├── image.jpg          # Teacher가 본 것과 같은 보정 RGB
        ├── depth.png          # 미터 단위 pseudo depth, MoGe log PNG
        ├── meta.json          # 정규화 intrinsics, provenance
        └── result.json        # 유효 비율 및 파일 checksum
```

예를 들어 로컬 PC에서 서버로 폴더 전체를 전송한다. 아래 서버 주소는 실제 주소로 바꾼다.

```bash
rsync -av /home/jeon/real_dataset/v4l2_pseudo_v1 \
  coder@SERVER:/home/coder/dataset/
```

또는 압축 파일을 서버로 전송하고 **아직 동일 이름의 폴더가 없는 경로**에서 푼다.

```bash
mkdir -p /home/coder/dataset
tar -xzf v4l2_pseudo_v1.tar.gz -C /home/coder/dataset
```

완성된 pseudo 데이터셋을 옮겼다면 원본 RGB/MCAP, 보정 RGB 데이터셋과 Teacher
가중치는 **Student 학습 실행에는 필요 없다**. Teacher를 바꾸어 정답을 재생성할
때에는 보정 RGB 데이터셋과 Teacher 가중치를 사용한다.

## 서버에서 실행

코드를 서버에 반영하고 기존 학습 가상환경을 활성화한 뒤 저장소 루트에서 실행한다.
학습 의존성은 기존 `train` extra와 동일하다. 데이터 준비에만 `v4l2` extra가 추가된다.

```bash
cd /home/coder/MoGe
source .venv-moge/bin/activate

# 설정·실행 명령만 생성. 데이터나 모델을 읽거나 학습하지 않는다.
python run_script/run_v4l2_distillation.py --dry-run

# 실제 모델·데이터로 10회 학습 확인. validation/주기적 저장은 생략(final은 저장).
python run_script/run_v4l2_distillation.py --smoke

# CARLA baseline 평가 후 3,000회 추가 학습
python run_script/run_v4l2_distillation.py
```

기본값은 `/home/coder/dataset`이며 경로가 다르면
`--data-root /path/to/dataset --pseudo-root /path/to/v4l2_pseudo_v1`로 지정한다.
`--batch-size 4`로 줄이면 accumulation 8로 유효 배치 32를 유지한다.
각 실행은 `workspace/v4l2_distillation/<timestamp>_<mode>/`에 별도로 저장된다.

| 항목 | 첫 실험 설정 |
| --- | --- |
| Student 입력 / 토큰 | 640×360 / 400 |
| 학습량 / 유효 배치 | 3,000회 / 32 (기본 8×4) |
| Train 샘플링 확률 | CARLA 75%, 실제 pseudo 25% |
| CARLA 내 데이터 비율 | 기존 20,817 : 근거리 9,351 |
| Pseudo loss 배율 | CARLA loss 계수의 0.25배 |
| 학습 부분 | DINOv2 마지막 4블록 + neck + scale/points/normal/mask heads |
| Backbone / neck LR | 각각 5e-7 |
| Scale / points·normal·mask LR | 5e-6 / 각각 1.5e-6 |
| 스케줄 | 200회 warmup → 종료까지 cosine, 최종 20% |
| Augmentation | 좌우 반전, mild blur (확률 0.3, kernel 3/5/7) |
| 검증 | 기존 CARLA 4,320장, 시작 시 + 250회마다 |
| Best 선택 | CARLA `depth_metric/rel` 최소값 |

샘플링은 복원추출이므로 각 미니배치가 정확히 75:25라는 뜻은 아니다.
252장의 작은 실제 세션을 반복 학습하므로 3,000회를 최적 학습량으로 보장하지 않는다.
pseudo 샘플링 비중과 loss 배율은 **별도 설정**이며 두 가지가 모두 기여도를 낮춘다.

Pseudo 데이터는 기존 points 전역·지역·edge loss와 metric scale loss만 사용한다.
Teacher의 normal/mask head 출력을 직접 증류하지 않는다. Student normal/mask는
CARLA GT에서 학습하며, 공유 특징이 바뀌기 때문에 실제 출력도 바뀔 수 있다.
Depth에서 normal을 계산하는 기존 loader 처리는 유지되지만 pseudo의 normal head loss는 없다.

저장 내용은 `train_config.json`, `validation_config.json`, `command.json`,
`resolved_command.json`, `initial_student.json`, `run.log`, `tensorboard/`,
`validation/`, `checkpoint/best.pt` 등이다. 초기 평가도 best 선택에 참여하므로
학습 결과가 시작 모델보다 나쁘면 best는 시작 모델로 남을 수 있다.

## 대조 실험

### 일반 파인튜닝과 LoRA 비교 sweep

`configs/experiments/v4l2_lora_sweep.json`에 세 조건을 정의했다.
아래 명령은 같은 Student에서 독립적으로 시작해 **순차 실행**한다.

| 실험명 | 마지막 4개 DINOv2 블록 | Backbone 계열 LR | Neck·heads |
| --- | --- | --- | --- |
| `direct_last4` | 기존 블록 파라미터 직접 학습 | 5e-7 | 기존대로 학습 |
| `lora_r4` | attention qkv·proj에 rank 4, alpha 8 | LoRA만 1e-5 | 위와 동일 |
| `lora_r8` | attention qkv·proj에 rank 8, alpha 16 | LoRA만 1e-5 | 위와 동일 |

LoRA 조건은 backbone의 원래 weight/bias/norm/MLP를 모두 고정한다.
qkv는 Q·K·V가 합쳐진 하나의 Linear이며 그 전체에 LoRA를 적용한다.
LoRA dropout은 사용하지 않고, B를 0으로 초기화해 학습 시작 출력은 기존
Student와 같게 만든다. Rank별 alpha/rank는 2로 통일한다.
학습률은 방법별 후보 설정이므로 이 실험은 동일 LR에서의 알고리즘 비교가 아니다.

공통 조건은 400토큰, CARLA:pseudo 75:25, pseudo loss 배율 0.25,
유효 배치 32, 3,000회, 같은 augmentation·seed·평가 데이터다.
Teacher pseudo depth는 기존 `v4l2_pseudo_v1`을 그대로 사용한다.
각 조건의 optimizer/scheduler/EMA는 새로 시작하고, 다른 조건의 결과를 이어받지 않는다.

```bash
cd /home/coder/MoGe
source .venv-moge/bin/activate

# 세 조건의 설정과 명령 생성만 수행
python run_script/run_v4l2_lora_sweep.py --dry-run

# 각 조건 10회씩 실제 학습 확인. validation 생략, final 체크포인트는 저장.
python run_script/run_v4l2_lora_sweep.py --smoke

# 일반 학습 → LoRA rank 4 → LoRA rank 8, 각각 3,000회
python run_script/run_v4l2_lora_sweep.py

# 특정 조건만 실행
python run_script/run_v4l2_lora_sweep.py --experiment lora_r4

# 메모리가 부족하면 세 조건 모두 동일 배치로 축소 (4×8=32)
python run_script/run_v4l2_lora_sweep.py --batch-size 4
```

출력: `workspace/v4l2_lora_sweep/<timestamp>_<mode>/`

```text
summary.csv / summary.json
manifest.json / invocation.json
direct_last4/   # train_config, logs, validation, checkpoint/best.pt
lora_r4/
lora_r8/
```

요약에는 초기 REL, best REL·δ1, best step, 학습 파라미터 수,
LoRA 파라미터 수, GPU peak allocated/reserved가 저장된다.
`dry_run`, `smoke_ok`, `ok`, `failed(...)` 상태를 구분한다.
실패한 조건이 있어도 남은 조건을 실행하고 최종 종료 코드는 실패를 반환한다.
기존 출력 폴더를 덮어쓰지 않으며 `--root`는 새 폴더로 지정한다.

LoRA는 backbone 학습량만 줄이고 neck·heads 학습을 유지한다.
따라서 전체 학습 메모리가 rank에 비례해서 줄어드는 것은 아니다.
CARLA 지표는 기존 성능 유지 여부를 보고, 실제 적응 효과는 별도 실제 세션의
Teacher/기존 Student/학습 Student 화면과 거리 기준을 함께 확인한다.

### LoRA 모델 사용과 일반 모델로 병합

LoRA `best.pt`는 원래 weight, 학습된 decoder, LoRA 파라미터, LoRA 설정을
모두 저장한 **완전한 모델 파일**이다. 작은 adapter-only 파일은 아니다.
수정된 MoGe 코드의 `MoGeModel.from_pretrained()`로 그대로 추론할 수 있다.
설정과 adapter가 불일치하면 로드를 거부한다. 원래 코드/배포 경로에서는
먼저 일반 Linear 가중치로 병합한다.

```bash
python tools/merge_moge2_lora.py \
  --checkpoint workspace/v4l2_lora_sweep/RUN/lora_r4/checkpoint/best.pt \
  --output pretrained/student400_v4l2_lora_r4_merged.pt
```

`RUN`은 실행 결과 폴더명으로 바꾼다. 병합은 `W + (alpha/rank) × B @ A`를
적용하고 adapter 텐서·설정을 제거한다. Decoder 학습 가중치는 그대로 유지한다.
기존 파일은 덮어쓰지 않으며 optimizer/EMA 상태를 이어가는 파일도 만들지 않는다.
자동 train/eval 병합은 하지 않아 EMA·재로딩 때 중복 병합되지 않는다.
배포 전에는 병합 모델의 실제 입력 출력, ONNX/TensorRT 변환 결과를 확인한다.

한 조건만 세부 조정할 때는 기존 단일 실행기에 옵션을 줄 수 있다.

```bash
python run_script/run_v4l2_distillation.py --lora-rank 4 --lora-lr 1e-5
```

### CARLA와 실제 pseudo 비율 비교

같은 Student와 학습량에서 실제 데이터의 효과를 비교한다. 각 명령은 별도 실행이다.

```bash
# CARLA만 같은 횟수 추가 학습
python run_script/run_v4l2_distillation.py --real-fraction 0

# 기본: 실제 pseudo 25%
python run_script/run_v4l2_distillation.py --real-fraction 0.25

# 별도 후보: 실제 pseudo 50%
python run_script/run_v4l2_distillation.py --real-fraction 0.5
```

현재 세션에는 독립적인 실제 GT가 없다. Teacher와의 일치는 실제 정확도 지표가 아니다.
별도 실제 촬영 세션에서 `RGB / Teacher / 기존 Student / 학습 Student`를 비교하고,
거리 기준이 있는 장면에서는 절대거리 오차도 확인한다.

## 데이터셋을 새로 만들 때

`configs/data/v4l2_real_sessions.json`에 세션을 추가한다. 같은 장소·경로의 연관
세션에는 동일한 `scene_group`을 부여하고 동일 split에 둔다. 같은 세션을 카메라별·
프레임별로 train/val에 나누지 않는다. 실제 독립 평가 세션은 `split: "test"`로
추가한다. 이 세션은 학습 index에 들어가지 않으며 CARLA validation에도 섞지 않는다.

원본 폴더는 다음 구조를 유지한다. 이미지 파일명은 nanosecond 정수 타임스탬프다.

```text
v4l2_raw/<session>/
├── camera/v4l2_driver/front_view/<camera>/color/*.bmp
└── camera/info/v4l2_driver/<camera>/*.mcap
```

MCAP decoder가 없는 환경에서는 기존 환경에 필요한 두 패키지만 추가하거나,
전체 설치를 새로 할 때 `pip install -e '.[train,v4l2]'`를 사용한다.

```bash
pip install mcap mcap-ros2-support

python tools/v4l2_distillation.py prepare \
  --source-root /home/coder/dataset/v4l2_raw \
  --output /home/coder/dataset/v4l2_rectified_v2

python tools/v4l2_distillation.py pseudo \
  --dataset /home/coder/dataset/v4l2_rectified_v2 \
  --output /home/coder/dataset/v4l2_pseudo_v2 \
  --teacher pretrained/best_teacher_vitl_t1200_last12_total5000.pt \
  --fp16

python run_script/run_v4l2_distillation.py \
  --pseudo-root /home/coder/dataset/v4l2_pseudo_v2
```

`prepare`는 각 카메라 MCAP의 모든 CameraInfo를 확인한다. 세션 중 calibration이
바뀌거나 plumb_bob 이외 모델이면 중단하며, 원본을 임의 resize하지 않는다.
기본 profile은 이전 실제 비교에 사용한 고정 G41/G45 보정 K다. 다른 로봇은
manifest의 rectification profile을 해당 로봇의 운용 전처리에 맞춰 변경한다.

Teacher는 **저장된 JPEG를 다시 읽어 추론**하므로 Student 입력과 정확히 같은
픽셀을 본다. RGB와 depth를 함께 반전하고, blur는 Student RGB에만 적용한다.
새 데이터셋 이름을 사용하며 완료된 데이터셋을 덮어쓰지 않는다. 생성 중 실패한
`.partial` 결과는 동일 명령에 `--resume`을 붙여 이어갈 수 있다. Teacher hash,
원본 dataset manifest hash, 생성 설정이 일치해야 한다.

Depth는 `apply_mask=False`로 추론한 뒤 Teacher mask, 보정 유효 영역,
유한값, 0.1–100m 범위의 교집합만 저장한다. 범위를 벗어난 값은 잘라 맞추지 않고
NaN(unknown)으로 제외한다. 프레임 유효 비율이 10% 미만이면 제외한다.
여기서 mask는 거리 정확도를 보증하는 confidence가 아니다.

선택적으로 `--flip-consistency-threshold 0.2`를 주면 원본과 반전 복원 depth의
`abs(a-b)/max(a,b)`가 0.2 이하인 픽셀만 유지한다. 기본값은 비활성화이며,
이 임계값은 검증된 최적값이나 confidence가 아니다.

`depth.png`는 MoGe 16-bit log encoding(0=unknown, 65535=infinity)이다.
Teacher 제외 픽셀은 전부 unknown으로 저장해 잘못된 sky supervision을 피한다.
메타데이터의 intrinsics는 OpenCV 픽셀 중심에 +0.5 보정 후 해상도로 나눈
MoGe 정규화 좌표이고, `depth_unit=1.0`으로 metric scale 학습을 켠다.

## 이번 구현 확인 범위

- 실제 3카메라 원본에서 보정 RGB 및 Teacher pseudo depth 252장 생성 완료.
- 파일 인코딩, unknown mask, camera intrinsics, 세션 분리, 변경된 Teacher의
  resume 거부, 데이터로더 연결, 혼합 loss 설정을 자동 테스트로 확인.
- 로컬 GPU에서 실제 CARLA+실제 pseudo 혼합 배치로 1회 업데이트(1×32)를 실행하고
  196개 parameter tensor 변경 및 모든 저장 가중치의 유한값을 확인.
  해당 동작 확인은 로컬 보유 Student 10,000회 best로 수행했으며 서버 기본값인
  18,500회 best의 성능 실험은 아니다. 이 실행의 peak allocated는 1.42 GiB였고,
  서버 기본 배치 8이나 validation 메모리를 대표하지 않는다.
- 서버에서 요청 모델을 사용한 3,000회 본 학습은 아직 실행하지 않았다.
- 일반 last4, LoRA rank 4, LoRA rank 8 sweep도 같은 로컬 Student와 혼합 데이터로
  각각 1회 업데이트(1×32)를 통과했다. LoRA 두 조건은 기존 encoder 가중치가
  그대로 유지되고 adapter가 갱신되며 저장 가중치가 모두 유한함을 확인했다.
  학습 파라미터는 각각 19,852,328 / 12,788,264 / 12,825,128개다.
- rank 4 실제 체크포인트를 일반 모델로 병합한 뒤 같은 RGB에서 출력도 비교했다.
  TF32와 fused attention을 끈 FP32 비교에서 최대 절대 오차는 points 약 9.54e-7,
  normal 약 2.13e-6, mask 약 1.19e-7, metric scale 0이었다.
  이는 구현 확인이며 깊이 추정 성능 개선을 입증하는 실험은 아니다.

## Mask·normal을 포함한 V4L2 v3 (2026-11-06)

`/home/jeon/real_dataset/v4l2_pseudo_v3`는 최신 Teacher
`/home/jeon/Downloads/best.pt`로 생성한 다중 출력 pseudo label이다.
기존 v2와 같은 보정 RGB 252장(카메라별 84장)을 사용하며 모두 train이다.
기존 학습에 사용된 프레임을 다시 나눠 독립 validation이라고 부르면 안 된다.
별도 촬영 세션과 장소를 확보해야 실제 데이터 일반화 성능을 확인할 수 있다.
Teacher label과의 일치도는 실제 정답에 대한 정확도가 아니다.

```bash
.venv-eval/bin/python tools/v4l2_distillation.py pseudo \
  --dataset /home/jeon/real_dataset/v4l2_rectified_v1 \
  --output /home/jeon/real_dataset/v4l2_pseudo_v3 \
  --teacher /home/jeon/Downloads/best.pt \
  --num-tokens 1200 --fp16 --include-auxiliary-labels

.venv-eval/bin/python tools/verify_v4l2_multitask.py \
  --dataset /home/jeon/real_dataset/v4l2_pseudo_v3 \
  --source /home/jeon/real_dataset/v4l2_rectified_v1
```

각 sample에 RGB, 기존 인코딩의 depth.png, meta.json과 아래 파일이 있다.

| 파일/배열 | 의미 |
| --- | --- |
| teacher_labels.npz / mask_probability | float32 H×W, Teacher mask head의 sigmoid 확률(0–1) |
| teacher_labels.npz / teacher_mask | 확률 > 0.5인 bool 영역. 의미론적 하늘 분류가 아닌 기하 유효성 예측 |
| teacher_labels.npz / geometry_mask | 양의 깊이 조건까지 반영한 Teacher infer mask |
| teacher_labels.npz / mask_valid | 보정으로 실제 입력이 존재하는 영역. Teacher 음성(무효) 영역도 포함 |
| teacher_labels.npz / normal | float32 H×W×3, Teacher normal head 출력의 단위벡터. 비정상/영벡터는 0 |
| teacher_labels.npz / normal_valid | 보정 유효 영역 ∩ geometry_mask ∩ 유한한 비영 normal |
| teacher_labels.npz / depth_valid | 기존 0.1–100m 필터를 통과한 depth 유효 영역 |
| teacher_mask.png, mask_valid.png, normal_valid.png | 해당 bool 배열의 0/255 PNG 복사본 |
| normal.png | MoGe 16-bit normal 인코딩. normal_valid 바깥은 unknown |

NPZ의 normal은 OpenCV 카메라 좌표(x 오른쪽, y 아래, z 전방)이다.
normal_valid 안에서만 손실을 계산한다. 좌우 반전 증강 시 배열의 가로축을
뒤집고 normal x 성분의 부호도 바꾼다. normal.png는 일반 RGB normal 시각화가
아니므로 `moge.utils.io.read_normal`로 읽는다. preview.png는 별도 시각화다.

Mask distillation(유효 영역 지도 학습)은 mask_valid에서 양성과 음성을 모두
포함해야 한다. depth_valid만 사용하면 하늘 같은 무효 영역을 배우지 못한다.
거리 범위 때문에 제외된 픽셀을 무조건 mask 음성으로 만들지 않는다.
Normal은 depth에서 재계산하지 않고 Teacher normal head의 예측을 저장했다.

**새 multitask sweep은 이 label의 데이터로더·증강·mask/normal loss 연결을 포함한다.**
새 format은 `moge_v4l2_multitask_v1`이며 `--multitask-labels`로 명시적으로 읽는다.
기존 depth 전용 실행 방식은 v3를 거부한다. format 문자열을 수정하지 말고
아래 새 sweep을 사용한다. Mask/normal head도 학습 대상에 포함된다.

서버 이동 시 압축 파일과 sha256 파일을 같은 폴더에 두고 다음처럼 확인/해제한다.
아래 경로는 저장 위치이며 기존 sweep 실행 명령이 아니다.

```bash
sha256sum -c v4l2_pseudo_v3.tar.gz.sha256
mkdir -p /home/coder/real_dataset
tar -xzf v4l2_pseudo_v3.tar.gz -C /home/coder/real_dataset
```

서버 데이터 경로는 `/home/coder/real_dataset/v4l2_pseudo_v3`이다.


## V4L2 mask·normal sweep

실행 파일: `run_script/run_v4l2_multitask_sweep.py`
설정: `configs/experiments/v4l2_multitask_sweep.json`

| 순서 | 실험 | Backbone 학습 방식 | 최종 mask / normal 가중치 |
| --- | --- | --- | --- |
| 1 | lora_r4 | 마지막 4개 block의 LoRA rank 4 | 0.025 / 0.025 |
| 2 | lora_r8 | 마지막 4개 block의 LoRA rank 8 | 0.025 / 0.025 |
| 3 | direct_4 | 마지막 4개 block 직접 학습 | 0.025 / 0.025 |

세 조건 모두 동일한 depth·points·metric scale + mask + normal 손실을 사용한다.

세 조건 모두 동일한 plain Student `best_flip_mild_t400_total18500.pt`에서
독립적으로 시작한다. 앞 실험의 final이나 기존 r8 1500회 모델을 이어받지 않는다.
LoRA는 rank 4/8에 각각 alpha 8/16을 사용하며 마지막 4개 block의 qkv/proj에
적용한다. LoRA LR은 1e-5, direct_4 backbone LR은 기존 설정인 5e-7이다.
따라서 학습 방식별 기존 학습률을 사용하는 비교이며 동일 학습률 비교는 아니다.
Neck와 모든 기존 출력 head도 학습한다. CARLA 75% / V4L2 25% sampling,
기존 pseudo 손실 배율 0.25, 3,000 updates, seed 0, 유효 배치 32를 사용한다.
기본 batch 8 × accumulation 4이며 메모리 부족 시 `--batch-size 4` 또는 1을
주면 누적 횟수가 조정된다. 이 가중치는 초기 비교값이며 최적값을 검증한 것은 아니다.

Mask는 Teacher sigmoid 확률을 target으로 사용하고 `mask_valid` 픽셀 수로
정규화한 BCE(이진 교차 엔트로피)를 계산한다. Teacher가 무효로 판단한 픽셀도
포함한다. Normal은 `normal_valid`의 단위벡터에 대해 `1-cos(theta)`의 평균을
사용한다. 반대 방향을 같은 normal로 취급하지 않는다. CARLA의 기존 mask/normal
손실은 세 조건 모두 유지한다. Teacher는 offline label로 고정되어 GPU에 올리지 않는다.

서버의 최신 코드와 v3 데이터셋을 준비한 뒤 실행한다.

```bash
cd ~/MoGe
source .venv-moge/bin/activate
python run_script/run_v4l2_multitask_sweep.py \
  --data-root /home/coder/dataset \
  --pseudo-root /home/coder/real_dataset/v4l2_pseudo_v3 \
  --student /home/coder/MoGe/pretrained/best_flip_mild_t400_total18500.pt \
  --batch-size 8
```

위 명령에 `--dry-run`을 추가하면 설정/명령만 생성한다. `--smoke --num-iterations 1`
을 추가하면 조건별 1 update만 실행해 배관을 점검하며 정식 성능 평가를 하지 않는다.
rank 4 조건만 먼저 실행하려면 `--experiment lora_r4`를 추가한다.
조건 선택을 반복해서 여러 개 지정할 수도 있으며 실행 순서는 manifest 순서다.
출력은 기본적으로 `workspace/v4l2_multitask_sweep/<timestamp>_train/`에 저장된다.
`--root`는 비어 있는 새 경로만 허용한다. 실패한 조건이 있으면 sweep은 중단한다.

CARLA validation은 시작 전과 250 updates마다 수행하며 best는 여전히 CARLA depth REL로
선택한다. 따라서 best.pt가 최상의 V4L2 mask/normal을 의미하지 않는다. 500 updates마다
rolling, 1000 updates마다 정기 checkpoint가 저장되며 마지막 final도 저장된다.
학습 TensorBoard에 `teacher_mask`, `teacher_normal` 손실이 기록되고 `summary.csv`에는
조건별 가중치, CARLA best REL/delta1, best step, 메모리 사용량이 정리된다.
V4L2 252장은 전부 train이며 독립 validation에는 포함하지 않는다. 새 촬영 세션이
준비되기 전에는 V4L2 출력 비교를 학습 데이터에서의 Teacher 일치도 확인으로 해석한다.

구현 검증(2026-11-06): 관련 테스트 15개와 LoRA/입력 처리/검증 회귀 테스트
21개가 통과했다. 로컬의 실제 18,500회 Student와 CARLA+v3 혼합 데이터로
이전 손실 비교 sweep(r8 depth / depth+mask / depth+mask+normal)을 각각
1 update(1×32) 실행해 final 저장, 유한한 가중치,
Teacher mask/normal loss 기록과 각 head의 조건 간 가중치 차이를 확인했다.
이 점검은 성능 개선 검증이 아니며 서버 batch 8의 메모리 요구량을 측정한 것도 아니다.
당시 결과는 `workspace/v4l2_multitask_sweep/local_smoke/`에 있다.
현재 r4/r8/direct_4 구성은 dry-run으로 세 조건의 손실, LoRA 설정, 학습률,
유효 배치와 시작 모델 경로를 확인했다. 새 구성의 본 학습은 실행하지 않았다.

서버 적용용 코드 묶음은 `/home/jeon/Downloads/v4l2_multitask_sweep_code.tar.gz`이다.
기존 LoRA 학습 코드가 있는 MoGe 저장소에 적용하며 체크포인트/데이터는 포함하지 않는다.
서버로 옮긴 압축 파일을 아래와 같이 적용한 후 위 학습 명령을 사용한다.

```bash
tar -xzf /path/to/v4l2_multitask_sweep_code.tar.gz -C /home/coder/MoGe
```
