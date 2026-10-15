# GT / Prediction PLY viewer

저장소 루트에서 실행합니다. Python 3.9 이상과 WebGL을 지원하는 브라우저가 필요하며, torch나 npm 패키지, 외부 CDN 연결은 필요하지 않습니다.

```bash
./run.sh
# http://127.0.0.1:8765
```

기본 검색 경로는 저장소의 `workspace`입니다. 특정 실험이나 sweep 전체를 지정할 수도 있습니다.

```bash
./run.sh --root workspace/gaemi_finetune_sweep_1500/20261011_034445 --port 8765
```

`gt.ply`와 `pred.ply`가 함께 있는 폴더를 재귀적으로 찾아 목록에 표시합니다. 실험명 / step / 샘플 번호로 검색한 뒤 선택하세요. 같은 폴더의 `image.png`, `info.json`도 표시합니다. 학습 중 생성된 결과는 새로고침으로 다시 검색합니다. 파일을 서버에 업로드하지 않고 로컬 GT / prediction PLY 두 개를 브라우저에서 직접 여는 기능도 있습니다.

`gt.ply`와 `pred.ply`를 함께 화면에 드래그하면 파일명으로 역할을 구분해 바로 엽니다. 역할을 알 수 없는 이름 두 개는 전달된 순서대로 GT / prediction에 배정하며 실제 배정을 화면에 표시합니다. 명확하게 지정하려면 각각 GT / Prediction 드롭 영역에 한 개씩 놓으세요. 한 개만 놓아도 즉시 표시되며, 다음 파일을 추가해 비교할 수 있습니다. 기존 점군을 교체할 때도 해당 영역에 드롭하면 됩니다.

원격 작업소에서는 포트 8765를 전달해서 접속합니다. SSH를 사용하는 경우 로컬 PC에서 다음 명령을 실행하고 브라우저로 `http://localhost:8765`를 엽니다.

```bash
ssh -L 8765:127.0.0.1:8765 coder@REMOTE_HOST
```

환경에서 외부 인터페이스 바인딩이 필요하면 `./run.sh --host 0.0.0.0`을 사용하세요. 서버에는 인증 기능이 없으므로 접근 가능한 사용자에게 지정한 root의 결과 파일이 공개됩니다. 서버는 읽기 전용입니다. `PYTHON_BIN`으로 Python 실행 파일을 지정할 수 있습니다.

- 나란히 보기: 두 화면에서 같은 카메라, 중심, 축척을 사용합니다.
- 겹쳐 보기: GT 청록, prediction 주황. 각 점군을 개별적으로 숨길 수 있습니다.
- 드래그는 가상 구면을 굴리는 트랙볼 회전입니다. 고정된 위쪽 방향이나 각도 제한 없이 회전하며, 화면 가장자리에서 드래그하면 롤도 가능합니다. Shift+드래그, 가운데 버튼 또는 우클릭 드래그는 이동, 휠은 확대/축소입니다. 휠 감도는 기존의 1/5이며, Shift+휠은 일반 휠보다 4배 세밀하게 조절합니다. 두 화면의 조작은 동기화됩니다.
- 카메라 XYZ축은 원본 핀홀 원점 `(0, 0, 0)`에 표시합니다. X는 빨강(오른쪽), Y는 초록(아래), Z는 파랑(전방 광축)입니다. 시점을 조작해도 원본 카메라 축은 점군과 같은 좌표계에 고정됩니다. 축은 가독성을 위해 점군 위에 그리며 표시 여부와 길이를 조절할 수 있습니다. 축 길이의 단위는 원본 PLY와 같습니다.
- 전체 맞춤은 현재 표시할 두 점군의 공통 범위에 맞춥니다. 축 표시가 켜져 있으면 카메라 원점과 축 끝도 범위에 포함합니다. 정면은 회전을 초기화합니다. 정면에서는 Z축을 광축 방향으로 바라보므로 짧게 보이며, 회전하면 확인하기 쉽습니다.
- 최대 깊이는 원본 PLY의 Z 값 기준입니다. 빈 값 또는 0은 제한 없음입니다. 먼 배경 때문에 장면이 작아 보이면 깊이를 제한하고 전체 맞춤을 누르세요.
- 원본 scale / shift / intrinsics 차이를 유지합니다. 화면 좌표 방향만 OpenCV의 Y-down/Z-forward에서 Y-up으로 바꾸며 두 점군에 동일하게 적용합니다. ICP나 스케일 정렬은 하지 않습니다.
- ASCII, binary little-endian, binary big-endian PLY를 지원합니다. 첫 element가 vertex이고 scalar x/y/z가 있어야 합니다. RGB가 없으면 회색이며 face는 표시하지 않습니다. 기존 validation exporter의 PLY는 그대로 지원합니다.
- 점군당 최대 100만 점을 균등 간격으로 표시합니다. 비유한 좌표는 제외합니다. 원본 파일은 변경하지 않습니다. 로컬 파일 입력은 파일당 256 MiB까지 지원합니다.
- 겹쳐 보기는 시각적 비교입니다. 두 PLY의 점 개수가 달라질 수 있어 점별 오차를 계산하지 않습니다.

개발 검증:

```bash
python3 -m unittest discover -s tests -p test_ply_viewer.py
node tests/test_ply_parser.mjs
node tests/test_trackball.mjs
```

## 이미지별 validation 오차 분석

`./run.sh` 실행 후 **http://localhost:8765/analysis.html** 을 엽니다. PLY 화면 상단에도 링크가 있습니다. 웹 화면은 추가 패키지 없이 실행되며 평가 명령은 MoGe 학습 환경에서 실행해야 합니다.

- 결과 A를 선택하고 불러오면 REL 등의 히스토그램, 이미지별 분포, 평균/P50/P90을 확인합니다. 학습 objective loss가 아니라 기존 validation에서 계산한 **이미지별 평가 지표**입니다.
- 히스토그램 막대를 클릭하면 해당 구간의 목록으로 좁힙니다. 점이나 목록 행을 선택하면 해당 평가 입력 RGB와 전체 지표가 표시됩니다.
- Town / 환경 / 카메라 / 경로로 필터링하고 오차 순으로 정렬합니다. 환경은 gaemi 경로의 `__day`, `__night`, `__indoor-proxy`에서 추정하며 그 외에는 unknown입니다.
- 선택적으로 결과 B를 지정하면 `dataset + filename`이 같은 이미지의 교집합을 비교합니다. B−A와 악화 순 정렬을 지원합니다. REL은 작을수록, δ1은 클수록, signed deviation은 0에 가까울수록 좋습니다. 평균/P50/P90은 메타데이터·검색 필터 전체 기준이며 히스토그램 구간 선택은 목록에 적용합니다.
- 현재 목록을 CSV로 내려받을 수 있습니다. 각 보고서의 전체 CSV도 JSON 옆에 저장됩니다.

gaemi validation 설정에 `"per_image_metrics": true`를 추가했습니다. **수정된 설정을 읽고 시작하는 학습**은 각 validation에서 다음 파일을 저장합니다. 이미 실행 중인 프로세스나 이전에 복사된 설정에는 자동 적용되지 않습니다.

```text
<experiment>/validation/per_image_step_XXXXXXXX.json
<experiment>/validation/per_image_step_XXXXXXXX.csv
<experiment>/validation/sample_images/<content_hash>.jpg
```

RGB는 내용 해시로 중복 저장을 피합니다. 다른 PC로 옮길 때에는 `sample_images`도 함께 복사하세요. 이 기능은 기존 validation 평균이나 best checkpoint 선정 기준을 변경하지 않습니다.

이미 끝난 실험은 전체 평균만으로 이미지별 값을 복원할 수 없으므로 **저장된 checkpoint로 validation만 다시 실행**합니다. 학습이나 optimizer 갱신은 하지 않습니다.

```bash
python -m moge.scripts.analyze_validation \
  --checkpoint /path/to/checkpoint/best.pt \
  --config configs/validation/gaemi_moge2_vits_sweep_5000.json \
  --data-root /home/coder/dataset \
  --output workspace/validation_analysis/neck_best \
  --batch-size 8

./run.sh
```

`--output`은 새 빈 폴더를 지정합니다. 다른 checkpoint를 별도 폴더에 평가하면 A/B 비교가 가능합니다. `--max-samples 32`는 데이터셋별 첫 32장으로 기능만 빠르게 확인하는 옵션이며 전체 성능 판단에는 사용하지 마세요. 기본값은 전체 validation입니다. `summary.json`에 checkpoint 경로, 평가 설정, 전체 평균을 함께 저장합니다. 단독 평가의 step은 0으로 표시됩니다.

비교 시에는 동일한 데이터·split·해상도·mask·모델 가중치 종류(EMA 여부)를 사용하세요. 화면은 설정 signature가 다르면 경고하지만 데이터 내용이나 checkpoint의 EMA 여부까지 자동 판별하지는 않습니다. 현재 미리보기는 RGB이며, 모든 이미지의 GT/prediction depth나 PLY를 추가 생성하지 않습니다.

### Map·구간 태그로 비교하기

분석 화면의 **태그 추가 / 관리**에서 이름과 조건을 지정합니다. 기존 이미지별 보고서에도 바로 사용할 수 있으며 재평가가 필요하지 않습니다.

- **Map 전체**: Map에서 `Town05` 등을 선택하고 scene/번호는 비운 뒤 이름을 입력하고 `조건으로 태그 추가`를 누릅니다. Map은 보고서의 `map_file`, `map`, `town` 순서로 사용합니다. 현재 gaemi 보고서는 Town 기준이며 별도 지도 파일의 내용을 읽지는 않습니다. 데이터 경로로 묶으려면 경로 접두사를 지정합니다.
- **주행 구간**: 이미지를 선택하고 `선택 이미지의 조건 채우기`를 누르면 dataset/map/scene과 샘플 번호가 입력됩니다. 시작/끝 번호를 바꾸고 `교차로`, `터널` 같은 이름으로 추가하세요. 예를 들어 `samples/000030`부터 `samples/000050`까지 양 끝을 포함합니다. 서로 다른 scene의 같은 번호가 섞이지 않도록 번호 구간은 scene 지정이 필수입니다. 카메라를 비우면 해당 구간의 모든 카메라를 포함합니다. 픽셀 영역이나 지도상의 공간 좌표를 지정하는 기능은 아닙니다.
- **현재 선택 목록**: 검색·카메라·히스토그램 구간 등을 고른 후 `현재 목록 전체에 태그`를 누르면 페이지 제한 없이 현재 표시 대상 이미지 전체를 고정 목록으로 저장합니다. 오차가 큰 구간을 골라 다른 checkpoint에서도 같은 이미지들을 비교할 때 유용합니다.

그룹 비교 표에서 태그/Map/Scene별 이미지 수, A 평균·P50·P90, B 평균·P90, 평균 차이를 확인합니다. 태그 행을 누르면 그 태그의 히스토그램과 이미지 목록이 표시됩니다. 그룹 표는 검색·메타데이터 필터를 반영하지만 태그 필터와 히스토그램 선택은 제외하여 다른 그룹을 함께 비교할 수 있습니다. A/B를 불러온 경우 공통 이미지에 대해서만 계산합니다.

같은 태그 이름의 여러 규칙은 합집합이며 동일 이미지를 한 번만 집계합니다. 다른 태그끼리는 겹칠 수 있어 이미지 수를 모두 합하면 전체보다 커질 수 있습니다. 표의 수치는 기존과 동일한 이미지별 검증 지표이며 학습 objective loss는 아닙니다. 현재 목록 CSV에는 태그도 포함하고 그룹 통계도 CSV로 내려받을 수 있습니다.

태그는 **브라우저 localStorage**에 자동 저장됩니다(접속 주소/포트별 저장). 같은 `dataset + filename`에 대해 실험/step 간 재사용하며 원본 보고서는 수정하지 않습니다. 브라우저 데이터를 삭제하거나 PC/주소를 바꾸기 전 `태그 JSON 내보내기`로 백업하세요. `가져오기`는 현재 태그 규칙을 교체합니다. 다른 사람이 같은 서버에 접속해도 태그는 자동 공유되지 않으므로 JSON을 전달하세요. 저장 공간이 부족하면 오류를 표시하며 기존 저장 규칙은 유지합니다.

### 경로별 지표 저장 스크립트와 외부 파일 불러오기

학습 환경(venv)을 활성화한 후 실행합니다. `--dataset-path`는 `--data-root`와 같은 옵션입니다. 기본 gaemi validation 설정의 크기/깊이 단위를 사용하며 다른 데이터 포맷은 `--config`로 평가 설정을 지정하세요.

```bash
bash run_script/export_validation_loss.sh \
  --checkpoint /home/coder/MoGe/workspace/실험/checkpoint/best.pt \
  --dataset-path /home/coder/dataset \
  --split .val.txt \
  --output workspace/validation_loss/neck_best \
  --batch-size 8
```

새 빈 output 폴더를 사용합니다. `--split .train.txt`로 같은 checkpoint의 train 데이터 오차도 평가할 수 있습니다(학습 업데이트 없음). `--max-samples 32`로 먼저 입출력을 점검할 수 있습니다. 이미지별 평가 지표는 다음 파일에 저장됩니다.

```text
workspace/validation_loss/neck_best/validation/per_image_step_00000000.json
workspace/validation_loss/neck_best/validation/per_image_step_00000000.csv
```

각 레코드에는 `dataset`, `filename`(데이터셋 상대 샘플 경로), `dataset_root`, `data_path`(절대 샘플 경로), `image_path`(상대 image.jpg 경로), 지표가 기록됩니다. JSON에는 checkpoint와 평가 설정/전체 평균도 함께 기록합니다. CSV는 경로별 표 분석용이고 viewer 직접 불러오기는 JSON을 사용합니다. 저장되는 값은 REL·δ1 등의 검증 지표이며 학습 objective loss는 아닙니다.

데이터셋과 결과 폴더가 서로 다른 위치에 있어도 아래처럼 실행할 수 있습니다.

```bash
./run.sh --root workspace --data-root /home/coder/dataset
# 로컬 PC에 데이터가 있는 경우
./run.sh --root workspace --data-root /home/jeon/gaemi_simulations/datasets/mono-depth
```

1. `http://localhost:8765/analysis.html`에서 **A 저장된 validation JSON 불러오기**로 위 JSON 파일을 선택합니다. 비교할 두 번째 파일은 B에 지정합니다. 파일은 브라우저에서만 읽습니다.
2. **A/B 데이터셋 경로**에서 원본 데이터가 있는 경로를 선택합니다. 등록한 경로가 하나이면 JSON 불러오기 시 자동 선택합니다. 여러 경로는 `--data-root`를 반복해서 등록합니다.
3. 다른 PC에서 생성한 JSON도 상대 `filename` 구조가 유지되면 현재 데이터셋 경로로 연결됩니다. JSON의 이전 절대 경로를 수정할 필요가 없습니다. 데이터 내용과 split의 일치 여부는 확인해야 합니다.

경로는 **viewer 서버의 파일시스템 기준**입니다. 서버는 명시적으로 등록한 데이터셋 루트 안의 `image.jpg`만 제공합니다. 데이터셋을 연결하면 원본 RGB를 표시하므로 평가 입력의 resize/crop 미리보기와 다를 수 있으며 화면에 구분해 표시합니다. 경로를 비우면 기존 서버 보고서의 저장된 미리보기를 사용합니다. 브라우저에서 직접 불러온 JSON은 인접 `sample_images` 폴더를 읽을 수 없으므로 데이터셋 연결이 필요합니다. 직접 선택한 JSON은 새로고침 후 다시 선택해야 하며 태그는 기존처럼 브라우저에 유지됩니다.
