# V4L2 LoRA r4 출력부 고정 범위 비교

기존 `lora_r4` (LoRA+neck+모든 head, CARLA best REL 0.260088)를 기준으로
아래 두 조건만 추가 학습한다. 모두 원래 18,500회 Student에서 독립 시작한다.

| 실험 | Backbone | 출력부 |
| --- | --- | --- |
| lora_r4_only | 마지막 4 block의 qkv/proj에 LoRA rank 4 | neck, 모든 head 고정 |
| lora_r4_scale | 같은 LoRA | scale head만 학습, 나머지 고정 |

CARLA 75% + V4L2 25%, depth·mask·normal Teacher pseudo 손실,
유효 배치 32, 3,000 updates, validation 250 updates 간격을 기존 r4와 동일하게 유지한다.
고정된 head에서도 손실의 기울기가 LoRA까지 전파된다. 기존 `lora_r4` best에서
이어서 학습하지 않는다. 두 조건의 결과는 기존 r4의 같은 설정 CARLA REL 및
독립 실제 영상 검증과 비교한다. 현재 V4L2 252장은 모두 train이다.

서버의 데이터는 `/home/coder/dataset/v4l2_pseudo_v3`이다. 두 MIG UUID는
`nvidia-smi -L`로 확인한다. 예시의 UUID는 이전 서버 메시지에서 확인된 값이며
서버 재설정 후 달라질 수 있다.

```bash
cd ~/MoGe
source .venv-moge/bin/activate
nvidia-smi -L
python run_script/run_v4l2_lora_freeze_sweep.py \
  --devices 'MIG-7a18dd2c-6193-55ed-b630-3ca8839362d7' 'MIG-7623a7a2-4d3a-5bfa-adf1-8edb3e8f1fbd' \
  --data-root /home/coder/dataset \
  --pseudo-root /home/coder/dataset/v4l2_pseudo_v3 \
  --student ~/MoGe/pretrained/best_flip_mild_t400_total18500.pt \
  --batch-size 8 --port-base 29700
```

첫 번째 MIG는 `lora_r4_only`, 두 번째는 `lora_r4_scale`에 배정된다.
출력은 `workspace/v4l2_lora_freeze_sweep/<timestamp>_train/` 아래 각 실험별로
저장된다. `.launch.log`, validation, checkpoint, summary.csv를 확인한다.
실행 전 `--dry-run`을 추가해 명령과 설정을 확인할 수 있다.

로컬에서 모델의 학습 가능 가중치, optimizer 그룹과 학습률 scheduler 그룹을
검증했고, 두 장치 병렬 dry-run을 통과했다. 두 조건 모두 실제 CARLA+V4L2 혼합
데이터로 1 update, 유한한 가중치 및 final 체크포인트 저장까지 확인했다.
서버의 본 학습은 실행하지 않았다.

세 final checkpoint의 V4L2 비교 영상은
`workspace/evaluation/v4l2_multitask_final_sweep_v2/`에 생성한다.
Teacher와 r4/r8/direct final의 depth·mask 확률·normal을 같은 252장 입력으로
시각화한다. Depth는 공통 0–20m 색 범위이며 20m 이상은 포화한다.
Mask는 sigmoid 확률, normal은 OpenCV 카메라 XYZ를 색으로 표현한다.
Teacher 일치도를 보는 학습 프레임 비교이므로 실제 GT 정확도로 해석하지 않는다.
