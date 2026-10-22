# MoGe2 원본 RGB CNN 경로

기존 학습 JSON의 `model`에 아래 항목을 추가하면 별도 CNN을 사용한다. 생략하거나 `enabled: false`면 기존 구조와 파라미터 이름을 유지한다.

```json
"rgb_detail": {
    "enabled": true,
    "channels": 16
}
```

CNN은 DINOv2용 리사이즈 전 RGB(augmentation 적용 후)를 받는다. Depthwise convolution과 1×1 convolution으로 입력의 1/2, 1/4, 1/8 크기 특징을 만들고, neck 출력의 마지막 세 단계에 각각 크기를 맞춰 더한다. 640×360·160토큰에서는 neck의 272×144, 136×72, 68×36 격자에 결합된다(가로×세로). Points·normal·mask head가 결합 특징을 사용하며 CLS→scale head 경로는 그대로다. 원본을 처리하는 보조 경로이며 Lite-Mono 구현은 아니다.

`channels`는 CNN 폭이다. 기본 16이며 줄이면 연산도 줄지만 정확도·지연 효과는 측정해야 한다. Fusion의 마지막 projection을 0으로 초기화하여, 기존 체크포인트로 시작할 때 출력이 기존 모델과 동일하다. 첫 업데이트는 projection을 학습하고 이후 CNN 내부로도 gradient가 전달된다.

## 학습 설정

모델에 경로가 활성화되어 있으면 `train_moge12`는 기본적으로 이를 학습한다. `--train_rgb_detail False`로 동결할 수 있다. 이 옵션은 모델 경로 자체를 생성하거나 제거하지 않는다. 기존 neck/백본/head 학습 옵션은 독립적으로 적용된다.

Optimizer를 컴포넌트별로 직접 지정한다면 `rgb_detail.*` 파라미터 그룹과 대응 scheduler 항목을 추가한다. 전체 non-backbone 파라미터를 포함하는 기본 optimizer는 자동으로 포함한다.

기존 sweep runner에서도 `learning_rates.rgb_detail`을 지원한다. 실험 항목의 `rgb_detail`로 모델 설정을 덮어쓸 수 있다. 예시 manifest는 기존 flip 설정에 CNN을 추가한 단일 실험이다. 새 경로 LR `1e-5`는 시작용 설정이며 최적값으로 검증한 것이 아니다.

```bash
python run_script/sweep_5000.py \
  --manifest configs/experiments/gaemi_robot160_rgb_detail.json \
  --root workspace/gaemi_robot160_rgb_detail \
  --dry-run
```

`--dry-run`을 `--smoke`로 바꾸면 짧은 학습 확인, 두 옵션을 빼면 본 학습이다. 실행 전 데이터·초기 체크포인트 경로는 기존 sweep과 동일하게 확인한다. 기존 motion sweep manifest는 변경하지 않았다.

## 체크포인트

CNN이 없는 모델은 초기 가중치로 사용할 수 있다. 새 구조 학습은 새 workspace에서 초기 체크포인트로 시작한다. 다른 구조의 optimizer/EMA를 resume하지 않는다. CNN 학습 후에는 저장된 `model_config`와 가중치를 함께 사용해야 하며, 임의로 경로를 꺼서 로드하면 학습된 CNN 기여가 사라진다.

정확도 향상과 기존 추론 속도 유지는 보장하지 않는다. 실제 로봇에서 640×360·3장 배치로 비교해야 한다. 전용 ONNX 변환 경로는 별도 검증이 필요하다.

## Augmentation 없는 1,500 iteration sweep

실행 스크립트: `run_script/run_gaemi_robot160_rgb_sweep_1500.sh`
설정: `configs/experiments/gaemi_robot160_rgb_sweep_1500.json`

| 실험 | CNN 폭 | CNN LR | 비교 목적 |
| --- | --- | --- | --- |
| `robot160_clean_rgb_c16_lr1e5_1500` | 16 | 1e-5 | 기준 CNN 설정 |
| `robot160_clean_rgb_c16_lr3e5_1500` | 16 | 3e-5 | 새 CNN의 학습률 증가 효과 |
| `robot160_clean_rgb_c32_lr1e5_1500` | 32 | 1e-5 | 기준 대비 CNN 폭 증가 효과 |

세 실험 모두 좌우 반전·모션블러·색상·원근 augmentation을 사용하지 않는다. 640×360, 160토큰, 1,500 iteration, 유효 배치 32(8×4), seed 0, 200 step warmup 후 일정 LR이다. 마지막 DINOv2 4개 블록과 neck·모든 head·CNN을 학습한다. 기존 컴포넌트 LR은 backbone/neck 1e-6, scale 1e-5, points/normal/mask 3e-6으로 유지한다. 데이터와 validation은 기존 robot160 설정을 사용하며 250 step마다 평가한다.

B와 A는 CNN 학습률만 다르고, C와 A는 CNN 폭만 다르다. 구조·학습률을 동시에 바꾸는 비교를 피한다. CNN LR은 검증할 후보값이며 최적값으로 확정한 설정이 아니다. CNN 폭을 늘린 모델은 별도로 로봇 추론 지연을 측정해야 한다.

```bash
# 활성화한 학습용 가상환경에서 실행
bash run_script/run_gaemi_robot160_rgb_sweep_1500.sh --dry-run
bash run_script/run_gaemi_robot160_rgb_sweep_1500.sh --smoke
bash run_script/run_gaemi_robot160_rgb_sweep_1500.sh
```

각 실행은 새 timestamp workspace를 사용한다. 기본 초기 모델은 `/home/coder/MoGe/pretrained/moge-2-vits.pt`이며 `--pretrained`로 바꿀 수 있다. 세 실험은 각각 같은 초기 가중치에서 시작하고 이전 실험을 이어 학습하지 않는다. 초기 모델의 validation baseline은 기본으로 한 번 실행되며, 이것은 네 번째 학습 실험이 아니다. 필요하면 `--skip-baseline`으로 생략한다.

기존 clean의 동일 학습량(step 1499) REL 0.377846, δ1 0.632411을 참고 기준으로 삼되, 실제 실행 조건이 같은지 확인한다. 데이터·초기 체크포인트를 변경하면 새 CNN 효과만 분리한 비교가 되지 않는다. 새 sweep의 결과는 각 실험 workspace와 상위 `summary.csv`, `summary.md`에 저장된다.
