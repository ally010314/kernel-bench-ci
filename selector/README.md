# selector

모델과 GPU 를 입력하면, 측정하지 않고 다음 세 가지를 예측합니다.

1. 커스텀 커널을 쓸지 (커스텀 / TRT 기본 / 측정 필요)
2. 어떤 타일 설정으로 쓸지 (fc1 · fc2 타일, stage)
3. TRT 대비 E2E 가 얼마나 빨라질지 (예상값과 구간)

```bash
python3 -m selector predict vitL
```

| 모델 | 칩 | 판정 | 설정 (fc1_fc2_stage) | 예상 E2E % (구간) |
|---|---|---|---|---:|
| vitL | orin | **커스텀** | `128x128_128x128_s4` | -6.5 (-9.9 ~ -3.0) |
| vitL | ada | **커스텀** | `128x128_128x128_s3` | -8.4 (-12.2 ~ -4.7) |
| vitL | rtx5090 | **측정 필요** | `128x64_64x32_s4` | -3.2 (-9.1 ~ +2.8) |

음수는 TRT 보다 빠르다는 뜻입니다. 예측이 불확실한 경우에는 "측정 필요"로 답합니다.

## 명령

| 명령 | 하는 일 |
|---|---|
| `predict <모델 \| models/x.yml>` | 칩마다 판정 · 타일 · 예상 이득 (`--chip`, `--format md\|json\|csv`) |
| `dataset` | MLflow 측정 결과로 학습 데이터 생성 |
| `train` | 학습 후 등록 기준을 통과하지 못하면 exit 1 (`--register` 로 MLflow Model Registry 등록) |
| `register <명세.yaml>` | 측정 전에 예측을 파일로 커밋해 두기 |
| `score <예측.csv>` | 측정이 끝난 뒤 등록한 예측을 채점 |
| `drift` | 최신 측정 데이터로 지금 모델을 다시 평가해, 기준을 넘으면 exit 1 |

`dataset` · `score` · `drift` 는 `MLFLOW_TRACKING_URI` 가 필요합니다.

## 파일

| 파일 | 내용 |
|---|---|
| `features.py` | 모델 크기 (N · dim · hidden · 블록 수) 와 타일 설정에서 특징 계산 |
| `model.py` | 이득 예측 모델과 판정 규칙 |
| `evaluate.py` | 모델 하나를 빼고 학습해 맞히는지 보는 교차 검증, 등록 기준 |
| `dataset.py` | MLflow → 학습 데이터 |
| `shapes.yml` | 측정한 모델들의 크기 |
| `data/dataset.csv` · `model/selector.json` | 학습 데이터 스냅샷, 커밋된 모델 (PR CI 가 MLflow 없이 재학습할 때 사용) |
| `tests/` | 단위 테스트 13개 |

## CI (`.github/workflows/selector.yml`)

- **PR:** `models/*.yml` 을 바꾸면 테스트 → 스냅샷으로 재학습해 등록 기준 확인 → 바뀐 모델의 판정을 PR 댓글로 답니다.
- **매일:** 최신 MLflow 데이터로 `drift` 를 보고, 재학습 · 등록합니다. 기준을 넘으면 issue 를 엽니다.
