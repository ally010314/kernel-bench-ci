# kernel-bench-ci

연구실 팀에서 CUTLASS 기반 2:4 Sparse MLP 커널과 TensorRT 플러그인을 개발하면서, 커널 수정에 따른 정확도·성능 회귀를 실제 GPU에서 검증하고 4종의 GPU 서버에서 실험을 실행·수집·판정하기 위해 구축한 CLI 도구 및 CI 시스템입니다.

## 프로젝트 개요

CLI가 검증과 판정 로직을 담당하고, CI 워크플로는 정해진 시점에 CLI를 실행합니다. CI의 성공·실패 여부도 CLI가 반환하는 종료 코드(exit code)를 기준으로 결정됩니다.

```
  사람 (터미널) ──┐
                  ├──▶  CLI 도구  ──▶  GPU 장비 4종 · MLflow · GitHub issue
  GitHub Actions ─┘
```

| CLI 도구 | 해결하는 문제 | 하는 일 | CI 에서 실행 |
|---|---|---|---|
| `bench/run.py` | 수정한 커널의 결과 정확도, 성능 판정 | 동일한 2:4 가중치의 FP32 결과와 비교해 정확도를, 장비별 baseline과 비교해 latency 회귀를 판정. exit 0 통과 · 1 정확도 · 2 성능 · 3 실행 오류 | `bench.yml` (커널 PR · main) |
| `experiments/expctl.py` | 장시간 실험의 실행 및 완료 여부 확인 | YAML 명세로 4개 장비에서 실험 실행. 시작 전 GPU 사용률·클럭·다른 사용자·디스크 점검 및 시험 실행, 종료 후 worker 상태와 결과 누락 판정 | `experiment.yml` (수동 실행), `collect.yml` (30분마다 판정) |
| `experiments/collect.py` | GPU별 실험 결과 수집 | 각 GPU의 실험 결과를 MLflow로 수집하고, 중복 측정은 다시 적재하지 않음 | `collect.yml` (30분마다) |
| `python3 -m selector` | GPU, 모델마다 최적의 커널 구성 탐색 | 누적 측정 데이터로 학습해 커스텀 커널 사용 여부, 최적 타일 구성, 예상 성능을 측정 없이 예측 | `selector.yml` (모델 PR 댓글 작성 · 매일 재학습)|

## 개발 배경

- **측정 결과를 믿기 어려웠습니다.** 커널 계산에 오류가 있어도 전체 결과의 평균 오차는 크게 달라지지 않아 문제를 놓칠 수 있었고, GPU 클럭이나 다른 사용자의 작업에 따라서도 latency가 달라졌습니다. → 정확도 판정 기준을 코드로 고정하고, GPU 클럭과 사용률을 확인한 뒤 측정하도록 했습니다.
- **실험이 정상적으로 완료됐는지 확인하기 어려웠습니다.** 모든 모델 실행이 실패해도 완료 문구가 출력되거나, worker가 종료된 뒤에도 상태가 RUNNING으로 남는 문제가 있었습니다. → 본 실험 전에 모델 하나를 끝까지 실행해 환경과 실행 과정을 확인하고, 실험 완료 여부는 상태 메시지가 아닌 결과 파일의 생성 및 기록 수를 기준으로 판정하도록 했습니다.
- **실험 결과가 흩어져 있었습니다.** 기존 실험 결과들이 MLflow에 수집되지 않은 경우가 많았습니다. → 장비별 결과를 30분마다 자동 수집하고, 누적된 측정 데이터로 GPU와 모델 조건에 따른 커스텀 커널 선택 및 최적 타일 구성을 예측하도록 했습니다.

## 사용

```bash
# 커널 회귀
ARCH=sm_87 MODEL=mixerL ./build.sh
python3 bench/run.py --model mixerL                     # 빌드한 커널을 실행하고 정확도와 성능 확인, 결과에 따라 exit code 반환
python3 bench/run.py --model mixerL --update-baseline   # 성능 개선이 의도한 변화면 baseline을 새로운 측정값으로 갱신 및 커밋

# 실험
python3 experiments/expctl.py validate                  # 실험 YAML 파일에 필수 항목이 있는지, 형식이 올바른지 검사
python3 experiments/expctl.py preflight tile-40-r4      # GPU 사용률, 클럭, 다른 사용자, 디스크 등을 점검하고 모델 하나를 시험 실행
python3 experiments/expctl.py run tile-40-r4            # 사전 점검을 통과한 장비에서 실제 실험 실행
python3 experiments/expctl.py judge --issue             # 실행 상태와 결과 파일을 확인해 실험 성공·실패 판정. 실패 시 GitHub Issue 생성

# 수집 · 예측
python3 experiments/collect.py --dry
python3 -m selector predict vitL deitS                  # GPU와 모델 조건에 따라 커널, 타일 선택 및 성능 예측
```

실험 명세 예 (`experiments/specs/tile-40-r4.yaml` 를 줄인 것):

```yaml
name: tile-40-r4
models: models_r4.txt                     # 돌릴 모델 목록
tiles: {fc1: [128x128, 64x64], fc2: [64x64, 128x64], stages: [3, 4]}
targets:                                  # 장비마다 TRT 버전, GPU (auto 는 빈 GPU 를 고른다)
  ada:     {trt: ["10.3.0"], gpus: auto, max_gpus: 2}
  rtx5090: {trt: ["11.2.1.2"], gpus: auto}
```

## 구조

```
kernel-bench-ci/
├── bench/              CLI ① 커널 회귀 판정 (run.py), 측정 하네스 (mlp_bench.cu), MLflow 기록
├── baselines/          장비 × 모델별 latency baseline
├── experiments/
│   ├── expctl.py       CLI ② 실험 실행 · 판정
│   ├── collect.py      CLI ③ 장비별 결과 수집
│   ├── devices.yml     GPU, 기대 클럭, TensorRT 버전 등 장비 설정
│   ├── specs/          실험 조건을 정의한 YAML 파일
│   ├── tools/          GPU 장비로 복사되어 실제 측정을 수행하는 스크립트
│   ├── records/        측정 전에 등록한 타일 예측 (selector 테스트가 사용)
│   └── tests/
├── selector/           CLI ④ 커널 선택 예측 (학습 · 평가 · 예측) + tests/
├── models/             배포할 모델 명세 (PR 로 올리면 selector 판정이 댓글로)
├── kernels/paper/      측정 대상 커널 (공개본에서는 소스를 가림)
├── deploy/monitoring/  MLflow + Grafana docker compose
├── .github/workflows/  bench · experiment · collect · selector
└── build.sh            커널 · 플러그인 · 하네스 빌드
```

## 품질 관리

- **테스트:** `experiments/tests` 39개, `selector/tests` 13개로 총 52개의 테스트를 작성했습니다. 실험 설정 검증, GPU 사용 가능 여부 판단, 다른 사용자 사용 여부 확인, 실험 상태 분류, 커널 선택 규칙 등을 검사합니다. 실제 GPU 없이도 실행할 수 있습니다.
- **PR 검사:** 코드 변경 사항을 올리면 변경 종류에 따라 자동으로 검사합니다. 실험 코드와 모델 명세는 테스트와 설정 검증을 수행하고, 커널 변경은 Orin·Ada GPU에서 5개 모델을 실행해 정확도와 성능을 확인합니다.
- **GPU 측정 환경 관리:** GPU 사용률이 5% 이하가 될 때까지 기다리고, GPU 클럭이 정해진 값과 다르면 측정을 중단합니다. 또한 한 장비에서는 한 번에 하나의 측정만 실행해 다른 작업으로 인한 성능 변화를 줄였습니다.
- **측정 결과 기록:** MLflow에 결과를 저장하는 작업과 CI 통과 여부를 판정하는 작업을 분리했습니다. 따라서 결과 기록에 실패하더라도 정확도·성능 검사 결과에 따른 CI 판정은 유지됩니다.
- **의존성 관리:** CUTLASS는 특정 커밋으로 버전을 고정해 빌드 환경이 달라지지 않도록 했습니다. Jetson에서는 MLflow 패키지를 설치하는 대신 REST API로 결과를 전송해 추가 설치 항목을 줄였습니다.

## 공개본 안내

연구용 저장소를 공개하기 위해 다음 두 가지를 바꿨습니다.

- **접속 정보:** 서버 주소, 접속 포트, 계정 정보는 실제 값 대신 `<host-ada>`, `<port>`, `<user>`와 같은 자리 표시자로 바꿨습니다.
- **커널 소스:** 논문 심사가 진행 중인 커널 소스(`kernels/paper/*.cu`, `*.cpp`)는 공개하지 않았습니다.

따라서 공개본에서는 실제 GPU 커널 빌드와 서버 접속은 할 수 없지만, GPU 없이 실행할 수 있는 테스트 코드는 그대로 사용할 수 있습니다.

커널 선택기 설명은 [selector/README.md](selector/README.md) 에 있습니다.
