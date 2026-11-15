#!/usr/bin/env python3
"""predict_tile.py — 측정 전에 칩 · 모델마다 커널 설정(fc1 타일, fc2 타일, fc2 stage)을 예측한다.

v1 규칙 (2026-11-07, 13모델 × 4칩 sep103 · sep128 측정에서 정함. 이 측정이 tile-40 의 채점 대상이 아니다):
  fc2   : CTA / SM 이 0.75 에 가장 가까운 타일. CTA = ceil(dim / M2) × ceil(N / N2)
  fc1   : CTA / SM ≥ 0.5 인 타일 중 면적이 가장 큰 것. 없으면 CTA 가 가장 많은 것. CTA = ceil(hidden / M1) × ceil(N / N1)
          0.5 는 칩 하나 빼기 평가에서 네 칩 중 세 칩이 고른 값이다.
  stage : fc2 공유 메모리 (M2 × 72 + N2 × 128 B/stage) 가 칩 한도 안이면 4, 아니면 3.
같은 거리면 이름순 앞의 타일을 고른다. 상위 3개는 규칙 거리 순서다.

v2 규칙 (2026-11-07, 같은 데이터에서 규칙 후보군을 교차검증으로 고름. 모델 하나 빼기 평균 손해 0.46%, 칩 하나 빼기 0.46%):
  fc2   : 마지막 라운드 채움률 CTA / (SM × ceil(CTA / SM)) 이 0.5 이상인 타일 중 CTA 가 가장 적은 것 (같으면 M 이 큰 것).
          없으면 채움률이 가장 높은 것.
  fc1   : CTA ≥ max(32, 0.5 × SM) 인 타일 중 면적이 가장 큰 것. 없으면 CTA 가 가장 많은 것.
  stage : 공유 메모리 한도 안의 stage 중, fc2 CTA 가 SM 보다 많으면 SM 당 동시 CTA 수 (min(한도 / stage 크기, 라운드 수)) 가
          가장 많은 것, 같으면 큰 stage. CTA 가 SM 이하면 한도 안의 가장 큰 stage.

v3 규칙 (2026-11-08, 13모델 + 새 8모델 = 21모델 × 4칩에서 정함. 13모델로 정해 새 8모델에 적용 0.35%, 모델 하나 빼기 0.55%, 칩 하나 빼기 0.85%):
  fc1 · fc2 : 같은 비용식이 가장 작은 타일. 비용 = ceil(CTA / SM) × M × N × (1 + 64/M + 64/N).
              라운드 수 × 한 CTA 일의 양 × 타일 테두리에 비례하는 메모리 비용. 끝 조각이 비어도 한 CTA 일로 센다.
              같으면 fc2 는 M 이 큰 것, fc1 은 면적이 큰 것. 64 의 계수 1 은 0.25~2 중 고른 값 (1 ~ 1.5 가 같은 결과).
  stage : v2 와 같다.

  python3 experiments/tools/predict_tile.py experiments/specs/tile-40.yaml [--rule v3] [--models models.txt] > out.csv
  --models 는 "이름 N dim hidden" 줄로 된 파일이다. 없으면 아래 13모델.
"""
import csv, math, sys
import yaml

SM = {"orin": 16, "ada": 22, "a6000": 84, "rtx5090": 170}
SMEM = {"orin": 166912, "ada": 101376, "a6000": 101376, "rtx5090": 101376}
# 모델: (토큰 수 N, dim, hidden). channel MLP 기준. CaiT 는 class attention 앞 블록 기준.
MODELS = {"deitS": (197, 384, 1536), "deit3S": (197, 384, 1536), "caitS24": (197, 384, 1536), "deitB": (197, 768, 3072),
          "deit3B": (197, 768, 3072), "beitB": (197, 768, 3072), "vitB384": (577, 768, 3072), "mixerB": (196, 768, 3072),
          "vitL": (197, 1024, 4096), "deit3L": (197, 1024, 4096), "mixerL": (196, 1024, 4096), "vitL384": (577, 1024, 4096),
          "vitH": (257, 1280, 5120)}
FC2_TARGET, FC1_MIN = 0.75, 0.5


def mn(t):
    a, b = t.split("x"); return int(a), int(b)


def cta(t, rows, n):
    a, b = mn(t); return math.ceil(rows / a) * math.ceil(n / b)


def fc2_rank(chip, model, cands):
    n, d, _ = MODELS[model]
    return sorted(cands, key=lambda t: (abs(cta(t, d, n) / SM[chip] - FC2_TARGET), t))


def fc1_pick(chip, model, cands):
    n, _, h = MODELS[model]
    ok = [t for t in cands if cta(t, h, n) / SM[chip] >= FC1_MIN]
    if ok:
        return max(ok, key=lambda t: (mn(t)[0] * mn(t)[1], t))
    return max(cands, key=lambda t: (cta(t, h, n), t))


def stage(chip, fc2, stages):
    a, b = mn(fc2); per = a * 72 + b * 128
    ok = [s for s in stages if s * per <= SMEM[chip]]
    return max(ok) if ok else min(stages)


def per_stage(fc2):
    a, b = mn(fc2); return a * 72 + b * 128


def fc2_v2(chip, model, cands):
    n, d, _ = MODELS[model]; S = SM[chip]
    def fill(t):
        c = cta(t, d, n); return c / (S * math.ceil(c / S))
    ok = sorted([t for t in cands if fill(t) >= 0.5], key=lambda t: (cta(t, d, n), -mn(t)[0], t))
    rest = sorted([t for t in cands if t not in ok], key=lambda t: (-fill(t), t))
    return ok + rest


def fc1_v2(chip, model, cands):
    n, _, h = MODELS[model]; need = max(32, 0.5 * SM[chip])
    ok = [t for t in cands if cta(t, h, n) >= need]
    if ok:
        return max(ok, key=lambda t: (mn(t)[0] * mn(t)[1], t))
    return max(cands, key=lambda t: (cta(t, h, n), t))


def stage_v2(chip, model, fc2, stages):
    n, d, _ = MODELS[model]; c = cta(fc2, d, n); S = SM[chip]
    ok = [s for s in stages if s * per_stage(fc2) <= SMEM[chip]]
    if not ok:
        return min(stages)
    if c > S:
        return max(ok, key=lambda s: (min(SMEM[chip] // (s * per_stage(fc2)), math.ceil(c / S)), s))
    return max(ok)


def cost_v3(chip, t, rows, n):
    a, b = mn(t); return math.ceil(cta(t, rows, n) / SM[chip]) * a * b * (1 + 64 / a + 64 / b)


def fc2_v3(chip, model, cands):
    n, d, _ = MODELS[model]
    return sorted(cands, key=lambda t: (cost_v3(chip, t, d, n), -mn(t)[0], t))


def fc1_v3(chip, model, cands):
    n, _, h = MODELS[model]
    return min(cands, key=lambda t: (cost_v3(chip, t, h, n), -mn(t)[0] * mn(t)[1], t))


def main(spec, rule="v1"):
    t = yaml.safe_load(open(spec))["tiles"]; w = csv.writer(sys.stdout)
    w.writerow(["chip", "model", "N", "dim", "pred", "fc1", "fc2", "stage", "fc2_cta_per_sm", "fc1_cta_per_sm", "fc2_rank2", "fc2_rank3"])
    for chip in SM:
        for m, (n, d, h) in MODELS.items():
            if rule == "v3":
                r = fc2_v3(chip, m, t["fc2"]); b = r[0]; a = fc1_v3(chip, m, t["fc1"]); s = stage_v2(chip, m, b, t["stages"])
            elif rule == "v2":
                r = fc2_v2(chip, m, t["fc2"]); b = r[0]; a = fc1_v2(chip, m, t["fc1"]); s = stage_v2(chip, m, b, t["stages"])
            else:
                r = fc2_rank(chip, m, t["fc2"]); b = r[0]; a = fc1_pick(chip, m, t["fc1"]); s = stage(chip, b, t["stages"])
            w.writerow([chip, m, n, d, f"{a}_{b}_s{s}", a, b, s, round(cta(b, d, n) / SM[chip], 2), round(cta(a, h, n) / SM[chip], 2), r[1], r[2]])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("spec"); ap.add_argument("--rule", default="v1", choices=["v1", "v2", "v3"]); ap.add_argument("--models")
    x = ap.parse_args()
    if x.models:
        MODELS.clear()
        for l in open(x.models):
            if l.strip() and not l.startswith("#"):
                k, n, d, h = l.split()[:4]; MODELS[k] = (int(n), int(d), int(h))
    main(x.spec, x.rule)
