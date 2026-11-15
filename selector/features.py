"""features.py — 측정 전에 계산할 수 있는 특징. 모델 모양 + 칩 SM 수 + v3 타일 규칙이 고른 설정의 라운드 정보.

타일 설정은 experiments/tools/predict_tile.py 의 v3 규칙을 그대로 쓴다 (사전 등록 3 · 4차 평균 손해 0.93 · 0.40%).
"""
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "tools"))
import predict_tile as PT  # noqa: E402

CHIPS = ["orin", "ada", "a6000", "rtx5090"]
SM = PT.SM
# tile-40 명세의 후보 (experiments/specs/tile-40*.yaml)
FC1 = ["128x128", "128x64", "128x32", "64x64"]
FC2 = ["64x64", "64x32", "128x64", "128x128", "256x64"]
STAGES = [3, 4]

_SHAPES = None


def shapes():
    global _SHAPES
    if _SHAPES is None:
        _SHAPES = yaml.safe_load(open(Path(__file__).with_name("shapes.yml")))
    return _SHAPES


def shape_of(model):
    """이름 (shapes.yml) 또는 dict (N, dim, hidden, blocks, token_hidden)."""
    if isinstance(model, dict):
        s = dict(model)
    else:
        if model not in shapes():
            raise KeyError(f"모양을 모르는 모델: {model} (selector/shapes.yml 또는 models/*.yml 의 shape 에 적는다)")
        s = dict(shapes()[model])
    s.setdefault("blocks", 12)
    s.setdefault("token_hidden", 0)
    for k in ("N", "dim", "hidden", "blocks", "token_hidden"):
        s[k] = int(s[k])
    return s


def tile_config(chip, s):
    """v3 규칙으로 고른 (fc1, fc2, stage, fc2 2 · 3순위)."""
    key = f"_sel_{s['N']}_{s['dim']}_{s['hidden']}"
    PT.MODELS[key] = (s["N"], s["dim"], s["hidden"])
    try:
        r2 = PT.fc2_v3(chip, key, FC2)
        fc1 = PT.fc1_v3(chip, key, FC1)
        st = PT.stage_v2(chip, key, r2[0], STAGES)
    finally:
        PT.MODELS.pop(key, None)
    return fc1, r2[0], st, r2[1:3]


def features(chip, model):
    s = shape_of(model)
    N, d, h, th = s["N"], s["dim"], s["hidden"], s["token_hidden"]
    mlp = 2 * N * d * h
    other = 2 * d * N * th if th else 4 * N * d * d + 2 * N * N * d  # Mixer 는 토큰 MLP, 아니면 attention
    fc1, fc2, st, alt = tile_config(chip, s)
    c1, c2, S = PT.cta(fc1, h, N), PT.cta(fc2, d, N), SM[chip]
    return dict(N=N, dim=d, hidden=h, blocks=s["blocks"], mixer=int(bool(th)), mlp_share=round(mlp / (mlp + other), 4),
                fc1=fc1, fc2=fc2, stage=st, fc2_alt=" ".join(alt),
                fc1_cta_sm=round(c1 / S, 3), fc2_cta_sm=round(c2 / S, 3), fc2_fill=round(c2 / (S * math.ceil(c2 / S)), 3),
                fc1_rounds=math.ceil(c1 / S), fc2_rounds=math.ceil(c2 / S))
