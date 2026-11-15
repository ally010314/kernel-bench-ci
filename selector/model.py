"""model.py — 이득 크기 예측 (M3) 과 판정 정책.

M3: gain = 칩별 절편 + 칩별 MLP 비중 기울기 + log N · log dim · TRT 11.2 + v3 설정의 fc1 CTA/SM · fc2 채움률 · fc2 라운드 · 1/N.
    작은 릿지 (λ = 1e-3). 점이 칩당 40~50개라 매개변수를 늘리지 않는다 (docs: S1 이득 예측).
정책: 커스텀 / TRT 기본 / 측정 필요 셋 중 하나로 답한다.
    - 모르는 칩 → 측정 필요 (칩별 계수라 새 칩에 쓸 수 없다)
    - 5090 또는 N < 64 → 측정 필요 (회귀가 승패를 못 맞히는 영역. 판정 손해 0.01~0.02%)
    - 나머지 → 커스텀. (예측이 trt_above 보다 크면 TRT 기본으로 돌리는 선택지가 있지만, 11/9 평가에서
      M3 의 손해 예측을 믿으면 LOMO 최악 손해가 1.4 → 3.4% 로 나빠져 기본값은 끈다)
예상 이득의 구간은 칩별 LOMO 절대 오차의 80% 분위 (band) 다.
"""
import json
import math

import numpy as np

from . import features as F

FORMS = ("B0", "B1", "M1", "M2", "M3")


def design(r, form):
    oh = [1.0 if r["chip"] == c else 0.0 for c in F.CHIPS]
    x = list(oh)
    if form in ("M1", "M2", "M3"):
        x += [o * r["mlp_share"] for o in oh]
    if form in ("M2", "M3"):
        x += [math.log(r["N"]), math.log(r["dim"]), 1.0 if str(r["trt"]).startswith("11.2") else 0.0]
    if form == "M3":
        x += [r["fc1_cta_sm"], r["fc2_fill"], r["fc2_rounds"], 1.0 / r["N"]]
    return x


class GainModel:
    def __init__(self, form="M3", w=None, lam=1e-3, band=None):
        self.form, self.w, self.lam = form, w, lam
        self.band = band or {}  # 칩 → 예측 구간 반폭 (%p)

    def fit(self, rows):
        if self.form == "B0":
            return self
        A = np.array([design(r, self.form) for r in rows])
        y = np.array([r["gain"] for r in rows])
        self.w = np.linalg.solve(A.T @ A + self.lam * np.eye(A.shape[1]), A.T @ y)
        return self

    def predict(self, r):
        if self.form == "B0":
            return -1.0
        return float(np.array(design(r, self.form)) @ self.w)

    def to_dict(self):
        return dict(form=self.form, lam=self.lam, w=[round(float(v), 8) for v in self.w] if self.w is not None else None, band=self.band)

    @classmethod
    def from_dict(cls, d):
        return cls(d["form"], np.array(d["w"]) if d.get("w") is not None else None, d.get("lam", 1e-3), d.get("band"))


class Policy:
    """measure_chips · min_n 은 사전 등록 대상이다. 바꾸면 버전을 올린다."""

    def __init__(self, measure_chips=("rtx5090",), min_n=64, trt_above=None):
        self.measure_chips, self.min_n, self.trt_above = tuple(measure_chips), min_n, trt_above

    def decide(self, chip, feats, pred):
        if chip not in F.CHIPS:
            return "measure", "학습에 없는 칩"
        if chip in self.measure_chips:
            return "measure", f"{chip}: 이득이 0 근처에 몰려 승패 예측이 약함"
        if feats["N"] < self.min_n:
            return "measure", f"N {feats['N']} < {self.min_n}: 작은 문제 크기는 학습 점이 적고 크게 진 적이 있음"
        if self.trt_above is not None and pred > self.trt_above:
            return "trt", f"예측 {pred:+.1f}% 가 기준 +{self.trt_above}% 보다 손해"
        return "custom", f"학습 칩 · N ≥ {self.min_n}: 측정한 137점 중 진 것 3점 (최대 1.4%)"

    def to_dict(self):
        return dict(measure_chips=list(self.measure_chips), min_n=self.min_n, trt_above=self.trt_above)


def regret(action, gain):
    """판정 손해 %. 측정하면 둘 다 재서 빠른 쪽을 쓰므로 0."""
    if action == "measure":
        return 0.0
    return max(gain, 0.0) if action == "custom" else max(-gain, 0.0)


class Selector:
    def __init__(self, gain_model, policy, meta=None):
        self.gain, self.policy, self.meta = gain_model, policy, meta or {}

    def predict(self, chip, model):
        f = F.features(chip, model) if chip in F.CHIPS else dict(F.shape_of(model))
        trt = "11.2" if chip == "rtx5090" else "10.3"
        pred = self.gain.predict(dict(f, chip=chip, trt=trt)) if chip in F.CHIPS else None
        action, why = self.policy.decide(chip, f, pred if pred is not None else 0.0)
        b = self.gain.band.get(chip)
        out = dict(chip=chip, action=action, reason=why, trt=trt,
                   expected_gain=None if pred is None else round(pred, 2),
                   interval=None if pred is None or b is None else [round(pred - b, 2), round(pred + b, 2)])
        if chip in F.CHIPS:
            out.update(fc1=f["fc1"], fc2=f["fc2"], stage=f["stage"], config=f"{f['fc1']}_{f['fc2']}_s{f['stage']}",
                       fc2_alt=f["fc2_alt"], fc2_cta_sm=f["fc2_cta_sm"], fc2_rounds=f["fc2_rounds"], mlp_share=f["mlp_share"])
        return out

    def to_dict(self):
        return dict(meta=self.meta, gain=self.gain.to_dict(), policy=self.policy.to_dict())

    def save(self, path):
        json.dump(self.to_dict(), open(path, "w"), indent=1, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        d = json.load(open(path))
        return cls(GainModel.from_dict(d["gain"]), Policy(**d["policy"]), d.get("meta"))
