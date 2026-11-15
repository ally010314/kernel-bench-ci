"""evaluate.py — 이득 예측 · 판정 정책의 오프라인 평가.

LOMO  : 모델 하나 빼기. 같은 모델의 모든 칩 · TRT 점을 함께 뺀다 (새 모델에 대한 성적).
회차  : 13모델로 정해 new → +new 로 정해 r3 → +r3 로 정해 r4. 사전 등록과 같은 상황.
지표  : 평균 절대 오차 %p, 승패 정확도, 판정 손해 (커스텀을 골랐는데 지면 그 손해, TRT 를 골랐는데 커스텀이 이기면 놓친 이득).
"""
import statistics as st

from . import features as F
from .model import FORMS, GainModel, Policy, regret


def lomo(rows, form):
    out = []
    for m in sorted({r["model"] for r in rows}):
        tr = [r for r in rows if r["model"] != m]
        g = GainModel(form).fit(tr)
        out += [(g.predict(r), r) for r in rows if r["model"] == m]
    return out


def sequential(rows, form):
    out = []
    for k in (1, 2, 3):
        te = [r for r in rows if r["round"] == k]
        if te:
            g = GainModel(form).fit([r for r in rows if r["round"] < k])
            out += [(g.predict(r), r) for r in te]
    return out


def scores(pairs):
    if not pairs:
        return {}
    e = [abs(p - r["gain"]) for p, r in pairs]
    reg = [regret("custom" if p < 0 else "trt", r["gain"]) for p, r in pairs]
    return dict(n=len(pairs), mae=round(st.mean(e), 2), sign_acc=round(sum((p < 0) == (r["gain"] < 0) for p, r in pairs) / len(pairs), 3),
                regret=round(st.mean(reg), 2), regret_worst=round(max(reg), 1))


def policy_scores(pairs, policy):
    reg, meas = [], 0
    for p, r in pairs:
        a, _ = policy.decide(r["chip"], r, p)
        meas += a == "measure"
        reg.append(regret(a, r["gain"]))
    return dict(n=len(pairs), measure=round(meas / len(pairs), 2), regret=round(st.mean(reg), 3), regret_worst=round(max(reg), 1))


def report(rows, policy=None):
    policy = policy or Policy()
    out = dict(n=len(rows), models=len({r["model"] for r in rows}), forms={})
    for form in FORMS:
        L, S = lomo(rows, form), sequential(rows, form)
        out["forms"][form] = dict(lomo=scores(L), seq=scores(S),
                                  lomo_by_chip={c: scores([x for x in L if x[1]["chip"] == c]) for c in F.CHIPS})
    L, S = lomo(rows, "M3"), sequential(rows, "M3")
    pols = {"항상 커스텀": Policy(measure_chips=(), min_n=0), "5090 은 측정": Policy(min_n=0),
            "5090 · N<64 측정 + M3 > +2.7 이면 TRT": Policy(trt_above=2.7), "채택 정책": policy}
    out["policy"] = {n: dict(lomo=policy_scores(L, p), seq=policy_scores(S, p)) for n, p in pols.items()}
    out["band"] = band(L)
    out["worst"] = sorted(([round(p, 1), r["gain"], r["chip"], r["model"]] for p, r in L), key=lambda t: -abs(t[0] - t[1]))[:8]
    return out


def band(pairs, q=0.8):
    """칩별 LOMO 절대 오차의 q 분위. 예상 이득 구간의 반폭으로 쓴다."""
    out = {}
    for c in F.CHIPS:
        e = sorted(abs(p - r["gain"]) for p, r in pairs if r["chip"] == c)
        if e:
            out[c] = round(e[min(len(e) - 1, int(q * len(e)))], 2)
    return out


def gate(rep, max_regret=0.1, max_worst=2.0):
    """등록 기준: 채택 정책의 LOMO · 회차 판정 손해가 기준 안이어야 한다."""
    p = rep["policy"]["채택 정책"]
    bad = [f"{k} 손해 {v['regret']} > {max_regret} 또는 최악 {v['regret_worst']} > {max_worst}"
           for k, v in p.items() if v["regret"] > max_regret or v["regret_worst"] > max_worst]
    return not bad, bad
