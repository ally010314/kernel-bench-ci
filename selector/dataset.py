"""dataset.py — 이득 예측의 학습 데이터. MLflow gpu-study/tile-search 의 run 하나 = 칩 · TRT · 측정 단계 · 모델 한 점.

정답 gain: graph/gain_bb — CUDA Graph 실행, TRT 5빌드 최속 대비 커스텀 최속의 E2E 변화 % (음수 = 커스텀이 빠름).
쓰는 단계 (stage):
  sep128           13모델 4칩, TRT 10.3 (5090 은 11.2). fc2 128×128 후보를 넣은 재측정
  sep              13모델 Ada · A6000 · 5090, TRT 11.2 (10.3 run 은 빌드 실패라 정답이 없다)
  tile-40-*        새 모델 8개씩 세 회차 (new · r3 · r4) + r4 Orin beitL512 재측정
빼는 것: sep103 (sep128 과 후보가 달라 잡음 비교용), res* · t4 · smoke (조건이 다른 예전 측정), 출력 판정 실패 (nan · nan_t).
round 는 회차 순서 검증에 쓴다: 0 = 13모델, 1 = new, 2 = r3, 3 = r4.
"""
import csv
import json
import os
import re
import urllib.request

from . import features as F

EXPERIMENT = "gpu-study/tile-search"


def round_of(stage):
    if stage in ("sep128", "sep"):
        return 0
    m = re.match(r"tile-40-(new|r3|r4)", stage)
    return {"new": 1, "r3": 2, "r4": 3}[m.group(1)] if m else None


def _post(uri, path, body):
    req = urllib.request.Request(f"{uri.rstrip('/')}/api/2.0/mlflow/{path}", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def _get(uri, path):
    return json.load(urllib.request.urlopen(f"{uri.rstrip('/')}/api/2.0/mlflow/{path}", timeout=30))


def mlflow_runs(uri, experiment=EXPERIMENT):
    eid = _get(uri, "experiments/get-by-name?experiment_name=" + urllib.request.quote(experiment))["experiment"]["experiment_id"]
    out, tok = [], None
    while True:
        b = {"experiment_ids": [eid], "max_results": 1000}
        if tok:
            b["page_token"] = tok
        r = _post(uri, "runs/search", b)
        out += r.get("runs", [])
        tok = r.get("next_page_token")
        if not tok:
            return out


def from_runs(runs):
    rows = []
    for x in runs:
        p = {d["key"]: d["value"] for d in x["data"].get("params", [])}
        m = {d["key"]: d["value"] for d in x["data"].get("metrics", [])}
        rnd = round_of(p.get("stage", ""))
        g = m.get("graph/gain_bb")
        if rnd is None or g is None or m.get("nan") or m.get("nan_t") or p.get("model") not in F.shapes():
            continue
        if p["device"] not in F.CHIPS or (p["stage"] == "sep" and not p["trt"].startswith("11.2")):
            continue
        rows.append(dict(src=p["stage"], chip=p["device"], trt="11.2" if p["trt"].startswith("11.2") else "10.3", model=p["model"],
                         round=rnd, gain=g, gain_s=m.get("stream/gain_bb"), trt_ms=m.get("graph/trt_min"), cus_ms=m.get("graph/cust_min"),
                         run_id=x["info"]["run_id"], **F.features(p["device"], p["model"])))
    rows.sort(key=lambda r: (r["round"], r["chip"], r["model"], r["trt"]))
    return rows


def from_mlflow(uri=None):
    uri = uri or os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise SystemExit("MLFLOW_TRACKING_URI 미설정")
    return from_runs(mlflow_runs(uri))


NUM = ("gain", "gain_s", "trt_ms", "cus_ms", "N", "dim", "hidden", "blocks", "mixer", "mlp_share", "fc1_cta_sm", "fc2_cta_sm",
       "fc2_fill", "fc1_rounds", "fc2_rounds", "round", "stage")


def save_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def load_csv(path):
    rows = []
    for r in csv.DictReader(open(path)):
        for k in NUM:
            if k in r and r[k] not in ("", "None"):
                r[k] = float(r[k])
        rows.append(r)
    return rows
