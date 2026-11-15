"""python3 -m selector <명령>

  dataset  [--out ds.csv]                         MLflow gpu-study/tile-search 에서 학습 데이터를 만든다
  train    [--data ds.csv] [--out selector/model/selector.json] [--log] [--register]
                                                  이득 예측 학습 · 오프라인 평가 · 등록 기준 판정.
                                                  --log 는 MLflow gpu-study/selector 에 run, --register 는 Model Registry 등록
  predict  <모델|models/x.yml ...> [--chip all] [--format md|json|csv]
                                                  칩마다 사용 여부 · 설정 · 예상 이득
  register <명세.yaml> [--rule-version v]         측정 전 예측을 records/predictions/<명세>_selector.csv 로 남긴다 (사전 등록)
  score    <예측.csv> [--stage 이름]              등록한 예측을 MLflow 측정 결과로 채점 (판정 손해 · 오차)
  drift    [--max-regret 0.1]                     지금 모델을 MLflow 의 최신 데이터 전체에 대 보고 기준을 넘으면 실패
env: MLFLOW_TRACKING_URI
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from . import dataset as D
from . import evaluate as E
from . import features as F
from .model import GainModel, Policy, Selector, regret

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(__file__).with_name("model") / "selector.json"
REGISTERED = "tile-gain-selector"


def _uri():
    u = os.environ.get("MLFLOW_TRACKING_URI")
    if not u:
        sys.exit("MLFLOW_TRACKING_URI 미설정")
    return u


def _rows(path):
    return D.load_csv(path) if path else D.from_mlflow(_uri())


def _sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


def train(rows):
    rep = E.report(rows)
    ok, why = E.gate(rep)
    g = GainModel("M3").fit(rows)
    g.band = rep["band"]
    meta = dict(version=time.strftime("%Y%m%d-%H%M"), git=_sha(), n=len(rows), models=len({r["model"] for r in rows}),
                stages=sorted({r["src"] for r in rows}), gate_ok=ok, gate=why,
                lomo_mae_m3=rep["forms"]["M3"]["lomo"]["mae"], policy=rep["policy"]["채택 정책"])
    return Selector(g, Policy(), meta), rep


def report_md(rep, meta):
    L = [f"### 선택기 학습 `{meta['version']}` ({meta['n']}점 · 모델 {meta['models']}개)", "",
         f"등록 기준: {'통과' if meta['gate_ok'] else '실패 — ' + '; '.join(meta['gate'])}", "",
         "| 이득 예측 | LOMO 오차 %p | 승패 정확도 | 회차 순서 오차 %p |", "|---|---:|---:|---:|"]
    for f, v in rep["forms"].items():
        L.append(f"| {f} | {v['lomo']['mae']} | {v['lomo']['sign_acc']} | {v['seq'].get('mae', '')} |")
    L += ["", "| 판정 정책 | 측정 비율 | 손해 평균 / 최악 % (LOMO) | 손해 평균 / 최악 % (회차) |", "|---|---:|---:|---:|"]
    for n, v in rep["policy"].items():
        L.append(f"| {n} | {v['lomo']['measure']} | {v['lomo']['regret']} / {v['lomo']['regret_worst']} | {v['seq']['regret']} / {v['seq']['regret_worst']} |")
    L += ["", "예상 이득 구간 (칩별 LOMO 오차 80% 분위, %p): " + ", ".join(f"{c} ±{b}" for c, b in rep["band"].items())]
    return "\n".join(L) + "\n"


def log_mlflow(sel, rep, md, register):
    sys.path.insert(0, str(ROOT / "bench"))
    from mlflow_log import Client
    uri = _uri(); c = Client(uri); e = c.experiment_id("gpu-study/selector")
    run = c.call("POST", "runs/create", json=dict(experiment_id=e, run_name=f"selector/{sel.meta['version']}", start_time=int(time.time() * 1000),
                                                    tags=[dict(key="mlflow.runName", value=f"selector/{sel.meta['version']}")]))["run"]["info"]
    rid, ts = run["run_id"], int(time.time() * 1000)
    m = {f"{f}/lomo_mae": v["lomo"]["mae"] for f, v in rep["forms"].items()}
    m.update({f"{f}/seq_mae": v["seq"]["mae"] for f, v in rep["forms"].items() if v["seq"]})
    pol = rep["policy"]["채택 정책"]
    m.update({"policy/lomo_regret": pol["lomo"]["regret"], "policy/lomo_worst": pol["lomo"]["regret_worst"], "policy/measure_rate": pol["lomo"]["measure"],
              "policy/seq_regret": pol["seq"]["regret"], "policy/seq_worst": pol["seq"]["regret_worst"], "n": sel.meta["n"]})
    m.update({f"band/{k}": v for k, v in rep["band"].items()})
    p = dict(form="M3", measure_chips=",".join(sel.policy.measure_chips), min_n=sel.policy.min_n, git=sel.meta["git"],
             stages=" ".join(sel.meta["stages"])[:490])
    c.call("POST", "runs/log-batch", json=dict(run_id=rid, params=[dict(key=k, value=str(v)) for k, v in p.items()],
                                                   metrics=[dict(key=k, value=float(v), timestamp=ts, step=0) for k, v in m.items()],
                                                   tags=[dict(key="gate", value="pass" if sel.meta["gate_ok"] else "fail")]))
    for name, body in (("selector/selector.json", json.dumps(sel.to_dict(), ensure_ascii=False, indent=1)), ("report.md", md),
                       ("report.json", json.dumps(rep, ensure_ascii=False, indent=1))):
        c.s.put(f"{uri.rstrip('/')}/api/2.0/mlflow-artifacts/artifacts/{e}/{rid}/artifacts/{name}", data=body.encode(), timeout=60).raise_for_status()
    c.call("POST", "runs/update", json=dict(run_id=rid, status="FINISHED", end_time=int(time.time() * 1000)))
    ver = None
    if register:
        try:
            c.call("POST", "registered-models/create", json=dict(name=REGISTERED, description="칩 · 모델 → 커스텀 sparse MLP 사용 여부 · 타일 설정 · 예상 E2E 이득"))
        except RuntimeError as er:
            if "ALREADY_EXISTS" not in str(er):
                raise
        ver = c.call("POST", "model-versions/create", json=dict(name=REGISTERED, source=f"mlflow-artifacts:/{e}/{rid}/artifacts/selector", run_id=rid,
                                                                  description=f"{sel.meta['version']} gate={'pass' if sel.meta['gate_ok'] else 'fail'}"))["model_version"]["version"]
        c.call("POST", "registered-models/alias", json=dict(name=REGISTERED, alias="candidate", version=ver))
        if sel.meta["gate_ok"]:
            c.call("POST", "registered-models/alias", json=dict(name=REGISTERED, alias="champion", version=ver))
    return rid, ver


def model_specs(args):
    """인자: shapes.yml 이름 또는 models/*.yml (timm 이름, shape, chips)."""
    out = []
    for a in args:
        if a.endswith((".yml", ".yaml")):
            y = yaml.safe_load(open(a))
            out.append((y.get("name") or Path(a).stem, y.get("shape") or y.get("name"), y.get("chips") or F.CHIPS))
        else:
            out.append((a, a, F.CHIPS))
    return out


def fmt_md(preds):
    icon = {"custom": "커스텀", "trt": "TRT 기본", "measure": "측정 필요"}
    L = ["| 모델 | 칩 | 판정 | 설정 (fc1_fc2_stage) | 예상 E2E % (구간) | 근거 |", "|---|---|---|---|---:|---|"]
    for name, p in preds:
        g = "" if p["expected_gain"] is None else f"{p['expected_gain']:+.1f}" + (f" ({p['interval'][0]:+.1f} ~ {p['interval'][1]:+.1f})" if p["interval"] else "")
        L.append(f"| {name} | {p['chip']} | **{icon[p['action']]}** | `{p.get('config', '')}` | {g} | {p['reason']} |")
    return "\n".join(L) + "\n\n음수가 TRT 대비 빠름. 측정 필요 = 예측 상위 설정과 TRT 를 둘 다 재서 빠른 쪽을 쓴다.\n"


def main(argv=None):
    ap = argparse.ArgumentParser(prog="selector")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("dataset"); a.add_argument("--out", default="selector_dataset.csv")
    a = sub.add_parser("train"); a.add_argument("--data"); a.add_argument("--out", default=str(MODEL)); a.add_argument("--log", action="store_true")
    a.add_argument("--register", action="store_true"); a.add_argument("--report")
    a = sub.add_parser("predict"); a.add_argument("models", nargs="+"); a.add_argument("--chip", default="all"); a.add_argument("--model-file", default=str(MODEL))
    a.add_argument("--format", default="md", choices=["md", "json", "csv"])
    a = sub.add_parser("register"); a.add_argument("spec"); a.add_argument("--model-file", default=str(MODEL)); a.add_argument("--out")
    a = sub.add_parser("score"); a.add_argument("pred"); a.add_argument("--stage", default=""); a.add_argument("--data")
    a = sub.add_parser("drift"); a.add_argument("--model-file", default=str(MODEL)); a.add_argument("--data"); a.add_argument("--max-regret", type=float, default=0.1)
    a.add_argument("--max-worst", type=float, default=2.0)
    x = ap.parse_args(argv)

    if x.cmd == "dataset":
        R = D.from_mlflow(_uri()); D.save_csv(R, x.out); print(len(R), "점 →", x.out)
    elif x.cmd == "train":
        R = _rows(x.data); sel, rep = train(R); md = report_md(rep, sel.meta)
        Path(x.out).parent.mkdir(parents=True, exist_ok=True); sel.save(x.out); print(md)
        if x.report:
            Path(x.report).write_text(md)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(md)
        if x.log or x.register:
            rid, ver = log_mlflow(sel, rep, md, x.register); print("MLflow run", rid, "등록 버전", ver)
        return 0 if sel.meta["gate_ok"] else 1
    elif x.cmd == "predict":
        sel = Selector.load(x.model_file); out = []
        for name, shape, chips in model_specs(x.models):
            for c in (F.CHIPS if x.chip == "all" else x.chip.split(",")):
                if c in chips or x.chip != "all":
                    out.append((name, sel.predict(c, shape)))
        if x.format == "md":
            print(fmt_md(out))
        elif x.format == "json":
            print(json.dumps([dict(model=n, **p) for n, p in out], ensure_ascii=False, indent=1))
        else:
            w = csv.writer(sys.stdout); w.writerow(["model", "chip", "action", "config", "expected_gain", "lo", "hi"])
            for n, p in out:
                w.writerow([n, p["chip"], p["action"], p.get("config", ""), p["expected_gain"], *(p["interval"] or ["", ""])])
    elif x.cmd == "register":
        spec = yaml.safe_load(open(x.spec)); sel = Selector.load(x.model_file)
        models = [l.split()[0] for l in open(Path(x.spec).parent / spec["models"]) if l.strip() and not l.startswith("#")]
        out = Path(x.out or ROOT / "experiments/records/predictions" / f"{spec['name']}_selector.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["chip", "model", "action", "config", "expected_gain", "lo", "hi", "selector_version"])
            for c in spec["targets"]:
                for m in models:
                    p = sel.predict(c, m)
                    w.writerow([c, m, p["action"], p.get("config", ""), p["expected_gain"], *(p["interval"] or ["", ""]), sel.meta.get("version", "")])
        print("등록 파일", out, "— 측정 전에 커밋한다")
    elif x.cmd == "score":
        R = {(r["chip"], r["model"]): r for r in _rows(x.data) if not x.stage or r["src"].startswith(x.stage)}
        P = list(csv.DictReader(open(x.pred))); res = []
        for p in P:
            r = R.get((p["chip"], p["model"]))
            if r is None:
                continue
            res.append(dict(chip=p["chip"], model=p["model"], action=p["action"], gain=r["gain"], regret=regret(p["action"], r["gain"]),
                            err=None if p["expected_gain"] in ("", "None") else abs(float(p["expected_gain"]) - r["gain"])))
        if not res:
            sys.exit("채점할 측정이 없다")
        e = [z["err"] for z in res if z["err"] is not None]
        s = dict(n=len(res), of=len(P), measure=sum(z["action"] == "measure" for z in res), regret=round(sum(z["regret"] for z in res) / len(res), 3),
                 worst=round(max(z["regret"] for z in res), 2), mae=round(sum(e) / len(e), 2) if e else None)
        print(json.dumps(dict(summary=s, rows=res), ensure_ascii=False, indent=1))
    elif x.cmd == "drift":
        sel = Selector.load(x.model_file); R = _rows(x.data)
        reg = [regret(sel.policy.decide(r["chip"], r, sel.gain.predict(r))[0], r["gain"]) for r in R]
        new = [r for r in R if r["src"] not in sel.meta.get("stages", [])]
        s = dict(n=len(R), new_points=len(new), regret=round(sum(reg) / len(reg), 3), worst=round(max(reg), 2), model=sel.meta.get("version"))
        print(json.dumps(s, ensure_ascii=False))
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            open(os.environ["GITHUB_STEP_SUMMARY"], "a").write(f"### 선택기 드리프트\n\n```\n{json.dumps(s, ensure_ascii=False)}\n```\n")
        return 1 if s["regret"] > x.max_regret or s["worst"] > x.max_worst else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
