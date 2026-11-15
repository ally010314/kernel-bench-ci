#!/usr/bin/env python3
"""mlflow_log.py — results.json 을 MLflow 에 run 하나로 기록

mlflow 패키지 없이 REST API 만 쓴다 (Jetson 디스크 절약: requests 만 필요).
  experiment : kernel-bench
  run name   : <device>@<sha>
  tags       : sha, device, branch, event, run_url, gpu, model, kernel
  params     : perf_tol, gpu_clock_mhz, clock_locked
  metrics    : fused_ms/N{n}, dense_ms/N{n}, speedup/N{n}, max_err/N{n}, mean_err/N{n}, delta/N{n},
               fc1_ms/N{n}, fc2_ms/N{n} (하네스가 분리 측정을 낸 경우)
               timestamp = 커밋 시각 (Grafana 에서 커밋 순서대로 보이게)

env: MLFLOW_TRACKING_URI (필수), MLFLOW_TRACKING_USERNAME / MLFLOW_TRACKING_PASSWORD (선택, basic auth)
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

EXPERIMENT = "kernel-bench"


class Client:
    def __init__(self, uri):
        self.base = uri.rstrip("/") + "/api/2.0/mlflow"
        self.s = requests.Session()
        u, p = os.environ.get("MLFLOW_TRACKING_USERNAME"), os.environ.get("MLFLOW_TRACKING_PASSWORD")
        if u and p:
            self.s.auth = (u, p)

    def call(self, method, path, **kw):
        r = self.s.request(method, f"{self.base}/{path}", timeout=15, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"{method} {path}: {r.status_code} {r.text[:300]}")
        return r.json()

    def experiment_id(self, name):
        try:
            return self.call("GET", "experiments/get-by-name", params={"experiment_name": name})["experiment"]["experiment_id"]
        except RuntimeError as e:
            if "RESOURCE_DOES_NOT_EXIST" not in str(e):
                raise
            return self.call("POST", "experiments/create", json={"name": name})["experiment_id"]


def commit_time_ms(root):
    try:
        return int(subprocess.check_output(["git", "log", "-1", "--format=%ct"], cwd=root, text=True).strip()) * 1000
    except Exception:
        return int(time.time() * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="build/results.json")
    ap.add_argument("--timestamp", help="metric 시각 (ISO 날짜/시각). 기본: HEAD 커밋 시각. 과거 버전 재생용")
    ap.add_argument("--tag", action="append", default=[], help="추가 태그 key=value (반복 가능)")
    a = ap.parse_args()

    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        print("MLFLOW_TRACKING_URI 미설정 — 기록 생략")
        return 0
    res = json.loads(Path(a.results).read_text())
    root = Path(__file__).resolve().parent.parent
    if a.timestamp:
        from datetime import datetime
        ts = int(datetime.fromisoformat(a.timestamp).timestamp() * 1000)
    else:
        ts = commit_time_ms(root)

    c = Client(uri)
    exp = c.experiment_id(EXPERIMENT)
    env = os.environ
    run_url = (f"{env['GITHUB_SERVER_URL']}/{env['GITHUB_REPOSITORY']}/actions/runs/{env['GITHUB_RUN_ID']}"
               if "GITHUB_RUN_ID" in env else "")
    first = next(iter(res["results"].values()))
    tags = {
        "sha": res["sha"],
        "device": res["device"],
        "gpu": first["gpu"],
        "branch": env.get("GITHUB_HEAD_REF") or env.get("GITHUB_REF_NAME", "local"),
        "event": env.get("GITHUB_EVENT_NAME", "local"),
        "run_url": run_url,
        "mlflow.runName": f"{res['device']}@{res['sha']}",
    }
    tags.update({k: res[k] for k in ("model", "kernel") if res.get(k)})   # kernel=paper 부터 논문 커널 측정
    tags.update(kv.split("=", 1) for kv in a.tag)
    run = c.call("POST", "runs/create", json={
        "experiment_id": exp, "start_time": ts, "run_name": tags["mlflow.runName"],
        "tags": [{"key": k, "value": v} for k, v in tags.items()]})["run"]["info"]
    rid = run["run_id"]

    dev = res.get("device_info", {})
    params = {"perf_tol": res["perf_tol"], "gpu_clock_mhz": dev.get("gpu_clock_mhz", ""),
              "clock_locked": dev.get("clock_locked", "")}
    metrics = []
    for n, r in res["results"].items():
        for key, src in (("fused_ms", "fused_ms"), ("dense_ms", "dense_ms"), ("fc1_ms", "fc1_ms"), ("fc2_ms", "fc2_ms"), ("speedup", "speedup_vs_dense"),
                         ("max_err", "max_abs_err"), ("mean_err", "mean_abs_err"), ("delta", "delta")):
            if src in r:
                metrics.append({"key": f"{key}/N{n}", "value": float(r[src]), "timestamp": ts, "step": 0})
    c.call("POST", "runs/log-batch", json={
        "run_id": rid, "metrics": metrics,
        "params": [{"key": k, "value": str(v)} for k, v in params.items()]})
    status = "FINISHED" if all(r.get("acc_ok", True) and r.get("perf_ok", True) for r in res["results"].values()) else "FAILED"
    c.call("POST", "runs/update", json={"run_id": rid, "status": status, "end_time": int(time.time() * 1000)})
    print(f"MLflow 기록: experiment={EXPERIMENT} run={tags['mlflow.runName']} ({len(metrics)} metrics, {status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
