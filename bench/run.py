#!/usr/bin/env python3
"""run.py — FusedMlpL 정확도 + latency 벤치 (CI 진입점)

흐름
  1. 고정 seed 로 fc1[4096,1024] / fc2[1024,4096] 가중치 생성 → 2:4 magnitude prune → pack
     (논문 커널이 data/allblk/vitL_b00_* 를 읽는다)
  2. N 별 입력 x 생성 → build/mlp_bench 실행 (출력 y + latency JSON)
  3. 정확도: **같은 pruned 가중치**의 fp32 dense 기준값과 비교
     (원 dense 모델과 비교하면 pruning 오차가 섞여 커널 회귀를 못 가린다)
  4. 성능: baselines/<device>.json 대비 median latency 가 tol 넘게 느려지면 실패

exit code: 0 통과 / 1 정확도 실패 / 2 성능 회귀 / 3 실행 오류
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
# 모델: (hidden, dim, 데이터 접두, 기본 N). 가중치는 합성이라 실모델 파일이 필요 없다.
MODELS = {
    "": (4096, 1024, "vitL", "49,100,196,197,400,1568,3136"),
    "mixerL": (4096, 1024, "vitL", "49,100,196,197,400,1568,3136"),
    "vitL": (4096, 1024, "vitL", "49,100,196,197,400,1568,3136"),
    "mixerB": (3072, 768, "mixerB", "49,100,196,197,400,784"),
    "resmlpB": (3072, 768, "resmlpB", "196,400,784,800,1568"),
    "deitS": (1536, 384, "deitS", "49,100,196,197,400,784"),
}
M1, K1, M2, K2 = 4096, 1024, 1024, 4096
PREFIX = "vitL"

# 정확도 기준 (fp16 출력, H 를 fp16 으로 반올림한 기준값 대비)
#   정상 커널 실측: max ~9e-4, mean ~6e-5 (Orin, seed 0) → 약 5배 여유.
#   fc2 meta 64워드 1비트 오염 시 max 3.6e-1 → 검출됨 (mean 은 1.4e-4 라 max 가 주 판정).
MAX_ABS_TOL = 5e-3
MEAN_ABS_TOL = 3e-4


def prune_24(W):
    """K 축 4개 그룹마다 절댓값 상위 2개만 남김."""
    M, K = W.shape
    g = W.reshape(M, K // 4, 4)
    idx = np.argsort(-np.abs(g), axis=-1)[..., 2:]
    g = g.copy()
    np.put_along_axis(g, idx, 0, axis=-1)
    return g.reshape(M, K)


def pack_24(W):
    """W[M,K] 2:4 → packed[M,K/2] fp16, meta[M,K/16] uint16 (row-major). 11_extract_all.py 와 동일."""
    M, K = W.shape
    g4 = W.reshape(M, K // 4, 4)
    nz = g4 != 0
    assert (nz.sum(-1) <= 2).all(), "not 2:4"
    ar = np.arange(4)
    order = np.argsort(np.where(nz, ar, ar + 4), axis=-1)
    keep = np.sort(order[..., :2], axis=-1)
    i0, i1 = keep[..., 0].astype(np.int64), keep[..., 1].astype(np.int64)
    v0 = np.take_along_axis(g4, i0[..., None], -1)[..., 0]
    v1 = np.take_along_axis(g4, i1[..., None], -1)[..., 0]
    packed = np.stack([v0, v1], -1).reshape(M, K // 2).astype(np.float16)
    nib = ((i0 & 0x3) | ((i1 & 0x3) << 2)).astype(np.uint16)
    meta = (nib.reshape(M, K // 16, 4) << np.array([0, 4, 8, 12], dtype=np.uint16)).sum(-1).astype(np.uint16)
    return packed, meta


def erf(x):
    """math.erf 벡터화 (scipy 의존 회피). CUTLASS GELU 도 erf 판."""
    return np.frompyfunc(math.erf, 1, 1)(x).astype(np.float32)


def gen_weights(work, seed):
    rng = np.random.default_rng(seed)
    data = work / "data"
    data.mkdir(parents=True, exist_ok=True)
    W = {}
    for fc, (M, K) in (("fc1", (M1, K1)), ("fc2", (M2, K2))):
        w = prune_24((rng.standard_normal((M, K)) * 0.02).astype(np.float16))
        b = (rng.standard_normal(M) * 0.02).astype(np.float16)
        p, m = pack_24(w)
        allblk = data / "allblk"
        allblk.mkdir(exist_ok=True)
        p.tofile(allblk / f"{PREFIX}_b00_{fc}_packed.bin")
        m.tofile(allblk / f"{PREFIX}_b00_{fc}_meta.bin")
        b.tofile(allblk / f"{PREFIX}_b00_{fc}_bias.bin")
        W[fc] = (w.astype(np.float32), b.astype(np.float32))
    return W


def reference(x, W):
    w1, b1 = W["fc1"]
    w2, b2 = W["fc2"]
    h = x.astype(np.float32) @ w1.T + b1
    h = 0.5 * h * (1.0 + erf(h / math.sqrt(2.0)))
    h = h.astype(np.float16).astype(np.float32)  # 커널은 H 를 fp16 으로 저장
    return h @ w2.T + b2


def device_info():
    info = {"host": os.uname().nodename}
    dt = Path("/proc/device-tree/model")
    if dt.exists():
        info["model"] = dt.read_text().strip("\x00\n ")
    # 서버 GPU (nvidia-smi): 측정하는 GPU 의 SM 클럭 · 사용률 기록 (CUDA_VISIBLE_DEVICES 첫 번째 = 하네스가 쓰는 GPU)
    try:
        idx = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
        q = subprocess.check_output(["nvidia-smi", "-i", idx, "--query-gpu=clocks.sm,utilization.gpu,uuid",
                                     "--format=csv,noheader,nounits"], text=True, timeout=10).strip().split(", ")
        info.update(gpu_clock_mhz=float(q[0]), gpu_util_pct=float(q[1]), gpu_uuid=q[2])
    except Exception:
        pass
    gpu = Path("/sys/class/devfreq/17000000.gpu")
    if gpu.exists():
        cur, lo, hi = (int((gpu / f).read_text()) for f in ("cur_freq", "min_freq", "max_freq"))
        info["gpu_clock_mhz"] = cur / 1e6
        info["clock_locked"] = lo == hi
    return info


def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default=None, help="토큰 수 N. 기본은 모델별 (MODELS)")
    ap.add_argument("--model", default=os.environ.get("MODEL", ""), help="커널 config 이름. baseline · 기록 이름에 붙는다")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perf-tol", type=float, default=0.05, help="baseline 대비 허용 느려짐 비율")
    ap.add_argument("--baseline-dir", default=str(ROOT / "baselines"))
    ap.add_argument("--update-baseline", action="store_true", help="이번 결과를 baseline 으로 저장")
    ap.add_argument("--work", default=str(ROOT / "build" / "work"))
    ap.add_argument("--out", default=str(ROOT / "build" / "results.json"))
    ap.add_argument("--summary", default=os.environ.get("GITHUB_STEP_SUMMARY"))
    ap.add_argument("--bench", default=str(ROOT / "build" / "mlp_bench"), help="측정할 하네스 바이너리")
    ap.add_argument("--sha", default=None, help="결과에 기록할 버전 식별자 (기본: git HEAD)")
    a = ap.parse_args()
    global M1, K1, M2, K2, PREFIX
    M1, K1, PREFIX, default_ns = MODELS.get(a.model, MODELS[""])
    M2, K2 = K1, M1
    a.ns = a.ns or default_ns

    bench = Path(a.bench).resolve()
    if not bench.exists():
        print("build/mlp_bench 없음 — ./build.sh 먼저", file=sys.stderr)
        return 3
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    ns = [int(n) for n in a.ns.split(",")]

    W = gen_weights(work, a.seed)
    rng = np.random.default_rng(a.seed + 1)
    xs = {}
    for n in ns:
        xs[n] = rng.standard_normal((n, K1)).astype(np.float16)
        xs[n].tofile(work / f"x_N{n}.bin")

    t0 = time.time()
    p = subprocess.run([str(bench), a.ns, str(a.iters), str(a.warmup)], cwd=work, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout, p.stderr, file=sys.stderr)
        return 3
    perf = {r["N"]: r for r in map(json.loads, p.stdout.strip().splitlines())}

    rows, acc_fail = [], False
    for n in ns:
        y = np.fromfile(work / f"y_N{n}.bin", dtype=np.float16).reshape(n, M2).astype(np.float32)
        ref = reference(xs[n], W)
        d = np.abs(y - ref)
        r = dict(perf[n])
        r.update(max_abs_err=float(d.max()), mean_abs_err=float(d.mean()), ref_absmax=float(np.abs(ref).max()),
                 finite=bool(np.isfinite(y).all()), speedup_vs_dense=perf[n]["dense_ms"] / perf[n]["fused_ms"])
        # 출력이 전부 0 이면 비교가 공허해진다 (RESULTS.md: timm Mixer 랜덤초기화 head=0 사례)
        r["acc_ok"] = r["finite"] and r["max_abs_err"] <= MAX_ABS_TOL and r["mean_abs_err"] <= MEAN_ABS_TOL \
            and float(np.abs(y).max()) > 0
        acc_fail |= not r["acc_ok"]
        rows.append(r)

    dev = device_info()
    gpu = rows[0]["gpu"]
    # Mixer-L 은 기존 장비 이름을 잇는다 (Grafana · baseline 시리즈 유지). 타일이 다른 config 만 이름을 나눈다.
    slug = f"{gpu.lower().replace(' ', '-')}-sm{rows[0]['sm']}" + (f"-{a.model}" if a.model not in ("", "mixerL") else "")
    base_path = Path(a.baseline_dir) / f"{slug}.json"
    base = json.loads(base_path.read_text())["results"] if base_path.exists() else {}

    perf_fail = False
    for r in rows:
        b = base.get(str(r["N"]))
        if b:
            r["baseline_ms"] = b["fused_ms"]
            r["delta"] = r["fused_ms"] / b["fused_ms"] - 1
            r["perf_ok"] = r["delta"] <= a.perf_tol
            perf_fail |= not r["perf_ok"]
            # fc1 / fc2 별 변화는 참고용 (판정은 fused 합계로만 한다)
            for k in ("fc1", "fc2"):
                if b.get(f"{k}_ms", 0) > 0 and r.get(f"{k}_ms", -1) > 0:
                    r[f"{k}_baseline_ms"] = b[f"{k}_ms"]
                    r[f"{k}_delta"] = r[f"{k}_ms"] / b[f"{k}_ms"] - 1

    result = {"sha": a.sha or git_sha(), "device": slug, "model": a.model, "kernel": "paper" if a.model else "", "device_info": dev, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "elapsed_s": round(time.time() - t0, 1), "perf_tol": a.perf_tol,
              "results": {str(r["N"]): r for r in rows}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))

    # ── 보고 ──
    lines = [f"### FusedMlpL bench — `{slug}` @ `{result['sha']}`", ""]
    if dev.get("clock_locked") is False:
        lines += [f"> ⚠️ GPU clock not locked ({dev['gpu_clock_mhz']} MHz) — latency 비교 신뢰도 낮음", ""]
    lines += ["| N | fused ms | dense ms | speedup | baseline ms | Δ | max err | mean err | result |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bl = f"{r['baseline_ms']:.4f}" if "baseline_ms" in r else "—"
        dl = f"{r['delta']*100:+.1f}%" if "delta" in r else "—"
        ok = r["acc_ok"] and r.get("perf_ok", True)
        tag = "✅" if ok else ("❌ acc" if not r["acc_ok"] else "❌ perf")
        lines.append(f"| {r['N']} | {r['fused_ms']:.4f} | {r['dense_ms']:.4f} | {r['speedup_vs_dense']:.3f}x | {bl} | {dl} "
                     f"| {r['max_abs_err']:.2e} | {r['mean_abs_err']:.2e} | {tag} |")
    if not base:
        lines += ["", f"_baseline 없음 ({base_path.name}) — 성능 비교 생략_"]
    if all(r.get("fc1_ms", -1) > 0 for r in rows):
        lines += ["", "**fc1 / fc2 분리** (커널 단독 재실행, 참고용 — 판정은 위 fused 기준)", "",
                  "| N | fc1+GELU ms | Δ fc1 | fc2+전치 ms | Δ fc2 | 기타 ms | 합 / fused |",
                  "|---|---|---|---|---|---|---|"]
        pct = lambda r, k: f"{r[k]*100:+.1f}%" if k in r else "—"
        for r in rows:
            lines.append(f"| {r['N']} | {r['fc1_ms']:.4f} | {pct(r, 'fc1_delta')} | {r['fc2_ms']:.4f} | {pct(r, 'fc2_delta')} "
                         f"| {r['other_ms']:.4f} | {r['split_sum_ms']/r['fused_ms']:.3f} |")
    report = "\n".join(lines)
    print(report)
    if a.summary:
        with open(a.summary, "a") as f:
            f.write(report + "\n")

    if a.update_baseline:
        base_path.parent.mkdir(parents=True, exist_ok=True)
        keep = ("fused_ms", "fused_p10", "fused_p90", "dense_ms", "fc1_ms", "fc2_ms")
        base_path.write_text(json.dumps({"sha": result["sha"], "device_info": dev,
                                         "results": {str(r["N"]): {k: r[k] for k in keep if k in r} for r in rows}}, indent=2) + "\n")
        print(f"baseline 저장: {base_path}")

    if acc_fail:
        return 1
    if perf_fail:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
