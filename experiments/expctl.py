#!/usr/bin/env python3
"""expctl.py — YAML 명세로 네 장비의 실험을 검증 · 사전 점검 · 실행 · 판정한다.

  validate <spec...>              명세 형식과 장비 지원 범위 확인 (GPU 안 씀)
  preflight <spec> [--only d,..]  GPU 사용률 · 클럭 · 다른 사용자 · 디스크 · 커널 소스 · 모델 1개 시험
  run <spec> [--only d,..]        사전 점검을 통과하면 장비별 worker 를 띄운다
  judge [--issue]                 실행한 실험을 판정한다. 실패하면 GitHub issue 를 한 번 연다

시험용: --inject NO_FP16_FLAG=1, --no-support-check, --skip-preflight.
"""
import argparse, hashlib, json, os, re, shlex, subprocess as sp, sys, time
from pathlib import Path
import yaml

ME = Path(__file__).resolve().parent; TOOLS = ME / "tools"
sys.path.insert(0, str(TOOLS))
import tiles
STATE = Path(os.environ.get("MLOPS_CACHE", Path.home() / ".cache/kernel-bench-collect")) / "experiments"
DEV = yaml.safe_load(open(ME / "devices.yml"))
UTIL_MAX, SM_TOL, SHM_MIN_GB, HOME_MIN_GB, TILES_MIN = 5, 15, 8, 1, 24
SERVICE_MIB = 500   # 이보다 작은 프로세스는 상시 서비스로 보고 막지 않는다
STALL_H = 5         # 이 시간 넘게 모델 하나도 끝나지 않으면 멈춘 것으로 본다. worker 의 모델 시간 제한보다 길다
SMOKE = ("deit_small_patch16_224", "384")


def sh(dev, cmd, timeout=120):
    """장비에서 명령을 실행한다. (rc, 출력)"""
    host = DEV[dev]["host"]
    a = ["bash", "-c", cmd] if host is None else ["ssh", "-p", "<port>", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"<user>@{host}.<domain>", cmd]
    try:
        r = sp.run(a, capture_output=True, text=True, timeout=timeout)
    except sp.TimeoutExpired:
        return 124, "timeout"
    return r.returncode, r.stdout + r.stderr


def load_spec(path):
    p = Path(path)
    if not p.exists():
        p = ME / "specs" / f"{path}.yaml"
    s = yaml.safe_load(open(p))
    s["_models"] = [l.split() for l in open(p.parent / s["models"]) if l.strip()]
    return s


def validate(s, support=True):
    """명세 오류 목록을 돌려준다."""
    e = []
    for k in ("name", "models", "fixed_tile", "targets"):
        if k not in s:
            e.append(f"{k} 없음")
    if e:
        return e
    if not re.fullmatch(r"[a-z0-9-]+", s["name"]):
        e.append(f"name 형식: {s['name']}")
    if not s["_models"] or any(len(m) != 3 for m in s["_models"]):
        e.append("모델 파일 형식: <key> <timm 이름> <minDim>")
    if s.get("type") == "stairs" and ("tiles" not in s or not s.get("stairs", {}).get("reps") or not s["stairs"].get("nmax")):
        e.append("stairs 는 tiles · stairs.reps · stairs.nmax 필요")
    if "tiles" in s:
        te = tiles.check(s["tiles"], s.get("flags"))
        e += te
        if not te and s["fixed_tile"] not in tiles.names(s["tiles"]):
            e.append(f"고정 타일 {s['fixed_tile']} 이 tiles 후보에 없음")
    for d, t in s["targets"].items():
        if d not in DEV:
            e.append(f"모르는 장비 {d}")
            continue
        for v in t.get("trt", []):
            if support and v not in DEV[d]["trt"]:
                e.append(f"{d} 는 TRT {v} 미지원 ({DEV[d]['trt']})")
        if "tiles" in s and "build" not in DEV[d]:
            e.append(f"{d} 는 빌드 설정 없음")
        bad = set() if t.get("gpus") == "auto" else set(t.get("gpus", [])) - set(DEV[d]["gpus"])
        if bad:
            e.append(f"{d} 허용되지 않은 GPU {sorted(bad)}")
        if not t.get("trt") or not t.get("gpus"):
            e.append(f"{d} trt · gpus 필요")
    return e


def judge_gpu(rows, others, want, gpus):
    """nvidia-smi 값으로 GPU 를 판정한다."""
    e = []
    for g in gpus:
        if g not in rows:
            e.append(f"GPU {g} 없음")
            continue
        sm, util = rows[g]
        if util > UTIL_MAX:
            e.append(f"GPU {g} 사용 중 {util}%")
        if abs(sm - want) > SM_TOL:
            e.append(f"GPU {g} 클럭 {sm} MHz, 기대 {want}")
        if g in others:
            e.append(f"GPU {g} 다른 사용자 {sorted(others[g])}")
    return e


GPU_Q = r"""
nvidia-smi --query-gpu=index,clocks.sm,utilization.gpu,clocks.applications.graphics --format=csv,noheader,nounits
echo ---
for u in $(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits | tr -d ' '); do
  IFS=, read id p mem <<< "$u"; g=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | tr -d ' ' | grep "$id" | cut -d, -f1)
  echo "$g $(ps -o user= -p $p) $mem"
done
"""


OWN_Q = r"""
for p in $(pgrep -u <user> 'python|nsys'); do
  c=$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null); case "$c" in *hold*|*expctl*|*jupyter*|*uvicorn*) continue;; esac
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep '^CUDA_VISIBLE_DEVICES=' | cut -d= -f2
done | sort -u
"""


def own_busy(dev):
    """자기 측정이 잡은 GPU. 변환 · 빌드 구간은 사용률이 0% 라 사용률만으로는 못 본다. 메모리 예약(hold)은 제외한다."""
    if DEV[dev]["host"] is None:
        rc, o = sh(dev, "pgrep -u <user> -f '[r]un2\\.py|[p]rof/prof\\.py|[p]rof/tac\\.py|[w]orker\\.py spec' | wc -l")
        return {0} if o.strip().isdigit() and int(o) > 0 else set()
    rc, o = sh(dev, OWN_Q)
    return {int(g) for x in o.split() for g in x.split(",") if g.isdigit()}


def check_gpu(dev, gpus):
    own = own_busy(dev) & set(gpus)
    e = [f"GPU {g} 자기 측정 진행 중" for g in sorted(own)]
    return e + read_gpu(dev, gpus)


def read_gpu(dev, gpus):
    if DEV[dev]["host"] is None:
        rc, o = sh(dev, "G=/sys/class/devfreq/17000000.gpu; cat $G/min_freq $G/max_freq; cat /sys/devices/platform/gpu.0/load 2>/dev/null || echo 0")
        v = o.split()
        if rc or len(v) < 3:
            return [f"클럭 읽기 실패: {o[:80]}"]
        e = [] if v[0] == v[1] else ["클럭 미고정"]
        if int(int(v[1]) / 1e6) != DEV[dev]["sm_mhz"]:
            e.append(f"클럭 {int(v[1]) / 1e6:.0f} MHz, 기대 {DEV[dev]['sm_mhz']}")
        if int(v[2]) / 10 > UTIL_MAX:
            e.append(f"GPU 사용 중 {int(v[2]) / 10:.0f}%")
        return e
    rc, o = sh(dev, GPU_Q)
    if rc:
        return [f"nvidia-smi 실패: {o[:80]}"]
    a, _, b = o.partition("---")
    rows = {}
    for x in (l.replace(" ", "").split(",") for l in a.strip().splitlines()):
        # 쉬는 GPU 의 현재 클럭은 낮게 읽힌다. 고정값(applications clock)이 있으면 그것을 본다.
        rows[int(x[0])] = (int(x[3]) if x[3].isdigit() else int(x[1]), int(x[2]))
    others = {}
    for l in b.strip().splitlines():
        p = l.split()
        if len(p) == 3 and p[1] != "<user>" and int(p[2]) >= SERVICE_MIB:
            others.setdefault(int(p[0]), set()).add(p[1])
    return judge_gpu(rows, others, DEV[dev]["sm_mhz"], gpus)


def locked_gpus(dev):
    rc, o = sh(dev, f"ls ~/exp/locks 2>/dev/null | grep '^{dev}-gpu'")
    return {int(x.rsplit("gpu", 1)[1]) for x in o.split()}


def pick_gpus(dev, want):
    """사용률 · 클럭 · 다른 사용자 · 잠금을 통과한 GPU 를 고른다. k3s 와 달리 ssh 로 직접 쓰는 사용자도 보인다."""
    lk = locked_gpus(dev)
    free = [g for g in DEV[dev]["gpus"] if g not in lk and not check_gpu(dev, [g])]
    return free[:want] if want else free


def resolve_gpus(s, only):
    """명세의 gpus: auto 를 실제 번호로 바꾼다. 빈 GPU 가 없는 장비를 돌려준다."""
    none = []
    for dev, t in s["targets"].items():
        if t["gpus"] == "auto" and (not only or dev in only):
            t["gpus"] = pick_gpus(dev, t.get("max_gpus"))
            print(f"{dev} 자동 선택 GPU {t['gpus']}", flush=True)
            if not t["gpus"]:
                none.append(dev)
    return none


def check_host(dev, trt, gpus, tile, need_tiles=True):
    e = []
    rc, o = sh(dev, "df -BG --output=avail /dev/shm | tail -1; df -BG --output=avail ~ | tail -1")
    v = [int(x.strip().rstrip("G")) for x in o.split()] if rc == 0 else []
    if len(v) == 2:
        if v[0] < SHM_MIN_GB:
            e.append(f"/dev/shm 여유 {v[0]} GB")
        if v[1] < HOME_MIN_GB:
            e.append(f"홈 여유 {v[1]} GB")
    else:
        e.append("디스크 읽기 실패")
    for v_ in trt:
        d = DEV[dev]["sodir"].format(trt=v_)
        rc, o = sh(dev, f"ls {d}/*/p.so 2>/dev/null | wc -l; ls {d}/{tile}/p.so 2>&1 | wc -l")
        n = o.split()
        if rc or not n or (need_tiles and int(n[0]) < TILES_MIN):
            e.append(f"TRT {v_} 타일 {n[0] if n else '?'}개, 최소 {TILES_MIN}")
    rc, o = sh(dev, f"for g in {' '.join(map(str, gpus))}; do cat ~/exp/locks/{dev}-gpu$g/owner 2>/dev/null; done")
    if o.strip():
        e.append(f"GPU 잠금 {o.strip()}")
    return e


def kernel_hash(dev):
    rc, o = sh(dev, f"md5sum {DEV[dev]['kernel']}")
    return o.split()[0][:8] if rc == 0 and o else "?"


def repo_kernel_hash():
    return hashlib.md5((ME.parent / "kernels/paper/fused_mlp.cu").read_bytes()).hexdigest()[:8]


def kernel_mismatch(hashes, repo):
    """측정 커널이 저장소 커널과 다른 장비. 어긋나면 결과를 재현할 수 없고 CI 가 다른 커널을 지키게 된다."""
    return {d: h for d, h in hashes.items() if h != repo}


def deploy(dev, eid):
    """tools 와 저장소 커널을 실험 폴더로 복사한다. 폴더는 실험마다 따로라 임시 파일이 겹치지 않는다."""
    d = f"exp/{eid}/{dev}"
    tar = sp.run(["tar", "czf", "-", "-C", str(TOOLS), ".", "-C", str(ME.parent), "kernels", "bench"], capture_output=True).stdout
    host = DEV[dev]["host"]
    cmd = f"mkdir -p ~/{d} && tar xzf - -C ~/{d}"
    a = ["bash", "-c", cmd] if host is None else ["ssh", "-p", "<port>", "-o", "BatchMode=yes", f"<user>@{host}.<domain>", cmd]
    sp.run(a, input=tar, check=True, timeout=60)
    return d


def smoke(dev, eid, d, v, gpu, tile, inject):
    so = DEV[dev]["sodir"].format(trt=v) + f"/{tile}/p.so"
    env = " ".join(f"{k}={shlex.quote(x)}" for k, x in inject.items())
    pre = f"LD_PREFIX={DEV[dev].get('ld_prefix', '')} EXP_ID={eid}"
    cmd = f"cd ~/{d} && export {pre} {env}; source ./env.sh {v} && $PY smoke.py {gpu} {v} {SMOKE[0]} {SMOKE[1]} $(eval echo {so})"
    rc, o = sh(dev, cmd, timeout=900)
    for l in o.splitlines()[::-1]:
        if l.startswith("{"):
            return json.loads(l)
    return dict(ok=False, stage="smoke", err=o[-200:])


def preflight(s, only, inject, eid, support=True, summary=None):
    rows, ok, hashes = [], True, {}
    for dev, t in s["targets"].items():
        if only and dev not in only:
            continue
        hashes[dev] = kernel_hash(dev)
        errs = check_gpu(dev, t["gpus"]) + check_host(dev, t["trt"], t["gpus"], s["fixed_tile"], "tiles" not in s)
        if not errs:
            d = deploy(dev, eid)
            for v in t["trt"]:
                r = smoke(dev, eid, d, v, t["gpus"][0], s["fixed_tile"], inject)
                if not r["ok"]:
                    errs.append(f"TRT {v} {r['stage']}: {r['err']}")
        rows.append((dev, errs)); ok &= not errs
    bad = {} if "tiles" in s else kernel_mismatch(hashes, repo_kernel_hash())   # 타일 빌드는 저장소 커널을 그대로 쓴다
    if bad:
        rows.append(("커널 소스", [f"저장소 {repo_kernel_hash()} 와 다름 {bad}"])); ok = False
    out = ["| 장비 | 결과 |", "|---|---|"] + [f"| {d} | {'통과' if not e else '실패: ' + '; '.join(e)} |" for d, e in rows]
    print("\n".join(out), flush=True)
    if summary:
        open(summary, "a").write(f"### 사전 점검 {eid}\n\n" + "\n".join(out) + "\n\n")
    return ok


def launch(s, only, inject, eid):
    STATE.mkdir(parents=True, exist_ok=True); rec = dict(id=eid, spec=s["name"], targets=[], at=time.time())
    for dev, t in s["targets"].items():
        if only and dev not in only:
            continue
        d = deploy(dev, eid)
        job = dict(id=eid, dev=dev, trt=t["trt"], gpus=t["gpus"], models=s["_models"], fixed_tile=s["fixed_tile"],
                   fail_fast=s.get("fail_fast", 2), ld_prefix=DEV[dev].get("ld_prefix"), inject=inject,
                   hold_gb=DEV[dev].get("hold_gb", 0), keep_gb=DEV[dev].get("keep_gb", 12), model_timeout_h=s.get("model_timeout_h", 4))
        if "tiles" in s:
            job.update(tiles=tiles.names(s["tiles"]), flags=s.get("flags", tiles.FLAGS), build=DEV[dev]["build"], type=s.get("type", "tiles"), stairs=s.get("stairs")); links = "true"
        else:
            links = "; ".join(f"mkdir -p ~/{d}/sv; ln -sfn $(eval echo {DEV[dev]['sodir'].format(trt=v)}) ~/{d}/sv/{v}" for v in t["trt"])
        js = shlex.quote(json.dumps(job))
        # RUNNER_TRACKING_ID 를 비운다. Actions runner 는 job 이 끝날 때 이 값을 가진 프로세스를 고아로 보고 죽인다.
        # 서버는 ssh 로 떠서 상관없지만, runner 와 같은 기계 (Orin) 의 worker 는 setsid 로도 피하지 못했다 (2026-11-08).
        rc, o = sh(dev, f"{links}; echo {js} > ~/{d}/spec.json; cd ~/{d}; RUNNER_TRACKING_ID= nohup setsid /usr/bin/python3 worker.py spec.json > worker.log 2>&1 < /dev/null & sleep 6; cat ~/{d}/status.json")
        if '"state"' not in o:
            sys.exit(f"{dev} worker 시작 실패: {o[-200:]}")
        rec["targets"].append(dev); print(f"{dev} 시작: {d}", flush=True)
    json.dump(rec, open(STATE / f"{eid}.json", "w"))


def classify(st, rows, alive, idle=0):
    """status.json · 결과 줄 수 · 프로세스 생존 · 마지막 진행 후 시간으로 상태를 정한다. 로그의 완료 문구는 보지 않는다."""
    if st is None:
        return "LOST"
    if st["state"] == "RUNNING":
        if not alive:
            return "LOST"
        return "STALLED" if idle > STALL_H * 3600 else "RUNNING"
    if st["state"] == "SUCCEEDED" and rows < st["expected"]:
        return "PARTIAL"
    return st["state"]


def judge(issue, summary=None):
    lines = ["| 실험 | 장비 | 상태 | 성공 / 전체 | 실패 사유 |", "|---|---|---|---:|---|"]
    bad = []
    for f in sorted(STATE.glob("*.json")):
        rec = json.load(open(f)); done = True
        for dev in rec["targets"]:
            d = f"exp/{rec['id']}/{dev}"
            rc, o = sh(dev, f"cat ~/{d}/status.json 2>/dev/null; echo; cat ~/{d}/res/*/summary.jsonl 2>/dev/null | wc -l")
            p = o.strip().splitlines()
            st = json.loads(p[0]) if p and p[0].startswith("{") else None
            n = int(p[-1]) if p and p[-1].isdigit() else 0
            alive = bool(st) and sh(dev, f"kill -0 {st['pid']} 2>/dev/null && echo y")[1].strip() == "y"
            c = classify(st, n, alive, time.time() - st["updated"] if st else 0)
            done &= c not in ("RUNNING", "STALLED")
            why = "; ".join(((st or {}).get("fails") or [])[:3]) or (st or {}).get("reason", "")
            skipped = sorted(set((st or {}).get("build_failed") or []) | set((st or {}).get("tile_failed") or {}))
            if skipped:
                why = (why + " " if why else "") + f"제외 타일 {len(skipped)}개: {', '.join(skipped[:4])}"
            lines.append(f"| {rec['id']} | {dev} | {c} | {(st or {}).get('ok', 0)} / {(st or {}).get('expected', '?')} | {why[:120]} |")
            if c in ("FAILED", "PARTIAL", "LOST", "STALLED"):
                bad.append((rec["id"], dev, c, why))
        if done and not rec.get("final"):
            rec["final"] = time.time(); json.dump(rec, open(f, "w"))
    print("\n".join(lines))
    if summary:
        open(summary, "a").write("### 실험 상태\n\n" + "\n".join(lines) + "\n\n")
    for eid, dev, c, why in bad:
        mark = STATE / f".alerted-{eid}-{dev}-{c}"
        print(f"::warning::{eid} {dev} {c}: {why[:160]}")
        if issue and not mark.exists():
            r = sp.run(["gh", "issue", "create", "--title", f"[실험 {c}] {eid} · {dev}", "--body", f"상태: {c}\n\n사유: {why}\n\n결과 폴더: `~/exp/{eid}/{dev}`"], capture_output=True, text=True)
            if r.returncode == 0:
                mark.touch()
            print(r.stdout + r.stderr)
    return 0


def main():
    a = argparse.ArgumentParser(); a.add_argument("cmd", choices=["validate", "preflight", "run", "judge"]); a.add_argument("specs", nargs="*")
    a.add_argument("--only", default=""); a.add_argument("--inject", action="append", default=[]); a.add_argument("--no-support-check", action="store_true"); a.add_argument("--skip-preflight", action="store_true")
    a.add_argument("--issue", action="store_true"); a.add_argument("--id")
    x = a.parse_args(); summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if x.cmd == "judge":
        sys.exit(judge(x.issue, summary))
    specs = x.specs or [p.stem for p in (ME / "specs").glob("*.yaml")]; rc = 0
    only = [o for o in x.only.split(",") if o]
    if only and x.cmd != "validate":
        for p in specs:
            miss = set(only) - set(load_spec(p)["targets"])
            if miss:
                sys.exit(f"{p} 대상이 아님: {sorted(miss)}")
    inject = dict(i.split("=", 1) for i in x.inject)
    for p in specs:
        s = load_spec(p); e = validate(s, not x.no_support_check)
        print(p, "명세 오류: " + "; ".join(e) if e else "명세 정상", flush=True)
        if e:
            rc = 1; continue
        if x.cmd == "validate":
            continue
        none = resolve_gpus(s, only)
        if none:
            sys.exit(f"빈 GPU 없음: {none}")
        eid = x.id or f"{s['name']}-{time.strftime('%m%d-%H%M')}"
        if not x.skip_preflight and not preflight(s, only, inject, eid, not x.no_support_check, summary):
            sys.exit(1)
        if x.cmd == "run":
            launch(s, only, inject, eid)
    sys.exit(rc)


if __name__ == "__main__":
    main()
