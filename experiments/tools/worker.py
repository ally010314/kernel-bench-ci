#!/usr/bin/env python3
"""worker.py spec.json — 장비에서 도는 실험 본체. status.json 이 유일한 진행 기록이다.

GPU 마다 스레드 하나가 모델 큐를 비운다. 처음부터 fail_fast 개가 연속 실패하면 남은 모델을 돌리지 않는다.
완료 판정은 로그 문구가 아니라 summary.jsonl 의 줄로 한다."""
import json, os, queue, signal, subprocess as sp, sys, threading, time
import stairs, tiles
Z = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(sys.argv[1]))
STAT = f"{Z}/status.json"; LOCKS = os.path.expanduser("~/exp/locks")
lock = threading.Lock()
st = dict(id=S["id"], dev=S["dev"], state="RUNNING", pid=os.getpid(), started=time.time(), updated=time.time(),
          expected=len(S["tiles"]) * S["stairs"]["reps"] if S.get("type") == "stairs" else len(S["models"]) * len(S["trt"]), ok=0, failed=0, fails=[], reason="", build_failed=[], tile_failed={})

def save():
    st["updated"] = time.time()
    with open(STAT + ".t", "w") as f: json.dump(st, f)
    os.replace(STAT + ".t", STAT)

def should_abort(ok, failed, limit):
    return ok == 0 and failed >= limit

def env_for(v):
    e = dict(os.environ, LD_PREFIX=S.get("ld_prefix") or "")
    o = sp.check_output(["bash", "-c", f"source {Z}/env.sh {v} && python3 -c 'import os,json;print(json.dumps(dict(os.environ)))'"], env=e, text=True)
    return json.loads(o.splitlines()[-1])

def take_locks():
    os.makedirs(LOCKS, exist_ok=True); got = []
    for g in S["gpus"]:
        p = f"{LOCKS}/{S['dev']}-gpu{g}"
        try:
            os.mkdir(p); open(f"{p}/owner", "w").write(S["id"]); got.append(p)
        except FileExistsError:
            for q in got: sp.run(["rm", "-rf", q])
            return None, open(f"{p}/owner").read() if os.path.exists(f"{p}/owner") else "?"
    return got, ""

def start_holds():
    """GPU 마다 메모리를 예약한다. 측정용으로 keep_gb 는 남긴다."""
    if not S.get("hold_gb"):
        return []
    E = env_for(S["trt"][0]); P = []
    for g in S["gpus"]:
        e = dict(E, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(g), LD_LIBRARY_PATH=E["LP"])
        lg = open(f"{Z}/hold{g}.log", "w")
        P.append(sp.Popen([E["PY"], f"{Z}/hold.py", str(S["hold_gb"]), str(S.get("keep_gb", 12))], cwd=Z, env=e, stdout=lg, stderr=sp.STDOUT, stdin=sp.DEVNULL))
    return P


def stop_holds(P):
    for p in P:
        p.terminate()
    for p in P:
        p.wait(timeout=30)


def result(res, k):
    f = f"{Z}/{res}/summary.jsonl"; last = None
    if os.path.exists(f):
        for l in open(f):
            d = json.loads(l)
            if d.get("k") == k: last = d
    return last

def run_model(a, cwd, env, log, limit):
    """모델 하나를 돌린다. 멈춘 측정이 GPU 를 잡고 있지 않도록 시간 제한이 지나면 자식까지 끝낸다."""
    p = sp.Popen(a, cwd=cwd, env=env, stdout=log, stderr=sp.STDOUT, stdin=sp.DEVNULL, start_new_session=True)
    try:
        return p.wait(timeout=limit)
    except sp.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait(); return 124

def build_tiles():
    """명세에 타일 후보가 있으면 저장소 커널로 빌드한다. 일부 실패는 기록하고 계속한다."""
    if not S.get("tiles"):
        return True
    libs = {v: env_for(v)["LIB"] for v in S["trt"]}
    bad, why = tiles.split_result(tiles.build(Z, S["tiles"], S["build"], S["flags"], libs), S["fixed_tile"])
    st["build_failed"] = bad; save()
    if why:
        st.update(state="FAILED", reason=why); save()
    return not why

def run_version(v):
    E = env_for(v); res = f"res/{v}"; os.makedirs(f"{Z}/{res}", exist_ok=True)
    E.update(EXP_ID=S["id"], RES=res, SODIR=f"sv/{v}", FIXT=S["fixed_tile"], FUSED_FC2_SPLITK="1", **S.get("inject", {}))
    q = queue.Queue(); [q.put(m) for m in S["models"]]; stop = threading.Event()
    def loop(g):
        while not stop.is_set():
            try: k, tn, d = q.get_nowait()
            except queue.Empty: return
            with open(f"{Z}/{res}/{k}.log", "w") as lg:
                rc = run_model([E["PY"], f"{Z}/run2.py", k, tn, d, str(g)], Z, E, lg, S.get("model_timeout_h", 4) * 3600)
            d_ = result(res, k)
            with lock:
                for t in tiles.failed_tiles(f"{Z}/{res}/{k}.jsonl"):
                    st["tile_failed"][t] = st["tile_failed"].get(t, 0) + 1
                if d_ and not d_.get("fail") and rc == 0: st["ok"] += 1
                else:
                    st["failed"] += 1; st["fails"].append(f"{v}/{k}: {(d_ or {}).get('fail') or 'rc=%d' % rc}")
                    if should_abort(st["ok"], st["failed"], S.get("fail_fast", 2)):
                        st["reason"] = "처음부터 연속 실패"; stop.set()
                save()
    T = [threading.Thread(target=loop, args=(g,)) for g in S["gpus"]]
    [t.start() for t in T]; [t.join() for t in T]
    return not stop.is_set()

def run_stairs():
    """타일을 차례로 돌린다. 반복마다 순서를 바꾼다. GPU 하나만 쓴다."""
    v = S["trt"][0]; E = env_for(v); res = f"{Z}/res/stairs"; os.makedirs(res, exist_ok=True); C = S["stairs"]
    for r in range(C["reps"]):
        for name in (S["tiles"] if r % 2 == 0 else S["tiles"][::-1]):
            ok, why = stairs.run(Z, name, S["build"], S["flags"], E, S["gpus"][0], r, C["nmax"], res)
            with open(f"{res}/summary.jsonl", "a") as f:
                f.write(json.dumps(dict(k=name, r=r, fail=why)) + "\n")
            if ok: st["ok"] += 1
            else: st["failed"] += 1; st["fails"].append(f"{name}: {why}")
            save()

def final_state():
    if st["ok"] == st["expected"]: return "SUCCEEDED"
    return "FAILED" if st["ok"] == 0 else "PARTIAL"

def main():
    save(); got, who = take_locks()
    if got is None:
        st.update(state="FAILED", reason=f"GPU 잠금: {who} 사용 중"); save(); return
    holds = []
    try:
        holds = start_holds()
        if S.get("type") == "stairs":
            run_stairs(); st["state"] = final_state(); save()
        elif build_tiles():
            for v in S["trt"]:
                if not run_version(v): break
            st["state"] = final_state(); save()
    except Exception as e:
        st.update(state="FAILED", reason=repr(e)[:300]); save()
    finally:
        stop_holds(holds)   # 상태를 쓰기 전이 아니라 끝날 때 푼다. 판정과 무관하게 항상 푼다
        for p in got: sp.run(["rm", "-rf", p])

if __name__ == "__main__": main()
