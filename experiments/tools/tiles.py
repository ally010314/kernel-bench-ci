#!/usr/bin/env python3
"""tiles.py — 타일 후보의 이름 · 매크로와 장비별 빌드. expctl 은 검증에, worker 는 빌드에 쓴다.

이름은 <fc1>_<fc2>_s<fc2 stage> 다 (예: 128x128_64x64_s4). 소스는 실험 폴더에 복사한 저장소 커널이다."""
import concurrent.futures as cf, glob, json, os, re, subprocess as sp

TILE = {"128x64": (128, 64, 64, 32), "64x64": (64, 32, 64, 32), "128x32": (128, 64, 32, 16), "64x32": (64, 32, 32, 16),
        "128x128": (128, 64, 128, 64), "256x64": (256, 64, 64, 32), "256x128": (256, 64, 128, 64)}
FC1 = ("128x128", "128x64", "128x32", "64x64")
FC2 = tuple(TILE)
FLAGS = ""   # 커널 최적화 플래그 (masked)
CLS, DATA, H, D, PAD = "FusedMlpR", "mx", 4096, 1024, 6000
JOBS = 8


def names(t):
    """명세의 tiles 를 후보 이름으로 편다."""
    return [f"{a}_{b}_s{s}" for a in t["fc1"] for b in t["fc2"] for s in t["stages"]]


def check(t, flags=None):
    """tiles · flags 명세 오류."""
    e = []
    for k in ("fc1", "fc2", "stages"):
        if not t.get(k):
            e.append(f"tiles.{k} 없음")
    if e:
        return e
    e += [f"fc1 타일 {x} 미지원 ({list(FC1)})" for x in t["fc1"] if x not in FC1]
    e += [f"fc2 타일 {x} 미지원 ({list(FC2)})" for x in t["fc2"] if x not in FC2]
    e += [f"stage {x} 미지원" for x in t["stages"] if x not in (2, 3, 4)]
    if flags is not None and not re.fullmatch(r"(\s*-D\w+(=\w+)?)+\s*", flags):
        e.append(f"flags 형식: {flags}")
    return e


def macro(tile, i):
    a, b, c, d = TILE[tile]
    return f"-DTB{i}_M={a} -DWP{i}_M={b} -DTB{i}_N={c} -DWP{i}_N={d}"


def tile_flags(name):
    a, b, s = name.split("_")
    return f"{macro(a, 1)} {macro(b, 2)} -DST1=3 -DST2={s[1:]}"


def compile_cmd(name, b, flags, src, out):
    df = f"""-DMLP_CLASS={CLS} '-DMLP_DATA="{DATA}"' -DMLP_M1={H} -DMLP_K1={D} -DMLP_M2={D} -DMLP_K2={H} -DN_PAD_MAX={PAD}"""
    inc = f"-I kernels/paper -I kernels -I {b['cutlass_inc']} -I {b['cutlass_util']}"
    return (f"cd {src} && . kernels/paper/config.sh && {b['nvcc_bin']}/nvcc -arch={b['arch']}{' -ccbin ' + b['ccbin'] if b.get('ccbin') else ''} -std=c++17 $PAPER_BASE {flags} {tile_flags(name)} {df} "
            f"-Xcompiler -fPIC -c {inc} kernels/paper/fused_mlp.cu -o {out}/k.o")


def link_cmd(b, v, lib, so, src, obj):
    inc = b["trt_inc"].format(major=v.split(".")[0])
    return (f"cd {src} && g++ -std=c++17 -O2 -shared -fPIC -I kernels/paper -I {inc} -I {b['cuda_inc']} -DMLP_CLASS={CLS} -DMLP_M1={H} -DMLP_K1={D} -DMLP_M2={D} -DMLP_K2={H} "
            f"kernels/paper/fused_mlp_plugin.cpp {obj} -L{lib} -l:{so} -L{b['cuda_lib']} -lcudart -o")


def expand(b):
    return {k: os.path.expanduser(x) for k, x in b.items()}


def run(c, log):
    with open(log, "a") as f:
        return sp.run(c, shell=True, executable="/bin/bash", stdout=f, stderr=sp.STDOUT).returncode


def build(Z, tiles, b, flags, libs):
    """타일마다 커널을 컴파일하고 TRT 버전마다 링크한다. {타일: 실패 사유 또는 ""} 를 돌려주고 build.json 에 남긴다."""
    b = expand(b); root = f"{Z}/bld"; os.makedirs(root, exist_ok=True)
    def one(n):
        o = f"{root}/{n}"; os.makedirs(o, exist_ok=True); log = f"{o}/log"
        if run(compile_cmd(n, b, flags, Z, o), log):
            return n, "compile"
        for v, lib in libs.items():
            so = os.path.basename(sorted(glob.glob(f"{lib}/libnvinfer.so.*"), key=len)[0]); d = f"{Z}/sv/{v}/{n}"; os.makedirs(d, exist_ok=True)
            if run(f"{link_cmd(b, v, lib, so, Z, o + '/k.o')} {d}/p.so", log):
                return n, f"link {v}"
        return n, ""
    with cf.ThreadPoolExecutor(JOBS) as e:
        res = dict(e.map(one, tiles))
    json.dump(res, open(f"{Z}/build.json", "w"))
    return res


def split_result(res, fixed):
    """빌드 결과에서 (실패 목록, 중단 사유). 고정 타일이 없으면 칩 비교의 기준이 사라지므로 멈춘다."""
    bad = sorted(n for n, w in res.items() if w)
    why = ""
    if len(bad) == len(res):
        why = "타일 빌드 전부 실패"
    elif res.get(fixed):
        why = f"고정 타일 빌드 실패: {fixed}"
    return bad, why


def failed_tiles(f):
    """run2 1단계 결과에서 실행에 실패한 타일. 공유 메모리 한도 같은 실패를 기록으로 남긴다."""
    bad = []
    if os.path.exists(f):
        for l in open(f):
            d = json.loads(l)
            if d.get("ph") == 1 and d.get("n") != "b1" and not d.get("x"):
                bad.append(d["n"])
    return bad


def rows_by_tile(rows):
    """run2 결과 줄을 타일별 값으로 묶는다. 1단계는 Graph 1회, 2단계는 상위 타일의 반복 측정이다."""
    rows = [dict(r, x=r.get("x") or ({"med": r["ms"]} if r.get("ms") else None), m=r.get("m", "s")) for r in rows]   # 초기 측정은 ms 필드 하나다
    P = {r["n"]: r for r in rows if r.get("ph") == 1 and r.get("n") != "b1"}
    diffs = [r["diff"] for r in P.values() if r.get("diff") is not None]
    lim = max(min(diffs) * 1.5, 0.05) if diffs else None
    two = {}
    for r in rows:
        if r.get("ph") == 2 and r.get("x") and r.get("n") in P:
            two.setdefault((r["n"], r["m"]), []).append(r["x"]["med"])
    out = {}
    for n, r in P.items():
        x = r.get("x"); m = {}
        if x:
            m["graph_ms"] = x["med"]
        for k in ("diff", "nan_c", "nan_t"):
            if r.get(k) is not None:
                m[k] = float(r[k])
        for md, lab in (("g", "graph"), ("s", "stream")):
            v = sorted(two.get((n, md), []))
            if v:
                m[f"repeat_{lab}_ms"] = v[len(v) // 2]; m[f"repeat_{lab}_n"] = float(len(v))
        out[n] = dict(metrics=m, ran=bool(x), selectable=bool(x and r.get("top1") and r.get("diff") is not None and r["diff"] <= lim))
    return out
