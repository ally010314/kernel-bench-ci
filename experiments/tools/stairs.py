#!/usr/bin/env python3
"""stairs.py — 타일을 고정하고 N 만 늘려 커널 단독 시간을 잰다. CTA 가 SM 배수를 넘을 때 시간이 뛰는지 본다.

fc1 · fc2 는 mixerL 크기(hidden 4096, dim 1024)다. 칩을 바꾸지 않으므로 메모리 · 캐시 차이가 섞이지 않는다."""
import json, math, os, subprocess as sp
import tiles

DIM = 1024


def ns_for(tile, nmax):
    """fc2 타일 N 의 배수. 끝 조각이 비지 않아 CTA 와 일이 함께 늘어난다."""
    tn = tiles.TILE[tile.split("_")[1]][2]
    return list(range(tn, nmax + 1, tn))


def cta(tile, n):
    tm, _, tn, _ = tiles.TILE[tile.split("_")[1]]
    return math.ceil(DIM / tm) * math.ceil(n / tn)


def build_cmd(name, b, flags, Z, out):
    inc = f"-I kernels/paper -I kernels -I {b['cutlass_inc']} -I {b['cutlass_util']}"
    nv = f"{b['nvcc_bin']}/nvcc -arch={b['arch']}" + (f" -ccbin {b['ccbin']}" if b.get("ccbin") else "")
    return (f"cd {Z} && . kernels/paper/config.sh && eval \"DEFS=($(paper_defs mixerL | sed 's/\"/\\\\\"/g'))\" && "
            f"{nv} -std=c++17 $PAPER_BASE {flags} {tiles.tile_flags(name)} -DN_PAD_MAX=3200 \"${{DEFS[@]}}\" -Xcompiler -fPIC -c {inc} kernels/paper/fused_mlp.cu -o {out}/k.o && "
            f"{nv} -O3 -std=c++17 \"${{DEFS[@]}}\" -I kernels/paper bench/mlp_bench.cu {out}/k.o -lcublas -o {out}/mlp_bench")


def points(name, r, results):
    """run.py 결과를 N 별 기록으로 바꾼다."""
    out = []
    for n, v in sorted(results.items(), key=lambda x: int(x[0])):
        out.append(dict(tile=name, r=r, N=int(n), cta=cta(name, int(n)), fused_ms=v.get("fused_ms"), fc1_ms=v.get("fc1_ms"), fc2_ms=v.get("fc2_ms")))
    return out


def run(Z, name, b, flags, env, gpu, r, nmax, res):
    """타일 하나를 빌드해 N 전체를 잰다. (성공 여부, 사유)"""
    b = tiles.expand(b); o = f"{Z}/bld_stairs/{name}"; os.makedirs(o, exist_ok=True); log = f"{o}/log"
    if not os.path.exists(f"{o}/mlp_bench") and tiles.run(build_cmd(name, b, flags, Z, o), log):
        return False, "build"
    e = dict(env, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu), FUSED_FC2_SPLITK="1", LD_LIBRARY_PATH=f"{env['LP']}:{b['cuda_lib']}")
    w = f"/dev/shm/stairs_{os.getpid()}"; out = f"{o}/r{r}.json"
    ns = ",".join(map(str, ns_for(name, nmax)))
    p = sp.run([env["PY"], f"{Z}/bench/run.py", "--model", "mixerL", "--bench", f"{o}/mlp_bench", "--ns", ns, "--baseline-dir", "/nonexistent", "--work", w, "--out", out, "--sha", "stairs"],
               cwd=Z, env=e, capture_output=True, text=True)
    sp.run(["rm", "-rf", w])
    if p.returncode not in (0, 2) or not os.path.exists(out):
        return False, f"run rc={p.returncode} {(p.stdout + p.stderr)[-160:]}"
    rows = points(name, r, json.load(open(out))["results"])
    with open(f"{res}/stairs.jsonl", "a") as f:
        for x in rows:
            f.write(json.dumps(x) + "\n")
    return True, ""
