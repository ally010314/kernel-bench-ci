#!/usr/bin/env python3
"""smoke.py <gpu> <trt> <timm 모델> <minDim> <so> — 측정 전 시험. 작은 모델 하나로 전 과정을 돌린다.
버전 확인 → export → surgery → TRT 빌드 → 커스텀 빌드 → 실행 → 출력 비교. 결과는 stdout 마지막 줄 JSON."""
import json, os, shutil, subprocess as sp, sys
gpu, want, tn, min_dim, so = sys.argv[1:6]
Z = os.path.dirname(os.path.abspath(__file__)); py = os.environ.get("PY", sys.executable)
W = f"/dev/shm/exp/{os.environ.get('EXP_ID', 'manual')}/smoke-{want}"
shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
E = dict(os.environ, LD_LIBRARY_PATH=os.environ["LP"], CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu,
         FUSED_FC2_SPLITK="1", PYTHONWARNINGS="ignore")
def done(stage, ok, err=""):
    print(json.dumps(dict(ok=ok, stage=stage, err=err[-300:])), flush=True); shutil.rmtree(W, ignore_errors=True); sys.exit(0)
def step(stage, a):
    r = sp.run(a, cwd=W, env=E, capture_output=True, text=True, timeout=900)
    if r.returncode:
        lines = [l for l in (r.stdout + r.stderr).splitlines() if "[E]" in l or "Error" in l]
        done(stage, False, (lines[0] if lines else (r.stdout + r.stderr)[-300:]))
    return r.stdout
got = step("version", [py, "-c", "import tensorrt;print(tensorrt.__version__)"]).strip().splitlines()[-1]
if got != want: done("version", False, f"TRT {got} != {want}")
if not os.path.exists(so): done("plugin", False, f"없음 {so}")
step("export", [py, f"{Z}/export.py", tn, "m.onnx", "m.json"])
step("surgery", [py, f"{Z}/surg.py", "m.onnx", "m.json", "c.onnx", min_dim])
step("trt_build", [py, f"{Z}/be.py", "m.onnx", "b.plan"])
step("custom_build", [py, f"{Z}/be.py", "c.onnx", "c.plan", so])
step("trt_run", [py, f"{Z}/rg3.py", "b.plan", so, "s", "b.pt"])
step("custom_run", [py, f"{Z}/rg3.py", "c.plan", so, "s", "c.pt"])
chk = step("output", [py, "-c", "import torch;a=torch.load('b.pt',weights_only=True).flatten();b=torch.load('c.pt',weights_only=True).flatten();print(bool(a.isfinite().all() and b.isfinite().all()),bool(a.argmax()==b.argmax()))"]).split()
if chk[-2:] != ["True", "True"]: done("output", False, f"finite·top1 {chk[-2:]}")
done("all", True)
