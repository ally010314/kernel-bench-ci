#!/usr/bin/env python3
# be.py <in.onnx> <out.plan> [plugin.so] — ONNX 를 FP16 + SPARSE_WEIGHTS 엔진으로 빌드한다.
# FP16 플래그가 있는 TRT(≤ 10.x)는 플래그를 켠다. 없으면 플러그인이 FP32 만 질의받아 빌드가 실패한다.
# TRT 11 은 플래그가 없어 fp16_graph.py 로 가중치 · 상수를 FP16 으로 내린 그래프를 빌드한다.
import ctypes, os, subprocess, sys
import tensorrt as trt

src, dst = sys.argv[1:3]
so = sys.argv[3] if len(sys.argv) > 3 else None
use_flag = hasattr(trt.BuilderFlag, "FP16") and os.environ.get("NO_FP16_FLAG") != "1"
if not use_flag:
    h = src + ".h16.onnx"
    if not os.path.exists(h):
        subprocess.check_call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "fp16_graph.py"), src, h])
    src = h
lg = trt.Logger(trt.Logger.ERROR)
if so:
    ctypes.CDLL(so)
trt.init_libnvinfer_plugins(lg, "")
b = trt.Builder(lg)
net = b.create_network(0)
p = trt.OnnxParser(net, lg)
if not p.parse_from_file(src):
    [print(p.get_error(i)) for i in range(p.num_errors)]
    raise SystemExit(1)
c = b.create_builder_config()
if use_flag:
    c.set_flag(trt.BuilderFlag.FP16)
if os.environ.get("NOSPARSE") != "1":
    c.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
c.avg_timing_iterations = int(os.environ.get("AVG", 8))
c.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
pr = b.create_optimization_profile()
i = net.get_input(0)
s = tuple(1 if d < 0 else d for d in i.shape)
pr.set_shape(i.name, s, s, s)
c.add_optimization_profile(pr)
pl = b.build_serialized_network(net, c)
assert pl is not None
open(dst, "wb").write(pl)
