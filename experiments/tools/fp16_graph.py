#!/usr/bin/env python3
"""fp16_graph.py — 커스텀 op 가 있는 ONNX 를 FP16 으로 바꾼다 (ModelOpt AutoCast 는 커스텀 op 그래프를 처리하려면 TRT 가 필요).
TRT 11 은 BuilderFlag.FP16 이 없어서 필요하다. 모든 FLOAT 가중치·상수를 FLOAT16 으로 내리고, 입력 뒤와 출력 앞에 Cast 를 넣어
입출력은 FP32 로 유지한다. 사용: fp16_graph.py <in.onnx> <out.onnx>"""
import sys, numpy as np, onnx
from onnx import TensorProto as T, helper as H, numpy_helper as N
m = onnx.load(sys.argv[1]); g = m.graph
def down(t):
    if t.data_type == T.FLOAT:
        t.CopyFrom(N.from_array(N.to_array(t).astype(np.float16), t.name))
for t in g.initializer: down(t)
for n in g.node:
    for a in n.attribute:
        if a.type == onnx.AttributeProto.TENSOR: down(a.t)
        if n.op_type == "Cast" and a.name == "to" and a.i == T.FLOAT: a.i = T.FLOAT16   # 어텐션의 Shape → Cast(FLOAT) → Sqrt 경로
i, o = g.input[0].name, g.output[0].name
for n in g.node:
    for lst in (n.input, n.output):
        for k, s in enumerate(lst):
            if s == i: lst[k] = i + "_h"
            elif s == o: lst[k] = o + "_h"
cast_in = H.make_node("Cast", [i], [i + "_h"], to=T.FLOAT16, name="in_cast")
cast_out = H.make_node("Cast", [o + "_h"], [o], to=T.FLOAT, name="out_cast")
nodes = [cast_in] + list(g.node) + [cast_out]
g.ClearField("node"); g.node.extend(nodes); g.ClearField("value_info")
big = m.ByteSize() > 1900000000 if False else sum(len(t.raw_data) for t in g.initializer) > 1500000000
onnx.save(m, sys.argv[2], save_as_external_data=big, all_tensors_to_one_file=True, location=sys.argv[2].split("/")[-1] + ".data"); print("ok", sys.argv[2])
