#!/usr/bin/env python3
# rg3.py <plan> <so> <s|g> <out.pt> — 기본 실행(s) 또는 CUDA Graph(g) 시간과 출력.
import ctypes,json,sys,time,statistics as st
import tensorrt as trt,torch
plan,so,mode,out=sys.argv[1:5]
lg=trt.Logger(trt.Logger.ERROR); ctypes.CDLL(so); trt.init_libnvinfer_plugins(lg,"")
rt=trt.Runtime(lg); eng=rt.deserialize_cuda_engine(open(plan,"rb").read()); ctx=eng.create_execution_context()
ins={};outs={}
for i in range(eng.num_io_tensors):
    n=eng.get_tensor_name(i); s=tuple(eng.get_tensor_shape(n)); dt=torch.float16 if eng.get_tensor_dtype(n)==trt.DataType.HALF else torch.float32
    if eng.get_tensor_mode(n)==trt.TensorIOMode.INPUT:
        ins[n]=torch.randn(*s,generator=torch.Generator().manual_seed(0)).to("cuda",dtype=dt); ctx.set_tensor_address(n,int(ins[n].data_ptr()))
    else:
        outs[n]=torch.zeros(*s,device="cuda",dtype=dt); ctx.set_tensor_address(n,int(outs[n].data_ptr()))
S=torch.cuda.Stream(); ts=[]
with torch.cuda.stream(S):
    for _ in range(50): ctx.execute_async_v3(S.cuda_stream); S.synchronize()
    if mode=="g":
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        f=lambda:(g.replay(),S.synchronize())
    else:
        f=lambda:(ctx.execute_async_v3(S.cuda_stream),S.synchronize())
    for i in range(800):
        t=time.perf_counter(); f(); d=time.perf_counter()-t
        if i>=300: ts.append(d*1e3)
torch.save(next(iter(outs.values())).float().cpu(),out)
ts.sort(); print(json.dumps(dict(med=round(st.median(ts),4),mean=round(st.fmean(ts),4),p10=round(ts[50],4))))
