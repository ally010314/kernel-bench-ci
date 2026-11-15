#!/usr/bin/env python3
# surg.py <in.onnx> <in.json> <out.onnx> <minDim> [접두]: dim>=minDim 인 MLP 를 FusedMlpR 로 교체하고 data/allblk 에 가중치를 뽑는다. 4차원 입력은 Reshape 로 감싼다.
import os,sys,re,json,numpy as np,onnx,onnx_graphsurgeon as gs
src,js,dst,minD=sys.argv[1:5]; minD=int(minD); P=sys.argv[5] if len(sys.argv)>5 else "mx"; J=json.load(open(js))
def pack(W):
    M,K=W.shape; g=W.reshape(M,K//4,4); nz=g!=0; assert (nz.sum(-1)<=2).all(); ar=np.arange(4); o=np.argsort(np.where(nz,ar,ar+4),axis=-1); s=np.sort(o[...,:2],axis=-1); i0,i1=s[...,0].astype(np.int64),s[...,1].astype(np.int64)
    v0=np.take_along_axis(g,i0[...,None],-1)[...,0]; v1=np.take_along_axis(g,i1[...,None],-1)[...,0]
    p=np.stack([v0,v1],-1).reshape(M,K//2).astype(np.float16); nib=((i0&3)|((i1&3)<<2)).astype(np.uint16); me=(nib.reshape(M,K//16,4)<<np.array([0,4,8,12],dtype=np.uint16)).sum(-1)
    return p,me.astype(np.uint16)
m=onnx.load(src); g=gs.import_onnx(m)
for _ in range(4):
    f=0
    for n in [x for x in g.nodes if x.op=="Identity"]:
        s,d=n.inputs[0],n.outputs[0]
        if not isinstance(s,gs.Constant): continue
        c=gs.Constant(d.name,s.values)
        for k in list(d.outputs): k.inputs=[c if t is d else t for t in k.inputs]
        n.outputs.clear(); f+=1
    g.cleanup().toposort()
    if not f: break
N={n.name:n for n in g.nodes}
def co(n):
    for t in n.inputs:
        if isinstance(t,gs.Constant): return t.values
    raise KeyError(n.name)
def da(n):
    for t in n.inputs:
        if not isinstance(t,gs.Constant): return t
F=[n for n in g.nodes if re.search(r"/(mlp|mlp_channels)/fc1/MatMul$",n.name)]
assert len(F)==len(J["mlps"]),(len(F),len(J["mlps"]))
os.makedirs("data/allblk",exist_ok=True); b=0; L=[]
for n,M in zip(F,J["mlps"]):
    pre=n.name[:-len("/fc1/MatMul")]; W1=co(n).T.astype(np.float16).copy(); W2=co(N[pre+"/fc2/MatMul"]).T.astype(np.float16).copy(); h,d=W1.shape; sh=M["shape"]
    if d<minD or d%8 or W2.shape!=(d,h): continue
    t=np.prod(sh[1:-1]); 
    p1,m1=pack(W1); p2,m2=pack(W2); b1=co(N[pre+"/fc1/Add"]).astype(np.float16); b2=co(N[pre+"/fc2/Add"]).astype(np.float16); tag=f"data/allblk/{P}_b{b:02d}"
    p1.tofile(tag+"_fc1_packed.bin"); m1.tofile(tag+"_fc1_meta.bin"); b1.tofile(tag+"_fc1_bias.bin"); p2.tofile(tag+"_fc2_packed.bin"); m2.tofile(tag+"_fc2_meta.bin"); b2.tofile(tag+"_fc2_bias.bin")
    X=da(n); Add=N[pre+"/fc2/Add"]; Y=Add.outputs[0]; Add.outputs.clear(); DT=np.float32; A=dict(M1=int(h),K1=int(d),M2=int(d),K2=int(h),block=b,plugin_version="1",plugin_namespace="")
    if len(sh)==4:
        Xr=gs.Variable(f"b{b}_xr",dtype=DT); Yr=gs.Variable(f"b{b}_yr",dtype=DT)
        g.nodes.append(gs.Node(op="Reshape",name=f"b{b}_rs1",inputs=[X,gs.Constant(f"b{b}_s1",np.array([1,-1,d],dtype=np.int64))],outputs=[Xr]))
        g.nodes.append(gs.Node(op="FusedMlpR",name=f"block{b}_fused_mlp_mx",attrs=A,inputs=[Xr],outputs=[Yr]))
        g.nodes.append(gs.Node(op="Reshape",name=f"b{b}_rs2",inputs=[Yr,gs.Constant(f"b{b}_s2",np.array([1,sh[1],sh[2],d],dtype=np.int64))],outputs=[Y]))
    else: g.nodes.append(gs.Node(op="FusedMlpR",name=f"block{b}_fused_mlp_mx",attrs=A,inputs=[X],outputs=[Y]))
    L.append((int(t),int(d))); b+=1
g.cleanup().toposort(); o=gs.export_onnx(g); big=sum(i.ByteSize() if hasattr(i,"ByteSize") else 0 for i in o.graph.initializer)>1.8e9
onnx.save(o,dst,save_as_external_data=big,all_tensors_to_one_file=True,location=os.path.basename(dst)+".data")
json.dump(L,open(dst+".json","w")); print("교체",b,"블록",sorted(set(L)))
