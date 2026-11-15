#!/usr/bin/env python3
# export.py <timm이름> <out.onnx> <out.json> [img] — 사전학습 가중치(없으면 무작위) → MLP fc1 · fc2 2:4 프루닝 → ONNX.
import sys,json,os,torch,timm,torch.nn as nn
name,out,js=sys.argv[1:4]; kw={}
if len(sys.argv)>4: kw["img_size"]=int(sys.argv[4])
try: m=timm.create_model(name,pretrained=True,**kw).eval(); pre=True
except Exception as e:
    print("pretrained 실패",str(e)[:100]); torch.manual_seed(0); m=timm.create_model(name,pretrained=False,**kw).eval(); pre=False
def prune(w):
    o,K=w.shape; g=w.detach().reshape(o,K//4,4); i=g.abs().argsort(-1,descending=True)[...,2:]; k=torch.ones_like(g); k.scatter_(-1,i,0.0); return (g*k).reshape(o,K)
S=[]; H=[]
def hk(n):
    def f(mod,inp): S.append((n,list(inp[0].shape)))
    return f
for n,x in m.named_modules():
    if n.split(".")[-1] in ("mlp","mlp_channels") and isinstance(getattr(x,"fc1",None),nn.Linear) and isinstance(getattr(x,"fc2",None),nn.Linear) and isinstance(getattr(x,"act",None),nn.GELU):
        for fc in (x.fc1,x.fc2):
            if fc.weight.shape[1]%4==0:
                with torch.no_grad(): fc.weight.copy_(prune(fc.weight))
        H.append(x.register_forward_pre_hook(hk(n)))
c=timm.data.resolve_model_data_config(m); sz=c["input_size"]; x=torch.randn(1,*sz)
with torch.no_grad(): m(x)
for h in H: h.remove()
torch.onnx.export(m,x,out,input_names=["input"],output_names=["output"],opset_version=17,do_constant_folding=True,dynamo=False)
json.dump(dict(name=name,pretrained=pre,img=sz,params=sum(p.numel() for p in m.parameters()),mlps=[dict(n=a,shape=b) for a,b in S]),open(js,"w"))
print(name,"pretrained",pre,"mlp",len(S),"onnx",os.path.getsize(out)//10**6,"MB")
