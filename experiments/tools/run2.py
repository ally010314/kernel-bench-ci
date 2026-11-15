#!/usr/bin/env python3
# run2.py <key> <timm> <minDim> <gpu> — TRT 5빌드 · Dense · 커스텀(CUDA Graph 로 타일 선별, 상위 2타일, 1위 타일 5빌드)을 기본 실행과 Graph 로 번갈아 잰다.
# TRT 도 5빌드라 빌드 선택 조건이 커스텀과 같다. NaN 은 위치·개수를 기록하고 유한한 값끼리 비교한다.
import sys,os,subprocess as sp,json,statistics as st,collections as C,shutil,glob
k,tn,minD,gpu=sys.argv[1:5]; H=os.path.expanduser("~"); Z=os.path.dirname(os.path.abspath(__file__)); W=f"/dev/shm/exp/{os.environ.get('EXP_ID','manual')}/{k}"; os.makedirs(W,exist_ok=True); py=os.environ.get("PY",sys.executable)
RS=os.environ.get("RES","res2"); SD=os.environ.get("SODIR","so"); os.makedirs(f"{Z}/{RS}",exist_ok=True)
BE=os.environ.get("BE","be.py")
lp=os.environ["LP"]
E=dict(os.environ,LD_LIBRARY_PATH=lp,CUDA_DEVICE_ORDER="PCI_BUS_ID",CUDA_VISIBLE_DEVICES=gpu,FUSED_FC2_SPLITK="1",PYTHONWARNINGS="ignore")
def sh(a,env=E,to=3000):
    r=sp.run(a,cwd=W,env=env,capture_output=True,text=True,timeout=to); print(" ".join(a[1:3])[-70:],(r.stdout+r.stderr)[-160:].replace("\n"," "),flush=True); return r.returncode==0
out=open(f"{Z}/{RS}/{k}.jsonl","w"); sum_=f"{Z}/{RS}/summary.jsonl"
def fail(w):
    open(sum_,"a").write(json.dumps(dict(k=k,tn=tn,fail=w))+"\n"); print("FAIL",k,w,flush=True); shutil.rmtree(W,ignore_errors=True); sys.exit(0)
if not sh([py,f"{Z}/export.py",tn,"m.onnx","m.json"]): fail("export")
SO=sorted(glob.glob(f"{Z}/{SD}/*/p.so")); so0=SO[0]
for s,d in [(f"b{i}",{}) for i in range(1,6)]+[("d1",{"NOSPARSE":"1"})]:
    if not sh([py,f"{Z}/{BE}","m.onnx",f"{s}.plan"],dict(E,**d)): fail("build "+s)
if not sh([py,f"{Z}/surg.py","m.onnx","m.json","c.onnx",minD]): fail("surgery")
if not sh([py,f"{Z}/{BE}","c.onnx","c.plan",so0]): fail("custom build")
import torch
J=json.load(open(f"{W}/m.json")); ref=f"{W}/ref.pt"; tmp=f"{W}/t.pt"
def run(pl,so,m,o):
    try: r=sp.run([py,f"{Z}/rg3.py",pl,so,m,o],cwd=W,env=E,capture_output=True,text=True,timeout=900)
    except Exception: return None
    for l in r.stdout.splitlines()[::-1]:
        if l.startswith("{"): return json.loads(l)
    print("runfail",pl,m,(r.stdout+r.stderr)[-200:].replace("\n"," "),flush=True); return None
def cmp():
    a=torch.load(tmp,weights_only=True).flatten(); b=torch.load(ref,weights_only=True).flatten(); ok=a.isfinite()&b.isfinite()
    return dict(diff=float((a[ok]-b[ok]).abs().max()) if ok.any() else None,top1=bool(torch.nan_to_num(a).argmax()==torch.nan_to_num(b).argmax()),nan_c=int((~a.isfinite()).sum()),nan_t=int((~b.isfinite()).sum()))
def one(n,pl,so,m,r,ph):
    x=run(pl,so,m,ref if (n=="b1" and m=="s" and not os.path.exists(ref)) else tmp); d={}
    if x and pl[0]=="c" and os.path.exists(ref): d=cmp()
    out.write(json.dumps(dict(k=k,ph=ph,n=n,m=m,r=r,x=x,**d))+"\n"); out.flush(); return x,d
nm=lambda s:s.split("/")[-2]
one("b1","b1.plan",so0,"s",0,1)
if not os.path.exists(ref): fail("ref")
R={}
for s in SO:
    x,d=one(nm(s),"c.plan",s,"g",0,1)
    if x: R[nm(s)]=(x["med"],d)
if not R: fail("all custom failed in graph")
dm=min((v[1]["diff"] for v in R.values() if v[1].get("diff") is not None),default=None)
if dm is None: fail("no finite diff")
ok=[n for n,v in R.items() if v[1].get("top1") and v[1].get("diff") is not None and v[1]["diff"]<=max(dm*1.5,0.05)]
if not ok: fail(f"output mismatch min diff {dm}")
top=sorted(ok,key=lambda n:R[n][0])[:2]; FX=os.environ.get("FIXT"); top=top+[FX] if FX and FX in R and FX not in top else top; print("top",top,flush=True); so1=f"{Z}/{SD}/{top[0]}/p.so"
CP=[]
for i in (2,3,4,5):
    if sh([py,f"{Z}/{BE}","c.onnx",f"c{i}.plan",so1]): CP.append(f"c{i}")
L=[(f"b{i}",f"b{i}.plan",so0) for i in range(1,6)]+[("d1","d1.plan",so0)]+[(n,"c.plan",f"{Z}/{SD}/{n}/p.so") for n in top]+[(f"{p}:{top[0]}",f"{p}.plan",so1) for p in CP]
D=C.defaultdict(list)
for r in range(3):
    for n,pl,so in L[r*4%len(L):]+L[:r*4%len(L)]:
        for m in "gs":
            x,d=one(n,pl,so,m,r,2)
            if x: D[(m,n)].append(x["med"])
md=lambda m,n:st.median(D[(m,n)]) if D[(m,n)] else None
res={}
for m in "gs":
    T=[md(m,f"b{i}") for i in range(1,6) if md(m,f"b{i}")]; Cs=[md(m,n) for n,_,_ in L[6:] if md(m,n)]
    if T and Cs: res[m]=dict(trt_min=min(T),trt_med=st.median(T),trt_max=max(T),dense=md(m,"d1"),cust_min=min(Cs),cust_med=st.median(Cs),gain_bb=round((min(Cs)/min(T)-1)*100,2),gain_mm=round((st.median(Cs)/st.median(T)-1)*100,2))
open(sum_,"a").write(json.dumps(dict(k=k,tn=tn,img=J.get("img"),shapes=sorted({tuple(x) for x in J.get("rep",[])}) if J.get("rep") else None,top=top,res=res,nan=R[top[0]][1].get("nan_c"),nan_t=R[top[0]][1].get("nan_t"),diff=R[top[0]][1].get("diff")))+"\n")
print("DONE",k,res.get("g",{}).get("gain_bb"),res.get("s",{}).get("gain_bb"),flush=True); shutil.rmtree(W,ignore_errors=True)
