#!/usr/bin/env python3
"""collect.py — expctl 로 띄운 실험의 결과를 장비에서 모아 MLflow 에 올린다.

run 하나는 측정 한 번이다. 실험 종류마다 experiment 를 나눈다.

  gpu-study/tile-search  장비 · TRT · 실험 · 모델마다 run. 1위 타일의 E2E 와 TRT 비교, 전체 타일 표(tiles.csv).
  gpu-study/stairs       장비 · 타일마다 run. CTA 를 step 으로 한 시간 곡선, 계단 폭 맞춤, 원본.

src_id 태그는 측정의 정체(장비 · 실험 · 모델 …), src_key 는 내용 해시다. 같은 src_id 의 내용이 바뀌면
옛 run 을 지우고 새로 올린다. 그래서 진행 중인 측정도 run 이 하나만 남는다. 장부 파일은 없고 MLflow 가 기준이다.
  python3 experiments/collect.py [--dry] [--no-fetch] [--only tile-search,stairs]
env: MLFLOW_TRACKING_URI (필수), MLOPS_CACHE (기본 ~/.cache/kernel-bench-collect)
CI: .github/workflows/collect.yml 이 Orin runner 에서 30분마다 돌린다.
"""
import argparse,glob,hashlib,json,os,re,subprocess as sp,sys,time
from pathlib import Path
HOME=Path.home(); ME=Path(__file__).resolve().parent; CACHE=Path(os.environ.get("MLOPS_CACHE",HOME/".cache/kernel-bench-collect"))
sys.path.insert(0,str(ME.parent/"bench")); sys.path.insert(0,str(ME/"tools")); from mlflow_log import Client
import tiles as TL
EXP="gpu-study/"
import yaml
DEV=yaml.safe_load(open(ME/"devices.yml"))
def sha(s): return hashlib.sha1(s.encode() if isinstance(s,str) else s).hexdigest()[:16]
def fetch():
    """장비마다 ~/exp (expctl 실험 폴더) 의 결과 jsonl 을 받아 CACHE/<장비>/_exp 에 푼다. Orin 은 로컬 ~/exp 를 바로 읽는다."""
    for d,c in DEV.items():
        if not c["host"]: continue
        dst=CACHE/d/"_exp"; dst.mkdir(parents=True,exist_ok=True)
        cm="cd ~ && find exp -maxdepth 5 -type f \\( -path '*/res/*.jsonl' -o -path '*/res_g*/*.jsonl' \\) 2>/dev/null | tar czf - -T - ; true"
        a=sp.run(["ssh","-p", "<port>","-o","BatchMode=yes","-o","ConnectTimeout=10",f"<user>@{c['host']}.<domain>",cm],capture_output=True,timeout=300)
        if a.stdout: sp.run(["tar","xzf","-","-C",str(dst)],input=a.stdout)
        else: print("fetch empty",d,a.stderr[-120:])
def rows(f):
    for i,l in enumerate(open(f,errors="ignore")):
        l=l.strip()
        if l:
            try: yield i,json.loads(l)
            except ValueError: pass
def stairs_fit(X,Y,S):
    """시간 = a + b·⌈CTA/S⌉ 의 R². X 는 CTA, Y 는 시간."""
    import math,statistics as st
    w=[math.ceil(c/S) for c in X]; mx,my=st.mean(w),st.mean(Y); sx=sum((a-mx)**2 for a in w); sy=sum((y-my)**2 for y in Y)
    if sx==0 or sy==0: return -1.0
    b=sum((a-mx)*(y-my) for a,y in zip(w,Y))/sx; a0=my-b*mx
    return 1-sum((y-a0-b*q)**2 for q,y in zip(w,Y))/sy

def med(v):
    v=sorted(v); return v[len(v)//2]
def num(x): return isinstance(x,(int,float)) and not isinstance(x,bool)
def it(exp,ident,content,name,params,tags=None,metrics=None,series=None,status="FINISHED",files=(),blobs=None):
    return dict(exp=exp,id=sha(ident),key=sha(f"{ident}|{content}"),name=name[:240],params={a:str(b) for a,b in params.items() if b not in ("",None)},
                tags=tags or {},metrics={a:float(b) for a,b in (metrics or {}).items() if num(b)},series=series or {},status=status,files=[str(f) for f in files],blobs=blobs or {})
def P(d,trt=""):
    c=DEV[d]; return dict(device=d,chip=c["chip"],trt=trt,sm=c.get("sm"),sm_clock_mhz=c["sm_mhz"],driver=c["driver"],cuda=c["cuda"])

def tiles_csv(T,top):
    """타일별 값 표. 1위 · 실행 실패 · 출력 판정 탈락도 모두 남긴다."""
    cols=["tile","is_top","ran","selectable","graph_ms","repeat_graph_ms","repeat_graph_n","repeat_stream_ms","repeat_stream_n","diff","nan_c","nan_t"]
    out=[",".join(cols)]
    for n,v in sorted(T.items(),key=lambda x:(not x[0]==top,x[1]["metrics"].get("graph_ms",9e9))):
        r=dict(tile=n,is_top=n==top,ran=v["ran"],selectable=v["selectable"],**v["metrics"]); out.append(",".join(str(r.get(c,"")) for c in cols))
    return ("\n".join(out)+"\n").encode()
def tile_search(d,trt,stage,src,folder,summ,extra):
    """summary 의 모델 줄과 <모델>.jsonl 의 타일 측정을 run 하나로 묶는다. summary 에 없어도 타일 파일이 있으면 올린다."""
    S={r.get("k"):r for _,r in rows(summ)} if summ and Path(summ).exists() else {}
    ks=sorted(set(S)|{f.stem for f in Path(folder).glob("*.jsonl") if f.stem!="summary"})
    for k in ks:
        r=S.get(k,{}); f=Path(folder)/f"{k}.jsonl"; T=TL.rows_by_tile([x for _,x in rows(f)]) if f.exists() else {}
        top=(r.get("top") or [""])[0]; m={}
        for md,lab in (("g","graph"),("s","stream")):
            for a,b in (r.get("res") or {}).get(md,{}).items(): m[f"{lab}/{a}"]=b
        for a in ("nan","nan_t","diff"): m[a]=r.get(a)
        m.update(tiles=len(T),tiles_ran=sum(v["ran"] for v in T.values()),tiles_selectable=sum(v["selectable"] for v in T.values()))
        if top in T:
            for a,b in T[top]["metrics"].items(): m[f"top/{a}"]=b
        p=dict(P(d,trt),stage=stage,model=k,timm=r.get("tn",""),**extra)
        t=dict(top_tile=top,top3=" ".join(r.get("top") or []),source=src,fail=str(r.get("fail") or ""))
        st_="FAILED" if r.get("fail") or (not r and not any(v["ran"] for v in T.values())) else "FINISHED"
        c=sha(json.dumps(r,sort_keys=True)+(f.read_text(errors="ignore") if f.exists() else ""))
        yield it("tile-search",f"ts|{d}|{trt}|{stage}|{k}",c,f"{d}/{trt}/{stage}/{k}",p,t,m,status=st_,files=[f] if f.exists() else [],blobs={"tiles.csv":tiles_csv(T,top)} if T else None)
def tile_search_items():
    for f in sorted(glob.glob(str(CACHE/"*/_exp/exp/*/*/res/*/summary.jsonl"))+glob.glob(str(HOME/"exp/*/orin/res/*/summary.jsonl"))):
        p=Path(f).parts; eid,d,trt=p[-5],p[-4],p[-2]
        if trt=="stairs": continue
        yield from tile_search(d,trt,eid,f"{d}:exp/{eid}",Path(f).parent,f,dict(exp_id=eid,trt_builds=5))

def stairs_items():
    """장비 · 타일마다 run. CTA 를 step 으로 한 곡선, 계단 폭 맞춤, 원본 jsonl."""
    import statistics as st
    for f in sorted(glob.glob(str(CACHE/"*/_exp/exp/*/*/res/stairs/stairs.jsonl"))):
        p=Path(f).parts; eid,d=p[-5],p[-4]; G={}
        for _,r in rows(f): G.setdefault(r["tile"],{}).setdefault(int(r["cta"]),[]).append(r)
        sm=DEV[d].get("sm"); raw=Path(f).read_bytes()
        for ti,D in sorted(G.items()):
            ser={a.replace("_ms","_us"):[(c,1000*med(q[a] for q in v if num(q.get(a)))) for c,v in sorted(D.items()) if any(num(q.get(a)) for q in v)] for a in ("fc2_ms","fc1_ms","fused_ms")}
            ser={a:b for a,b in ser.items() if b}; ser["N"]=[(c,float(v[0]["N"])) for c,v in sorted(D.items())]
            m=dict(points=len(D),reps=min(len(v) for v in D.values()),cta_min=min(D),cta_max=max(D)); t={}
            y=ser.get("fc2_us")
            if y and len(y)>3:
                X=[c for c,_ in y]; Y=[v for _,v in y]; R=sorted(((stairs_fit(X,Y,S),S) for S in range(4,max(X)+1)),reverse=True)
                m.update(fit_S=R[0][1],fit_r2=R[0][0],linear_r2=stairs_fit(X,Y,1))
                if sm: m["sm_r2"]=stairs_fit(X,Y,sm)
                inc=[(X[i],Y[i]-Y[i-1]) for i in range(1,len(Y))]; md=st.median(abs(v) for _,v in inc)
                t.update(jumps=",".join(str(c) for c,v in inc if v>max(10.0,5*md)),fit_top5=",".join(f"{s}:{r:.4f}" for r,s in R[:5]))
            yield it("stairs",f"st|{eid}|{d}|{ti}",sha(raw),f"{d}/{eid}/{ti}",dict(P(d),exp_id=eid,tile=ti),t,m,series=ser,files=[f])

KINDS={"tile-search":tile_search_items,"stairs":stairs_items}
def items(only=None):
    for k,f in KINDS.items():
        if not only or k in only: yield from f()

class ML(Client):
    def call(self,method,path,**kw):
        """MLflow 서버가 가끔 연결을 끊는다. 연결 오류만 3번까지 다시 시도한다."""
        import requests
        for i in range(4):
            try: return super().call(method,path,**kw)
            except requests.exceptions.ConnectionError:
                if i==3: raise
                time.sleep(2*(i+1))
    def existing(self,e):
        """src_id → (run_id, src_key)."""
        s={};tok=None
        while True:
            d={"experiment_ids":[e],"max_results":1000}
            if tok: d["page_token"]=tok
            r=self.call("POST","runs/search",json=d)
            for x in r.get("runs",[]):
                t={a["key"]:a["value"] for a in x["data"].get("tags",[])}
                if "src_id" in t: s[t["src_id"]]=(x["info"]["run_id"],t.get("src_key"))
            tok=r.get("next_page_token")
            if not tok: return s
    def put(self,e,it,uri):
        r=self.call("POST","runs/create",json=dict(experiment_id=e,start_time=int(time.time()*1000),run_name=it["name"],tags=[dict(key="mlflow.runName",value=it["name"])]))["run"]["info"]
        rid=r["run_id"]; ts=int(time.time()*1000)
        tags=[dict(key=a,value=str(b)[:4900]) for a,b in dict(it["tags"],src_id=it["id"],src_key=it["key"]).items() if b not in ("",None)]
        ps=[dict(key=a,value=b[:490]) for a,b in it["params"].items()]
        ms=[dict(key=re.sub(r"[^\w./ -]","_",a),value=b,timestamp=ts,step=0) for a,b in it["metrics"].items()]
        ms+=[dict(key=re.sub(r"[^\w./ -]","_",a),value=float(v),timestamp=ts,step=int(c)) for a,sr in it["series"].items() for c,v in sr]
        for i in range(0,len(ps),90): self.call("POST","runs/log-batch",json=dict(run_id=rid,params=ps[i:i+90]))
        for i in range(0,len(tags),90): self.call("POST","runs/log-batch",json=dict(run_id=rid,tags=tags[i:i+90]))
        for i in range(0,len(ms),900): self.call("POST","runs/log-batch",json=dict(run_id=rid,metrics=ms[i:i+900]))
        art=[(Path(f).name,Path(f).read_bytes()) for f in it["files"]]+list(it["blobs"].items())
        for n,b in art:
            q=uri.rstrip("/")+f"/api/2.0/mlflow-artifacts/artifacts/{e}/{rid}/artifacts/{n}"
            a=self.s.put(q,data=b,timeout=120); a.raise_for_status()
        self.call("POST","runs/update",json=dict(run_id=rid,status=it["status"],end_time=int(time.time()*1000)))
def main():
    a=argparse.ArgumentParser(); a.add_argument("--dry",action="store_true"); a.add_argument("--no-fetch",action="store_true"); a.add_argument("--only",default=""); x=a.parse_args()
    only=[k for k in x.only.split(",") if k]
    if not x.no_fetch: fetch()
    uri=os.environ.get("MLFLOW_TRACKING_URI")
    if not uri: sys.exit("MLFLOW_TRACKING_URI 미설정")
    import collections as C
    L=list(items(only)); print("items",dict(C.Counter(i["exp"] for i in L)),"failed",sum(i["status"]=="FAILED" for i in L),flush=True)
    if x.dry: return
    ml=ML(uri); new=[]; bad=0; repl=0
    for ex in sorted({i["exp"] for i in L}):
        e=ml.experiment_id(EXP+ex); old=ml.existing(e)
        for i in L:
            if i["exp"]!=ex: continue
            o=old.get(i["id"])
            if o and o[1]==i["key"]: continue
            try:
                ml.put(e,i,uri); new.append(i)
                if o: ml.call("POST","runs/delete",json=dict(run_id=o[0])); repl+=1
            except Exception as er: bad+=1; print("ERR",ex,i["name"],str(er)[:160],flush=True)
            if len(new)%100==99: print(len(new)+1,flush=True)
    print("logged",len(new),"replaced",repl,"errors",bad,time.strftime("%m-%d %H:%M"))
    fails=[f"{i['exp']}/{i['name']}" for i in new if i["status"]=="FAILED"]
    for n in fails: print(f"::warning::실험 실패 {n}")
    out=os.environ.get("GITHUB_STEP_SUMMARY")
    if out:
        with open(out,"a") as f:
            f.write(f"### gpu-study 수집\n\n새 run {len(new)}건 (교체 {repl}), 업로드 오류 {bad}건, 새 실패 {len(fails)}건\n\n")
            f.write("| experiment | 건수 |\n|---|---:|\n"+"".join(f"| {EXP}{a} | {b} |\n" for a,b in C.Counter(i["exp"] for i in new).most_common()))
            if fails: f.write("\n실패:\n"+"".join(f"- `{n}`\n" for n in fails[:50]))
    sys.exit(1 if bad else 0)
if __name__=="__main__": main()
