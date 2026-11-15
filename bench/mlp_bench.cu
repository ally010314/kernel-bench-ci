// mlp_bench.cu — FusedMlpL 단독 실행 하네스 (CI 용)
//   - 작업 디렉토리의 가중치를 FusedMlpL 이 읽는다 (run.py 가 생성).
//   - 각 N 에 대해 x_N{n}.bin 을 읽어 실행, y_N{n}.bin 으로 출력 저장 (정확도 비교는 python 쪽).
//   - latency: warmup 후 iters 회 cudaEvent 로 개별 측정 → 중앙값/p10/p90.
//   - 비교 기준: 같은 shape 의 cuBLAS dense fp16 GEMM 2회 (fc1, fc2) — bias/GELU 제외라 기준이 오히려 유리.
//   - fc1 / fc2 분리: run() 1회를 CUDA Graph 로 캡처 → 노드를 실행 순서대로 하나씩 떼어 단독 graph 로 재실행·측정.
//     커널 인터페이스를 바꾸지 않고 run() 이 내는 launch 를 그대로 잰다 (첫 커널 = fc1+GELU, 다음 커널 = fc2+전치 store,
//     그 밖의 노드 = 패딩 memcpy 등 → other). 단독 실행이라 앞 커널이 남긴 L2 상태는 없다 → 합은 fused 와 약간 다를 수 있어
//     split_sum_ms 를 같이 낸다.
//   출력: N 당 JSON 한 줄 (stdout).
//
//   usage: mlp_bench <N,N,...> [iters=200] [warmup=50]
#include "fused_mlp.h"        // 논문 커널 (kernels/paper)
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#define CK(x) do{ cudaError_t ck_e=(x); if(ck_e!=cudaSuccess){ fprintf(stderr,"CUDA %s @%s:%d\n",cudaGetErrorString(ck_e),__FILE__,__LINE__); exit(2);} }while(0)
#define CB(x) do{ cublasStatus_t cb_s=(x); if(cb_s!=CUBLAS_STATUS_SUCCESS){ fprintf(stderr,"cuBLAS %d @%s:%d\n",(int)cb_s,__FILE__,__LINE__); exit(2);} }while(0)

#ifndef MLP_M1
#define MLP_M1 4096
#define MLP_K1 1024
#define MLP_M2 1024
#define MLP_K2 4096
#endif
#ifndef MLP_CLASS
#define MLP_CLASS FusedMlpL   // history/ 의 과거 커널
#endif
static constexpr int M1=MLP_M1, K1=MLP_K1, M2=MLP_M2, K2=MLP_K2;

static std::vector<__half> read_bin(const std::string& p, size_t n){
  FILE* f=fopen(p.c_str(),"rb"); if(!f){ fprintf(stderr,"open fail %s\n",p.c_str()); exit(2); }
  std::vector<__half> v(n); size_t r=fread(v.data(),sizeof(__half),n,f); fclose(f);
  if(r!=n){ fprintf(stderr,"size mismatch %s\n",p.c_str()); exit(2); } return v;
}
static void write_bin(const std::string& p, const std::vector<__half>& v){
  FILE* f=fopen(p.c_str(),"wb"); fwrite(v.data(),sizeof(__half),v.size(),f); fclose(f);
}

struct Stats { double med, p10, p90; };

// run() 이 낸 노드들을 실행 순서(선형 체인 가정)로 정렬해 돌려준다.
static std::vector<cudaGraphNode_t> chain_order(cudaGraph_t g){
  size_t n=0; CK(cudaGraphGetRootNodes(g,nullptr,&n));
  std::vector<cudaGraphNode_t> roots(n); CK(cudaGraphGetRootNodes(g,roots.data(),&n));
  std::vector<cudaGraphNode_t> out;
  if(n!=1){ fprintf(stderr,"split: root 노드 %zu 개 (선형 체인 아님)\n",n); return out; }
  cudaGraphNode_t cur=roots[0];
  while(true){
    out.push_back(cur);
    size_t m=0; CK(cudaGraphNodeGetDependentNodes(cur,nullptr,&m));
    if(m==0) break;
    if(m>1){ fprintf(stderr,"split: 분기 노드 (선형 체인 아님)\n"); out.clear(); return out; }
    CK(cudaGraphNodeGetDependentNodes(cur,&cur,&m));
  }
  return out;
}
template<class F> static Stats time_it(F&& fn, int iters, int warmup, cudaStream_t s){
  for(int i=0;i<warmup;i++) fn();
  CK(cudaStreamSynchronize(s));
  std::vector<cudaEvent_t> ev(iters+1);
  for(auto& e:ev) CK(cudaEventCreate(&e));
  CK(cudaEventRecord(ev[0],s));
  for(int i=0;i<iters;i++){ fn(); CK(cudaEventRecord(ev[i+1],s)); }
  CK(cudaStreamSynchronize(s));
  std::vector<double> t(iters);
  for(int i=0;i<iters;i++){ float ms; CK(cudaEventElapsedTime(&ms,ev[i],ev[i+1])); t[i]=ms; }
  for(auto& e:ev) cudaEventDestroy(e);
  std::sort(t.begin(),t.end());
  auto q=[&](double p){ return t[std::min<size_t>(t.size()-1,(size_t)(p*(t.size()-1)+0.5))]; };
  return {q(0.5),q(0.1),q(0.9)};
}

int main(int argc, char** argv){
  if(argc<2){ fprintf(stderr,"usage: %s <N,N,...> [iters] [warmup]\n",argv[0]); return 1; }
  std::vector<int> Ns; { std::string s=argv[1]; size_t p=0;
    while(p<s.size()){ size_t q=s.find(',',p); if(q==std::string::npos) q=s.size(); Ns.push_back(std::stoi(s.substr(p,q-p))); p=q+1; } }
  int iters = argc>2? atoi(argv[2]) : 200;
  int warmup= argc>3? atoi(argv[3]) : 50;

  cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop,0));
  cudaStream_t s; CK(cudaStreamCreate(&s));
  MLP_CLASS mlp(M1,K1,M2,K2,/*block=*/0);   // 5-인자: 07-19 판(res_mode 없음)과도 호환

  cublasHandle_t h; CB(cublasCreate(&h)); CB(cublasSetStream(h,s));
  int maxN=*std::max_element(Ns.begin(),Ns.end());
  __half *dW1,*dW2,*dH;
  CK(cudaMalloc(&dW1,sizeof(__half)*M1*K1)); CK(cudaMalloc(&dW2,sizeof(__half)*M2*K2));
  CK(cudaMalloc(&dH,sizeof(__half)*(size_t)M1*maxN));
  CK(cudaMemset(dW1,0,sizeof(__half)*M1*K1)); CK(cudaMemset(dW2,0,sizeof(__half)*M2*K2));

  for(int N:Ns){
    auto hx=read_bin("x_N"+std::to_string(N)+".bin",(size_t)N*K1);
    __half *dx,*dy,*dy2;
    CK(cudaMalloc(&dx,sizeof(__half)*N*K1)); CK(cudaMalloc(&dy,sizeof(__half)*N*M2)); CK(cudaMalloc(&dy2,sizeof(__half)*N*M2));
    CK(cudaMemcpy(dx,hx.data(),sizeof(__half)*N*K1,cudaMemcpyHostToDevice));

    if(mlp.run(dx,dy,N,s)!=0){ fprintf(stderr,"run failed N=%d\n",N); return 3; }
    CK(cudaStreamSynchronize(s));
    std::vector<__half> hy((size_t)N*M2);
    CK(cudaMemcpy(hy.data(),dy,sizeof(__half)*N*M2,cudaMemcpyDeviceToHost));
    write_bin("y_N"+std::to_string(N)+".bin",hy);

    auto fused=time_it([&]{ mlp.run(dx,dy,N,s); },iters,warmup,s);
    // dense 기준: row-major x[N,K1] · W1ᵀ → H[N,M1], H · W2ᵀ → y[N,M2]  (column-major 관점에서 W·xᵀ)
    const __half one=__float2half(1.f), zero=__float2half(0.f);
    auto dense=time_it([&]{
      CB(cublasHgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,M1,N,K1,&one,dW1,K1,dx,K1,&zero,dH,M1));
      CB(cublasHgemm(h,CUBLAS_OP_T,CUBLAS_OP_N,M2,N,K2,&one,dW2,K2,dH,K2,&zero,dy2,M2));
    },iters,warmup,s);

    // ── fc1 / fc2 분리 ──
    double fc1=-1, fc2=-1, other=0; int nk=0;
    {
      cudaGraph_t g; CK(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal));
      int rc=mlp.run(dx,dy,N,s);
      CK(cudaStreamEndCapture(s,&g));
      if(rc!=0){ fprintf(stderr,"capture run failed N=%d\n",N); return 3; }
      for(cudaGraphNode_t nd: chain_order(g)){
        cudaGraphNodeType t; CK(cudaGraphNodeGetType(nd,&t));
        cudaGraph_t g1; CK(cudaGraphCreate(&g1,0)); cudaGraphNode_t c;
        if(t==cudaGraphNodeTypeKernel){
          cudaKernelNodeParams kp; CK(cudaGraphKernelNodeGetParams(nd,&kp));
          CK(cudaGraphAddKernelNode(&c,g1,nullptr,0,&kp));
        } else if(t==cudaGraphNodeTypeMemcpy){
          cudaMemcpy3DParms mp; CK(cudaGraphMemcpyNodeGetParams(nd,&mp));
          CK(cudaGraphAddMemcpyNode(&c,g1,nullptr,0,&mp));
        } else if(t==cudaGraphNodeTypeMemset){
          cudaMemsetParams ms; CK(cudaGraphMemsetNodeGetParams(nd,&ms));
          CK(cudaGraphAddMemsetNode(&c,g1,nullptr,0,&ms));
        } else { fprintf(stderr,"split: 처리 못하는 노드 type %d\n",(int)t); CK(cudaGraphDestroy(g1)); continue; }
        cudaGraphExec_t ge; CK(cudaGraphInstantiate(&ge,g1,0));
        double ms=time_it([&]{ CK(cudaGraphLaunch(ge,s)); },iters,warmup,s).med;
        if(t==cudaGraphNodeTypeKernel){ nk++; if(nk==1) fc1=ms; else if(nk==2) fc2=ms; else other+=ms; }
        else other+=ms;
        CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g1));
      }
      CK(cudaGraphDestroy(g));
    }

    printf("{\"N\": %d, \"fused_ms\": %.5f, \"fused_p10\": %.5f, \"fused_p90\": %.5f, "
           "\"dense_ms\": %.5f, \"dense_p10\": %.5f, \"dense_p90\": %.5f, "
           "\"fc1_ms\": %.5f, \"fc2_ms\": %.5f, \"other_ms\": %.5f, \"split_sum_ms\": %.5f, \"n_kernels\": %d, "
           "\"gpu\": \"%s\", \"sm\": %d%d}\n",
           N,fused.med,fused.p10,fused.p90,dense.med,dense.p10,dense.p90,
           fc1,fc2,other,fc1+fc2+other,nk,prop.name,prop.major,prop.minor);
    fflush(stdout);
    cudaFree(dx); cudaFree(dy); cudaFree(dy2);
  }
  cudaFree(dW1); cudaFree(dW2); cudaFree(dH); cublasDestroy(h); cudaStreamDestroy(s);
  return 0;
}
