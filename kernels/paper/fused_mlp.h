// fused_mlp.h — 플러그인 ↔ 커널 경계 인터페이스. channel MLP(fc1 + GELU + fc2) 를 커널 2개로 융합한다.
//   MLP_CLASS 는 플러그인 이름이기도 하다 (ONNX op 이름과 같아야 한다).
#pragma once
#include <cuda_runtime.h>
#ifndef MLP_CLASS
#define MLP_CLASS FusedMlpL
#endif
#ifndef MLP_DATA
#define MLP_DATA "vitL"
#endif

class MLP_CLASS {
public:
  // M1,K1: fc1 [hidden,dim];  M2,K2: fc2 [dim,hidden]
  // block: >=0 이면 data/allblk/<MLP_DATA>_b{NN}_* 를 로드한다 (cwd 상대경로)
  MLP_CLASS(int M1, int K1, int M2, int K2, int block = -1);
  ~MLP_CLASS();
  // x_dev: [N, K1] fp16, out_dev: [N, M2] fp16
  int run(const void* x_dev, void* out_dev, int N, cudaStream_t stream);
private:
  struct Impl; Impl* impl_;
};
