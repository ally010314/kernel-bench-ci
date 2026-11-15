#!/usr/bin/env bash
# build.sh — 논문 커널 → TRT 플러그인(.so) + 벤치 하네스
#   ARCH: sm_87 (Orin, 기본) / sm_89 (RTX Ada)
#   MODEL: mixerL (기본) / vitL / mixerB / resmlpB / deitS — 크기 · 타일은 kernels/paper/config.sh
#   CUTLASS_DIR: 기본 third_party/cutlass (submodule)
set -euo pipefail
cd "$(dirname "$0")"
ARCH=${ARCH:-sm_87}
MODEL=${MODEL:-mixerL}
CUTLASS_DIR=${CUTLASS_DIR:-third_party/cutlass}
CUDA=${CUDA:-/usr/local/cuda}
NVCC=$CUDA/bin/nvcc
TRT_INC=${TRT_INC:-/usr/include/$(uname -m)-linux-gnu}
TRT_LIB=${TRT_LIB:-/usr/lib/$(uname -m)-linux-gnu}
OUT=build
mkdir -p $OUT

. kernels/paper/config.sh
INC="-I kernels/paper -I kernels -I $CUTLASS_DIR/include -I $CUTLASS_DIR/tools/util/include"
FLAGS="$PAPER_BASE $(frozen_flags "$ARCH" "$MODEL") -DN_PAD_MAX=3200"
eval "DEFS=($(paper_defs "$MODEL" | sed 's/"/\\"/g'))"      # MLP_DATA 의 따옴표를 그대로 넘긴다
BUILD_PLUGIN=${BUILD_PLUGIN:-1}   # 0 이면 TRT 플러그인 생략 (TensorRT 없는 빌드 환경용)
echo "[1/3] kernel object ($ARCH $MODEL)"
$NVCC -arch=$ARCH -std=c++17 $FLAGS "${DEFS[@]}" -Xcompiler -fPIC -c $INC kernels/paper/fused_mlp.cu -o $OUT/fused_mlp.o
if [ "$BUILD_PLUGIN" = 1 ]; then
  echo "[2/3] TRT plugin (libfused_mlp.so)"
  g++ -std=c++17 -O2 -shared -fPIC -I kernels/paper -I $TRT_INC -I $CUDA/include "${DEFS[@]}" \
    kernels/paper/fused_mlp_plugin.cpp $OUT/fused_mlp.o \
    -L $TRT_LIB -L $CUDA/lib64 -lnvinfer -lcudart -o $OUT/libfused_mlp.so
else
  echo "[2/3] TRT plugin skipped (BUILD_PLUGIN=0)"
fi
echo "[3/3] bench harness (mlp_bench)"
$NVCC -arch=$ARCH -O3 -std=c++17 "${DEFS[@]}" -I kernels/paper bench/mlp_bench.cu $OUT/fused_mlp.o -lcublas -o $OUT/mlp_bench
echo "done: $OUT/libfused_mlp.so $OUT/mlp_bench"
