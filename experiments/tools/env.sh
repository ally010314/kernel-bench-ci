# env.sh <TRT 버전> — 장비별 TRT 환경을 잡는다. LD_PREFIX 는 devices.yml 의 ld_prefix.
v=$1
if [ "$(uname -m)" = aarch64 ]; then
  source /home/<user>/<team-dir>/venv/bin/activate
  LIB=/usr/lib/aarch64-linux-gnu; LP=/usr/local/cuda-12.6/lib64:$LIB
  export HF_HOME=/dev/shm/hf_exp   # Orin 디스크가 작다
else
  source ~/<venv>/bin/activate
  V=~/trt_vers/ve_$v/lib/python3.12/site-packages
  export PYTHONPATH=$V
  LIB=$V/tensorrt_libs; LP="${LD_PREFIX:+$LD_PREFIX:}$LIB"
fi
export LIB LP PY=$(command -v python)
