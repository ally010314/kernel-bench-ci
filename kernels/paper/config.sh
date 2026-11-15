# config.sh — 모델 · 장비별 논문 커널 빌드 플래그. build.sh 가 읽는다.
#   공개본에서는 커널 최적화 플래그와 칩 · 모델별 타일 조합을 가렸습니다 (masked). 모델 크기 매크로만 남아 있습니다.
PAPER_BASE=""   # masked
#   paper_defs <model> → stdout 에 모델 크기 매크로 (fc1 M1 · K1 = hidden · dim)
paper_defs(){
  case "$1" in
    mixerL|vitL) echo "-DMLP_CLASS=FusedMlpL -DMLP_DATA=\"vitL\" -DMLP_M1=4096 -DMLP_K1=1024 -DMLP_M2=1024 -DMLP_K2=4096" ;;
    mixerB)      echo "-DMLP_CLASS=FusedMlpL -DMLP_DATA=\"mixerB\" -DMLP_M1=3072 -DMLP_K1=768 -DMLP_M2=768 -DMLP_K2=3072" ;;
    resmlpB)     echo "-DMLP_CLASS=FusedMlpR -DMLP_DATA=\"resmlpB\" -DMLP_M1=3072 -DMLP_K1=768 -DMLP_M2=768 -DMLP_K2=3072" ;;
    deitS)       echo "-DMLP_CLASS=FusedMlpR -DMLP_DATA=\"deitS\" -DMLP_M1=1536 -DMLP_K1=384 -DMLP_M2=384 -DMLP_K2=1536" ;;
    *) echo "paper_defs: 설정 없음 $1" >&2; return 1 ;;
  esac
}
#   frozen_flags <arch> <model> → 장비 · 모델별 커널 플래그 (masked)
frozen_flags(){ echo ""; }
