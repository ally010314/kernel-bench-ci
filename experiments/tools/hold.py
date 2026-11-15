#!/usr/bin/env python3
"""hold.py <GB> <남길 GB> — 측정 중 다른 작업이 GPU 메모리를 차지하지 못하게 예약한다. 계산은 하지 않는다.

worker 의 자식으로 돌고, 부모가 사라지면 스스로 끝난다. worker 가 비정상 종료해도 예약이 남지 않는다."""
import os, sys, time
import torch

want, keep = float(sys.argv[1]), float(sys.argv[2])
ppid = os.getppid()
free = torch.cuda.mem_get_info()[0] / 2**30
gb = max(0.0, min(want, free - keep))
x = torch.empty(int(gb * 2**30), dtype=torch.uint8, device="cuda") if gb > 0 else None
print(f"held {gb:.1f} GB, free before {free:.1f} GB", flush=True)
while os.getppid() == ppid:
    time.sleep(5)
