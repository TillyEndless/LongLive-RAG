#!/usr/bin/env python3
from pathlib import Path
import subprocess

ROOT=Path('/data/zxl/LongLive-RAG-group11_15_h200')
OUT=ROOT/'results/group14_15_corrected_sparse/group15_dense50'
PY='/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'
LOG=ROOT/'results/group14_15_corrected_sparse/group15_dense50_gpu1.log'
OUT.mkdir(parents=True,exist_ok=True)
for i in range(1,11):
    case=f'case{i:02d}'
    cfg=ROOT/f'results/group14_15_corrected_sparse/configs/group15_dense50_{case}.yaml'
    out=OUT/case; out.mkdir(parents=True,exist_ok=True)
    valid=bool(list(out.glob('*.mp4'))) and any(p.name.endswith('_runtime.json') for p in out.glob('*.json'))
    if valid:
        with LOG.open('a') as f: f.write(f'SKIP_VALID {case}\n')
        continue
    with LOG.open('a') as f:
        f.write(f'START gpu=1 {case}\n'); f.flush()
        rc=subprocess.run(['env','CUDA_VISIBLE_DEVICES=1',PY,'-u',str(ROOT/'inference.py'),'--config_path',str(cfg)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
        f.write(f'END {case} rc={rc}\n')
    if rc: raise SystemExit(rc)
with LOG.open('a') as f: f.write('INFERENCE_COMPLETE group15_dense50 gpu1\n')
