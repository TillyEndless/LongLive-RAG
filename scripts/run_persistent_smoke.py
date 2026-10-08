#!/usr/bin/env python3
import json, subprocess
from pathlib import Path

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
CFG = ROOT / 'configs/group12_15_corrected_campaign'
OUT = ROOT / 'results/group12_15_corrected_campaign/smoke_persistent'
PYTHON = '/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'
INFERENCE = ROOT / 'inference.py'
jobs = [
    ('group12', 'group12_case01.yaml'),
    ('group13', 'group13_case01.yaml'),
    ('group14_sparse20', 'group14_sparse20_case01.yaml'),
    ('group15_sparse20', 'group15_sparse20_case01.yaml'),
]
state = {'backend': 'BF16_attention_with_persistent_GPU_lowbit_owner', 'jobs': []}
for label, source_name in jobs:
    source = CFG / source_name
    out = OUT / label / 'case01'
    config = CFG / f'persistent_smoke_{label}.yaml'
    text = source.read_text()
    old_out = next(line.split(': ', 1)[1].strip() for line in text.splitlines() if line.startswith('output_folder:'))
    text = text.replace(f'output_folder: {old_out}', f'output_folder: {out}')
    text = text.replace('num_output_frames: 474', 'num_output_frames: 30')
    config.write_text(text)
    out.mkdir(parents=True, exist_ok=True)
    log = out / 'inference.log'
    cmd = ['env', 'CUDA_VISIBLE_DEVICES=1', PYTHON, '-u', str(INFERENCE), '--config_path', str(config)]
    with log.open('w') as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    videos = list(out.rglob('*.mp4'))
    runtime = list(out.rglob('*runtime.json'))
    ok = rc == 0 and bool(videos) and bool(runtime)
    state['jobs'].append({'label': label, 'returncode': rc, 'output': str(out), 'owner_verified_by_runtime': bool(runtime), 'status': 'PASS' if ok else 'FAIL'})
    (OUT / 'smoke_state.json').write_text(json.dumps(state, indent=2) + '\n')
    if not ok:
        raise SystemExit(f'smoke failed: {label} rc={rc}')
print(json.dumps(state, indent=2))
