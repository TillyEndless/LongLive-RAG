#!/usr/bin/env python3
"""Detached 10-latent-block smoke campaign for corrected Groups 12--15.

This intentionally does not check free VRAM: the user requested forced launch.
At most one process is assigned to each GPU. Failed jobs are appended to the
queue and retried until they produce a valid 10-block smoke artifact.
"""
from __future__ import annotations
import collections
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
CFG_ROOT = ROOT / 'configs/group12_15_persistent_campaign'
OUT_ROOT = ROOT / 'results/group12_15_persistent_campaign/smoke_1block'
STATE = OUT_ROOT / 'runner_state.json'
LOG = OUT_ROOT / 'runner.log'
PYTHON = '/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'
INFERENCE = ROOT / 'inference.py'
GPUS = [x.strip() for x in os.environ.get('SMOKE_GPUS', '0,1').split(',') if x.strip()]
JOBS = [
    'group12', 'group13',
    'group14_sparse30', 'group14_sparse20', 'group14_sparse10', 'group14_sparse05',
    'group15_sparse30', 'group15_sparse20', 'group15_sparse10', 'group15_sparse05',
]

def now():
    return datetime.now(timezone.utc).isoformat()

def log(msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    with LOG.open('a') as f:
        f.write(line + '\n')

def write_state(pending, active, completed, failed):
    payload = {
        'updated_at': now(), 'stage': 'RUNNING' if pending or active else 'COMPLETE',
        'pending': list(pending),
        'active': {gpu: x['label'] for gpu, x in active.items()},
        'completed': completed, 'failed_attempts': failed,
        'output_root': str(OUT_ROOT), 'block_contract': '1 latent block = 3 latent frames',
    }
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(STATE)

def source_config(label):
    matches = sorted(CFG_ROOT.glob(f'{label}_case01.yaml'))
    if not matches:
        raise FileNotFoundError(f'no case01 config for {label}')
    return matches[0]

def make_smoke_config(label):
    src = source_config(label)
    outdir = OUT_ROOT / label / 'case01'
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = OUT_ROOT / 'configs' / f'{label}_case01.yaml'
    cfg.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text()
    lines = []
    replaced_output = False
    replaced_frames = False
    for line in text.splitlines():
        if line.startswith('output_folder:'):
            lines.append(f'output_folder: {outdir}')
            replaced_output = True
        elif line.startswith('num_output_frames:'):
            lines.append('num_output_frames: 3')
            replaced_frames = True
        else:
            lines.append(line)
    if not replaced_output:
        lines.append(f'output_folder: {outdir}')
    if not replaced_frames:
        lines.append('num_output_frames: 3')
    lines.append('smoke_1block: true')
    cfg.write_text('\n'.join(lines) + '\n')
    return cfg, outdir

def valid(outdir):
    videos = [p for p in outdir.glob('*.mp4') if p.stat().st_size > 0]
    return bool(videos) and any(
        (p.with_name(p.stem + '_runtime.json')).exists() and
        (p.with_name(p.stem + '_memory_measurement.json')).exists()
        for p in videos
    )

def start(gpu, label):
    cfg, outdir = make_smoke_config(label)
    logpath = outdir / 'smoke_inference.log'
    fh = logpath.open('a')
    cmd = ['env', f'CUDA_VISIBLE_DEVICES={gpu}', PYTHON, '-u', str(INFERENCE), '--config_path', str(cfg)]
    log(f'START gpu={gpu} label={label} command={" ".join(cmd)}')
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    return {'label': label, 'proc': proc, 'fh': fh, 'outdir': outdir, 'cfg': cfg, 'started': now()}

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    pending = collections.deque(
        label for label in JOBS
        if not valid(OUT_ROOT / label / 'case01')
    )
    active = {}
    completed = []
    failed = collections.Counter()
    log(f'SMOKE_RUNNER_START jobs={len(JOBS)} gpus={GPUS} forced_vram=true')
    while pending or active:
        for gpu in GPUS:
            if gpu not in active and pending:
                label = pending.popleft()
                active[gpu] = start(gpu, label)
        write_state(pending, active, completed, dict(failed))
        time.sleep(5)
        for gpu, item in list(active.items()):
            rc = item['proc'].poll()
            if rc is None:
                continue
            item['fh'].close()
            label = item['label']
            ok = rc == 0 and valid(item['outdir'])
            log(f'END gpu={gpu} label={label} rc={rc} valid={ok}')
            del active[gpu]
            if ok:
                completed.append(label)
                log(f'SMOKE_PASS label={label} completed={len(completed)}/{len(JOBS)}')
            else:
                failed[label] += 1
                tail = item['outdir'] / 'smoke_inference.log'
                if tail.exists():
                    lines = tail.read_text(errors='replace').splitlines()[-12:]
                    log(f'FAIL_TAIL label={label}\\n' + '\\n'.join(lines))
                pending.append(label)
                log(f'REQUEUE label={label} attempt={failed[label]} queue={list(pending)}')
        write_state(pending, active, completed, dict(failed))
    write_state(pending, active, completed, dict(failed))
    log('SMOKE_RUNNER_COMPLETE')

if __name__ == '__main__':
    main()
