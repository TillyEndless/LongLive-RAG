#!/usr/bin/env python3
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
CFG = ROOT / 'configs/group12_15_corrected_campaign'
OUT = ROOT / 'results/group12_15_corrected_campaign'
STATE = OUT / 'campaign_state.json'
LOG = OUT / 'campaign_scheduler.log'
PYTHON = '/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'
INFERENCE = ROOT / 'inference.py'
EVALUATOR = ROOT / 'scripts/evaluate_group12_15_corrected.py'
GPU = '1'
MIN_FREE_MIB = 100000

def now():
    return datetime.now(timezone.utc).isoformat()

def write_state(payload):
    payload = dict(payload)
    payload['updated_at'] = now()
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(STATE)

def log(msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    with LOG.open('a') as f:
        f.write(line + '\n')

def free_mib():
    out = subprocess.check_output([
        'nvidia-smi', '-i', GPU, '--query-gpu=memory.free', '--format=csv,noheader,nounits'
    ], text=True)
    return int(out.strip().splitlines()[0])

def wait_for_gpu(job, required_free_mib=MIN_FREE_MIB):
    while True:
        free = free_mib()
        write_state({'stage': 'WAITING_FOR_GPU' if free < required_free_mib else 'READY',
                     'gpu': GPU, 'free_mib': free, 'required_free_mib': required_free_mib,
                     'current_job': job})
        if free >= required_free_mib:
            return
        log(f'WAITING_FOR_GPU job={job} free_mib={free} required={required_free_mib}')
        time.sleep(20)

def outputs_ok(outdir):
    videos = list(outdir.glob('*.mp4')) + list(outdir.rglob('*.mp4'))
    return bool(videos) and any(p.stat().st_size > 0 for p in videos)

def run_job(job, smoke=False):
    cfg = Path(job['config'])
    outdir = Path(job['output'])
    run_cfg = cfg
    run_out = outdir
    if smoke:
        run_out = OUT / 'smoke' / job['label'] / job['case']
        run_cfg = CFG / f'smoke_{job["label"]}_{job["case"]}.yaml'
        if not run_cfg.exists():
            text = cfg.read_text()
            text = text.replace(f'output_folder: {outdir}', f'output_folder: {run_out}')
            text = text.replace('num_output_frames: 474', 'num_output_frames: 30')
            run_cfg.write_text(text)
    run_out.mkdir(parents=True, exist_ok=True)
    if outputs_ok(run_out):
        log(f'SKIP existing output job={job["label"]}/{job["case"]}')
        return True
    required = 35000 if smoke else MIN_FREE_MIB
    wait_for_gpu(f'{job["label"]}/{job["case"]}', required_free_mib=required)
    command = ['env', f'CUDA_VISIBLE_DEVICES={GPU}', PYTHON, '-u', str(INFERENCE),
               '--config_path', str(run_cfg)]
    log(f'START job={job["label"]}/{job["case"]} command={" ".join(command)}')
    write_state({'stage': 'RUNNING_SMOKE' if smoke else 'RUNNING', 'gpu': GPU,
                 'current_job': job, 'command': command})
    joblog = run_out / 'inference.log'
    with joblog.open('a') as fh:
        proc = subprocess.run(command, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    ok = proc.returncode == 0 and outputs_ok(run_out)
    log(f'END job={job["label"]}/{job["case"]} rc={proc.returncode} outputs_ok={ok}')
    if not ok:
        log(f'RETRY_ONCE job={job["label"]}/{job["case"]}')
        wait_for_gpu(f'retry:{job["label"]}/{job["case"]}', required_free_mib=required)
        with joblog.open('a') as fh:
            fh.write('\n[retry]\n')
            proc = subprocess.run(command, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        ok = proc.returncode == 0 and outputs_ok(run_out)
        log(f'RETRY_END job={job["label"]}/{job["case"]} rc={proc.returncode} outputs_ok={ok}')
    return ok

manifest = json.loads((OUT / 'campaign_manifest.json').read_text())
jobs = manifest['jobs']
by = {(j['label'], j['case']): j for j in jobs}

smoke_keys = [
    ('group12', 'case01'), ('group13', 'case01'),
    ('group14_sparse20', 'case01'), ('group15_sparse20', 'case01')
]
ordered = smoke_keys + [
    (j['label'], j['case']) for j in jobs
    if (j['label'], j['case']) not in smoke_keys
]

write_state({'stage': 'STARTING', 'gpu': GPU, 'queued_jobs': len(ordered),
             'completed_jobs': 0, 'failed_jobs': []})
log(f'CAMPAIGN_START jobs={len(ordered)} gpu={GPU} min_free_mib={MIN_FREE_MIB}')

for index, key in enumerate(ordered, 1):
    job = by[key]
    smoke = key in smoke_keys
    if smoke:
        log(f'SMOKE_GATE {index}/{len(ordered)} {key[0]}/{key[1]}')
    ok = run_job(job, smoke=smoke)
    if not ok:
        write_state({'stage': 'FAILED', 'gpu': GPU, 'failed_job': job,
                     'failed_index': index, 'queued_jobs': len(ordered)})
        raise SystemExit(2)
    if smoke:
        log(f'SMOKE_PASS {key[0]}/{key[1]}')
    write_state({'stage': 'RUNNING', 'gpu': GPU, 'completed_jobs': index,
                 'queued_jobs': len(ordered), 'last_job': job})

write_state({'stage': 'INFERENCE_COMPLETE', 'gpu': GPU, 'completed_jobs': len(ordered),
             'queued_jobs': len(ordered)})
log('INFERENCE_COMPLETE; starting unified canonical evaluator')
write_state({'stage': 'EVALUATING', 'gpu': GPU, 'completed_jobs': len(ordered),
             'queued_jobs': len(ordered), 'evaluator': str(EVALUATOR)})
eval_log = OUT / 'evaluator.log'
with eval_log.open('a') as fh:
    proc = subprocess.run([PYTHON, '-u', str(EVALUATOR)], cwd=ROOT,
                          stdout=fh, stderr=subprocess.STDOUT)
if proc.returncode != 0:
    write_state({'stage': 'EVALUATION_FAILED', 'gpu': GPU, 'returncode': proc.returncode})
    raise SystemExit(proc.returncode)
write_state({'stage': 'COMPLETE', 'gpu': GPU, 'completed_jobs': len(ordered),
             'queued_jobs': len(ordered), 'final_table': str(OUT / 'group12_15_final_10case.csv')})
log('CAMPAIGN_COMPLETE')
