#!/usr/bin/env python3
import json, os, re, subprocess, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
OUT = ROOT / 'results/group12_15_corrected_campaign'
MANIFEST = OUT / 'campaign_manifest.json'
STATE = OUT / 'campaign_state.json'
HEARTBEAT = OUT / 'heartbeat.jsonl'
LOG = OUT / 'campaign_scheduler.log'
PYTHON = '/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'
INFERENCE = ROOT / 'inference.py'
EVALUATOR = ROOT / 'scripts/evaluate_group12_15_corrected.py'
GPU = '1'
NORMAL_MIB = 35 * 1024
OOM_RETRY_MIB = 60 * 1024

def now(): return datetime.now(timezone.utc).isoformat()

def write_state(data):
    data = dict(data); data['updated_at'] = now()
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(STATE)

def log(msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    with LOG.open('a') as f: f.write(line + '\n')

def free_mib():
    out = subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True)
    return int(out.strip().splitlines()[0])

def heartbeat(stage, current, normal, retry, completed, eval_pending):
    free = free_mib()
    item = {'timestamp': now(), 'gpu': int(GPU), 'free_vram_gib': free/1024,
            'gpu_utilization': None, 'stage': stage, 'current_job': current,
            'completed_jobs': completed, 'normal_queue': len(normal),
            'oom_retry_queue': len(retry), 'evaluator_state': eval_pending}
    try:
        u = subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=utilization.gpu','--format=csv,noheader,nounits'], text=True)
        item['gpu_utilization'] = u.strip()
    except Exception: pass
    with HEARTBEAT.open('a') as f: f.write(json.dumps(item) + '\n')
    return free

def wait_for(job, threshold, normal, retry, completed, eval_pending):
    while True:
        free = heartbeat('WAITING_FOR_GPU', job, normal, retry, completed, eval_pending)
        write_state({'stage':'WAITING_FOR_GPU', 'gpu':GPU, 'free_mib':free,
                     'normal_threshold_mib':NORMAL_MIB, 'oom_retry_threshold_mib':OOM_RETRY_MIB,
                     'current_job':job, 'normal_queue':normal, 'oom_retry_queue':retry,
                     'completed_cases':completed, 'evaluator_state':eval_pending})
        if free >= threshold: return
        time.sleep(20)

def video_ok(job):
    out = Path(job['output'])
    vids = [p for p in out.rglob('*.mp4') if p.stat().st_size > 0] if out.exists() else []
    return bool(vids)

def is_oom(text, rc):
    return rc != 0 and bool(re.search(r'out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|alloc.*failed', text, re.I))

manifest = json.loads(MANIFEST.read_text())
jobs = manifest['jobs']
normal = list(range(len(jobs)))
retry = []
completed = []
failed = []
for i, job in enumerate(jobs):
    key = f"{job['label']}/{job['case']}"
    if video_ok(job): completed.append(key)
    else: job['_idx'] = i
normal = [i for i in normal if i not in {j['_idx'] for j in jobs if f"{j['label']}/{j['case']}" in completed}]
eval_pending = True
write_state({'stage':'STARTING','gpu':GPU,'normal_threshold_mib':NORMAL_MIB,
             'oom_retry_threshold_mib':OOM_RETRY_MIB,'normal_queue':normal,
             'oom_retry_queue':retry,'completed_cases':completed,'failed_cases':failed,
             'evaluator_state':'PENDING'})
log(f'POLICY_CAMPAIGN_START gpu={GPU} normal>=35GiB oom_retry>=60GiB jobs={len(normal)}')

while normal or retry:
    free = heartbeat('SCHEDULING', None, normal, retry, completed, eval_pending)
    if retry and free >= OOM_RETRY_MIB:
        idx = retry.pop(0)
        queue_name = 'OOM_RETRY_QUEUE'
    elif normal and free >= NORMAL_MIB:
        idx = normal.pop(0)
        queue_name = 'NORMAL_QUEUE'
    else:
        target = retry[0] if retry else (normal[0] if normal else None)
        threshold = OOM_RETRY_MIB if retry and not normal else NORMAL_MIB
        if target is not None: wait_for(jobs[target]['label']+'/'+jobs[target]['case'], threshold, normal, retry, completed, eval_pending)
        continue
    job = jobs[idx]; key = f"{job['label']}/{job['case']}"
    if video_ok(job):
        if key not in completed: completed.append(key)
        continue
    out = Path(job['output']); out.mkdir(parents=True, exist_ok=True)
    log(f'START {queue_name} {key} free_mib={free}')
    write_state({'stage':'RUNNING','gpu':GPU,'free_mib':free,'current_job':key,
                 'normal_queue':normal,'oom_retry_queue':retry,'completed_cases':completed,
                 'failed_cases':failed,'evaluator_state':eval_pending})
    cmd = ['env', f'CUDA_VISIBLE_DEVICES={GPU}', PYTHON, '-u', str(INFERENCE), '--config_path', job['config']]
    log_path = out / 'inference.log'
    start = time.time()
    with log_path.open('a') as fh:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True)
        while proc.poll() is None:
            time.sleep(20)
            heartbeat('RUNNING', key, normal, retry, completed, eval_pending)
    text = log_path.read_text(errors='replace')[-200000:]
    rc = proc.returncode
    if rc == 0 and video_ok(job):
        completed.append(key)
        log(f'COMPLETE {key} elapsed_s={time.time()-start:.1f}')
    elif is_oom(text, rc):
        retry.append(idx)
        log(f'OOM_RETRY_PENDING {key}; moved to OOM_RETRY_QUEUE')
    else:
        failed.append({'job':key,'returncode':rc})
        log(f'FAILED_NON_OOM {key} rc={rc}; continuing queue')
    write_state({'stage':'SCHEDULING','gpu':GPU,'free_mib':free_mib(),'current_job':None,
                 'normal_queue':normal,'oom_retry_queue':retry,'completed_cases':completed,
                 'failed_cases':failed,'evaluator_state':eval_pending})

write_state({'stage':'EVALUATING','gpu':GPU,'normal_queue':[],'oom_retry_queue':[],
             'completed_cases':completed,'failed_cases':failed,'evaluator_state':'RUNNING'})
log('INFERENCE_QUEUES_EMPTY; starting canonical evaluator')
with (OUT/'evaluator.log').open('a') as fh:
    proc = subprocess.run([PYTHON,'-u',str(EVALUATOR)], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
if proc.returncode == 0:
    write_state({'stage':'COMPLETE','gpu':GPU,'completed_cases':completed,'failed_cases':failed,
                 'evaluator_state':'COMPLETE','normal_queue':[],'oom_retry_queue':[]})
    log('POLICY_CAMPAIGN_COMPLETE')
else:
    write_state({'stage':'EVALUATION_FAILED','gpu':GPU,'completed_cases':completed,
                 'failed_cases':failed,'evaluator_state':'FAILED','returncode':proc.returncode})
    log(f'EVALUATION_FAILED rc={proc.returncode}')
    raise SystemExit(proc.returncode)
