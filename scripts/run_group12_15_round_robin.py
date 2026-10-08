#!/usr/bin/env python3
"""Detached Group12-15 scheduler with shared-GPU and permanent drain modes."""
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
GPUS = ['0', '1']
NORMAL_MIB = 30 * 1024
DRAIN_MIB = 30 * 1024

def now(): return datetime.now(timezone.utc).isoformat()
def key(label, case): return f'{label}/{case}'

def log(msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    with LOG.open('a') as f: f.write(line + '\n')

def save(state):
    state = dict(state); state['updated_at'] = now()
    tmp = STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=2) + '\n'); tmp.replace(STATE)

def free_mib(gpu):
    s = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'], text=True)
    return int(s.strip().splitlines()[0])

def gpu_pids(gpu):
    try:
        s = subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True)
        return [int(x.strip()) for x in s.splitlines() if x.strip().isdigit()]
    except Exception: return []

def cmdline(pid):
    try: return Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
    except Exception: return ''

def ownership(gpu):
    pids = gpu_pids(gpu); campaign = [p for p in pids if str(ROOT) in cmdline(p)]
    return campaign, [p for p in pids if p not in campaign]

def heartbeat(state, current=None):
    free_by_gpu = {gpu: free_mib(gpu) for gpu in GPUS}
    owners = {gpu: ownership(gpu) for gpu in GPUS}
    campaign = sorted({p for gpu in GPUS for p in owners[gpu][0]})
    external = sorted({p for gpu in GPUS for p in owners[gpu][1]})
    h = {'timestamp':now(), 'gpu':None, 'free_vram_gib':None,
         'gpu_utilization':None, 'scheduler_mode':state['scheduler_mode'],
         'current_job':current, 'campaign_gpu_pids':campaign, 'external_gpu_pids':external,
         'gpu_status':{gpu:{'free_vram_gib':free_by_gpu[gpu]/1024,'campaign_pids':owners[gpu][0],'external_pids':owners[gpu][1]} for gpu in GPUS},
         'normal_queue':state.get('normal_queue',[]), 'retry_queue':state.get('retry_queue',[]),
         'drain_queue':state.get('drain_queue',[]), 'evaluator_state':state.get('evaluator_state','PENDING')}
    try:
        h['gpu_utilization'] = {gpu: subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=utilization.gpu','--format=csv,noheader,nounits'], text=True).strip() for gpu in GPUS}
    except Exception: pass
    with HEARTBEAT.open('a') as f: f.write(json.dumps(h) + '\n')
    return free_by_gpu, campaign, external

def choose_gpu(threshold):
    free = {gpu: free_mib(gpu) for gpu in GPUS}
    eligible = [gpu for gpu in GPUS if free[gpu] >= threshold]
    return (max(eligible, key=lambda gpu: free[gpu]), free) if eligible else (None, free)

def video_ok(group, case):
    out = Path(group['jobs'][case]['output'])
    return any(p.stat().st_size > 0 for p in out.rglob('*.mp4')) if out.exists() else False

def first_incomplete(group):
    for case in group['cases']:
        if not video_ok(group, case): return case
    return None

def oom(text, rc):
    return rc != 0 and bool(re.search(r'out of memory|CUDA error: out of memory|CUBLAS_STATUS_ALLOC_FAILED|alloc.*failed', text, re.I))

manifest = json.loads(MANIFEST.read_text())
groups = {}
order = []
for j in manifest['jobs']:
    label = j['label']
    if label not in groups:
        groups[label] = {'label':label, 'group':j['group'], 'cases':[], 'jobs':{}, 'oom_count':0, 'last_oom_case':None, 'failed':None}
        order.append(label)
    if j['case'] not in groups[label]['jobs']:
        groups[label]['cases'].append(j['case']); groups[label]['jobs'][j['case']] = j

state = {'scheduler_mode':'SHARED_GPU_NORMAL','normal_threshold_gib':30,'drain_threshold_gib':30,
         'normal_queue':order[:], 'retry_queue':[], 'drain_queue':[], 'groups':{},
         'evaluator_state':'PENDING', 'current_job':None}
for label, g in groups.items():
    state['groups'][label] = {'first_incomplete_case':first_incomplete(g), 'completed_cases':[c for c in g['cases'] if video_ok(g,c)],
                              'eval_complete_cases':[], 'oom_count':0, 'last_oom_case':None, 'status':'PENDING'}
save(state); log(f'ROUND_ROBIN_START gpus=0,1 normal>=30GiB drain>=30GiB groups={len(order)}')

while state['normal_queue'] or state['retry_queue'] or state['drain_queue']:
    free_by_gpu, campaign, external = heartbeat(state, state.get('current_job'))
    if state['scheduler_mode'] == 'SHARED_GPU_NORMAL' and not external:
        state['scheduler_mode'] = 'DRAIN_MODE'
        state['drain_queue'] = state['normal_queue'] + state['retry_queue']
        state['normal_queue'] = []; state['retry_queue'] = []
        log(f'ENTER_DRAIN_MODE drain_queue={state["drain_queue"]}')
        save(state); continue
    if state['scheduler_mode'] == 'DRAIN_MODE':
        queue = state['drain_queue']; threshold = DRAIN_MIB
    else:
        # Normal groups first; retry groups are used once normal work is exhausted.
        queue = state['normal_queue'] if state['normal_queue'] else state['retry_queue']; threshold = NORMAL_MIB
    if not queue:
        time.sleep(20); continue
    selected_gpu, free_by_gpu = choose_gpu(threshold)
    if selected_gpu is None:
        time.sleep(20); continue
    label = queue.pop(0); g = groups[label]; case = first_incomplete(g)
    if case is None:
        state['groups'][label]['status'] = 'INFERENCE_COMPLETE'; continue
    j = g['jobs'][case]; out = Path(j['output']); out.mkdir(parents=True, exist_ok=True)
    state['current_job'] = key(label,case); state['groups'][label]['first_incomplete_case'] = case
    state['groups'][label]['status'] = 'RUNNING'; save(state)
    state['current_gpu'] = int(selected_gpu)
    log(f'START mode={state["scheduler_mode"]} job={label}/{case} gpu={selected_gpu} free_mib={free_by_gpu[selected_gpu]}')
    cmd = ['env',f'CUDA_VISIBLE_DEVICES={selected_gpu}',PYTHON,'-u',str(INFERENCE),'--config_path',j['config']]
    lp = out / 'inference.log'; start = time.time()
    with lp.open('a') as fh:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, text=True)
        while p.poll() is None:
            time.sleep(20); heartbeat(state, key(label,case))
    txt = lp.read_text(errors='replace')[-200000:]; rc = p.returncode
    state['current_job'] = None; state['current_gpu'] = None
    if rc == 0 and video_ok(g,case):
        st = state['groups'][label]; st['completed_cases'] = [c for c in g['cases'] if video_ok(g,c)]
        st['first_incomplete_case'] = first_incomplete(g); st['status'] = 'INFERENCE_COMPLETE' if st['first_incomplete_case'] is None else 'PARTIAL'
        log(f'COMPLETE {label}/{case} elapsed_s={time.time()-start:.1f}')
        if st['first_incomplete_case'] is not None: queue.append(label)
    elif oom(txt,rc):
        st = state['groups'][label]; st['oom_count'] += 1; st['last_oom_case'] = case; st['status']='OOM_RETRY_PENDING'
        if state['scheduler_mode'] == 'DRAIN_MODE': state['drain_queue'].append(label)
        else: state['retry_queue'].append(label)
        log(f'OOM_RETRY_PENDING {label}/{case} queue_tail')
    else:
        state['groups'][label]['status'] = 'FAILED_NON_OOM'; state['groups'][label]['failed'] = {'case':case,'returncode':rc}
        log(f'FAILED_NON_OOM {label}/{case} rc={rc}')
    save(state)

state['evaluator_state'] = 'RUNNING'; state['stage'] = 'EVALUATING'; save(state); log('QUEUES_EMPTY_START_BATCH_EVALUATOR')
with (OUT/'evaluator.log').open('a') as fh:
    p = subprocess.run([PYTHON,'-u',str(EVALUATOR)], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
state['evaluator_state'] = 'COMPLETE' if p.returncode == 0 else 'FAILED'; state['stage'] = 'COMPLETE' if p.returncode == 0 else 'EVALUATION_FAILED'; save(state)
log(f'CAMPAIGN_END status={state["stage"]} evaluator_rc={p.returncode}')
