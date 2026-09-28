#!/usr/bin/env python3
"""Resumable dual-GPU RR runner for the corrected Group12--15 campaign."""
from __future__ import annotations
import collections
import json
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
MANIFEST = ROOT / 'results/group12_15_persistent_campaign/campaign_manifest.json'
RUN_ROOT = ROOT / 'results/group12_15_persistent_campaign/canonical10_rr'
STATE = RUN_ROOT / 'runner_state.json'
LOG = RUN_ROOT / 'runner.log'
PYTHON = '/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'
EVAL_PYTHON = '/data/zxl/SolarWM/.env_vbench310/bin/python'
INFERENCE = ROOT / 'inference.py'
EVALUATOR = ROOT / 'scripts/evaluate_group12_15_corrected.py'
GPUS = ['0', '1']
NORMAL_FREE_MIB = 30 * 1024
RETRY_FREE_MIB = 30 * 1024

def now():
    return datetime.now(timezone.utc).isoformat()

def log(msg):
    line = f'[{now()}] {msg}'
    print(line, flush=True)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        f.write(line + '\n')

def save_state(pending, active, completed, failed, stage='RUNNING', extra=None):
    data = {
        'updated_at': now(), 'stage': stage,
        'pending': list(pending),
        'active': {gpu: item['key'] for gpu, item in active.items()},
        'completed': sorted(completed), 'failed_attempts': dict(failed),
        'normal_threshold_mib': NORMAL_FREE_MIB, 'retry_threshold_mib': RETRY_FREE_MIB,
        'gpus': GPUS, 'output_root': str(ROOT / 'results/group12_15_persistent_campaign'),
    }
    if extra:
        data.update(extra)
    tmp = STATE.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(STATE)

def ordered_jobs(manifest):
    jobs = {(j['label'], j['case']): j for j in manifest['jobs']}
    labels = ['group12', 'group13', 'group14_sparse30', 'group14_sparse20',
              'group14_sparse10', 'group14_sparse05', 'group15_sparse30',
              'group15_sparse20', 'group15_sparse10', 'group15_sparse05']
    missing=[]
    for label in labels:
        cases={case for (lab, case) in jobs if lab == label}
        expected={f'case{i:02d}' for i in range(1,11)}
        if cases != expected:
            missing.append(f'{label}: {sorted(cases)}')
    if missing:
        raise RuntimeError('CANONICAL10_MANIFEST_GATE_FAILED: ' + '; '.join(missing))
    out = []
    for i in range(1, 11):
        case = f'case{i:02d}'
        for label in labels:
            if (label, case) in jobs:
                j = dict(jobs[(label, case)])
                j['key'] = f'{label}/{case}'
                out.append(j)
    return out

def make_config(job):
    src = Path(job['config'])
    dst = RUN_ROOT / 'configs' / f"{job['label']}_{job['case']}.yaml"
    dst.parent.mkdir(parents=True, exist_ok=True)
    text = src.read_text()
    # Source campaign configs predate the corrected contract and encode the
    # removal percentage in their sparse label.  The runtime field is the
    # retained interaction ratio, so normalize only the generated snapshot.
    label = str(job['label'])
    if '_sparse' in label:
        removed = int(label.rsplit('_sparse', 1)[1])
        retained = 1.0 - removed / 100.0
        text = re.sub(r'(?m)^(\s*group_sparse_ratio:\s*)[^\n]+$',
                      rf'\g<1>{retained:.2f}', text)
        text = re.sub(r'(?m)^ratio_semantics:\s*[^\n]+$',
                      'ratio_semantics: retained_interaction_ratio', text)
        if 'ratio_semantics:' not in text:
            text += '\nratio_semantics: retained_interaction_ratio\n'
    text = re.sub(r'^skip_existing:\s*true\s*$', 'skip_existing: false', text, flags=re.M)
    text += '\n# canonical10_rr runner snapshot\n'
    dst.write_text(text)
    return dst

def runtime_ok(path, group):
    try:
        obj = json.loads(path.read_text())
    except Exception:
        return False
    mode = {12:'group12_corrected',13:'group13_corrected',14:'group14_corrected',15:'group15_corrected'}[group]
    owner = 'INT8/FP8_E4M3' if group in (12,14) else 'NVFP4/NVFP4'
    return all(obj.get(k) == v for k,v in {
        'GROUP_RUNTIME_MODE': mode,
        'CPU_COMPRESSED_HISTORY_ACTIVE': 'NO',
        'CPU_HISTORICAL_BF16_ARCHIVE_ACTIVE': 'YES',
        'PERSISTENT_STORAGE_MODE': 'LOWBIT_STORAGE_BF16_COMPUTE',
        'GPU_PERSISTENT_KV_OWNER': owner,
        'FINAL_ATTENTION_DTYPE': 'bfloat16',
        'NATIVE_LOWBIT_KERNEL_USED': 'NO',
    }.items())

def valid_output(job):
    out = Path(job['output'])
    videos = sorted(out.glob('*.mp4'))
    if not videos:
        return False
    for video in videos:
        runtime = video.with_name(video.stem + '_runtime.json')
        memory = video.with_name(video.stem + '_memory_measurement.json')
        if not runtime.exists() or not memory.exists() or not runtime_ok(runtime, int(job['group'])):
            continue
        probe = subprocess.run(
            ['ffprobe','-v','error','-select_streams','v:0','-count_frames',
             '-show_entries','stream=nb_read_frames,width,height,r_frame_rate',
             '-of','json',str(video)], capture_output=True, text=True)
        if probe.returncode != 0:
            continue
        try:
            stream=json.loads(probe.stdout)['streams'][0]
            frames=int(stream.get('nb_read_frames') or 0)
            width=int(stream.get('width') or 0); height=int(stream.get('height') or 0)
            fps_num, fps_den = (stream.get('r_frame_rate') or '0/1').split('/')
            fps = float(fps_num) / float(fps_den)
            if frames == 474 and abs(fps-16.0) < 0.01 and (width,height)==(832,480):
                return True
        except Exception:
            pass
    return False

def free_mib(gpu):
    out=subprocess.check_output(['nvidia-smi','-i',gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    return int(out.strip().splitlines()[0])

def wait_gpu(gpu, required, key, pending, active, completed, failed):
    while True:
        free=free_mib(gpu)
        save_state(pending, active, completed, failed, extra={'wait_gpu':gpu,'wait_key':key,'free_mib':free,'required_mib':required})
        if free >= required:
            return
        log(f'WAITING_FOR_GPU gpu={gpu} key={key} free_mib={free} required_mib={required}')
        time.sleep(20)

def run_one(gpu, job, pending, active, completed, failed):
    key=job['key']; out=Path(job['output']); out.mkdir(parents=True, exist_ok=True)
    if valid_output(job):
        log(f'SKIP_VALID key={key}')
        completed.add(key); return True
    threshold = RETRY_FREE_MIB if failed.get(key,0) else NORMAL_FREE_MIB
    wait_gpu(gpu, threshold, key, pending, active, completed, failed)
    cfg=make_config(job)
    logpath=RUN_ROOT / 'logs' / f'{job["label"]}_{job["case"]}.log'; logpath.parent.mkdir(parents=True, exist_ok=True)
    cmd=['env',f'CUDA_VISIBLE_DEVICES={gpu}',PYTHON,'-u',str(INFERENCE),'--config_path',str(cfg)]
    log(f'START gpu={gpu} key={key} threshold_mib={threshold} command={" ".join(cmd)}')
    with logpath.open('a') as fh:
        proc=subprocess.run(cmd,cwd=ROOT,stdout=fh,stderr=subprocess.STDOUT)
    ok=proc.returncode==0 and valid_output(job)
    log(f'END gpu={gpu} key={key} rc={proc.returncode} valid={ok}')
    if not ok:
        failed[key]=failed.get(key,0)+1
        tail=logpath.read_text(errors='replace').splitlines()[-20:]
        log(f'REQUEUE key={key} attempt={failed[key]} tail=' + ' | '.join(tail))
        pending.append(key)
        return False
    completed.add(key)
    return True

def run_evaluation(manifest, completed, pending, active, failed):
    wait_gpu(GPUS[0], NORMAL_FREE_MIB, 'EVALUATION', pending, active, completed, failed)
    evlog=RUN_ROOT/'evaluation.log'
    cmd=['env',f'CUDA_VISIBLE_DEVICES={GPUS[0]}',EVAL_PYTHON,'-u',str(EVALUATOR)]
    log(f'EVALUATION_START command={" ".join(cmd)}')
    with evlog.open('a') as fh:
        proc=subprocess.run(cmd,cwd=ROOT,stdout=fh,stderr=subprocess.STDOUT)
    if proc.returncode:
        log(f'EVALUATION_FAILED rc={proc.returncode}; runner will retry')
        return False
    log('EVALUATION_COMPLETE')
    return True

def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    manifest=json.loads(MANIFEST.read_text())
    jobs=ordered_jobs(manifest); by={j['key']:j for j in jobs}
    completed=set(); failed={}; pending=collections.deque()
    if STATE.exists():
        old=json.loads(STATE.read_text())
        completed=set(old.get('completed',[])); failed=dict(old.get('failed_attempts',{}))
        pending=collections.deque(old.get('pending',[]))
        for key in old.get('active',{}).values():
            if key not in completed: pending.appendleft(key)
        if old.get('stage')=='COMPLETE':
            log('ALREADY_COMPLETE'); return
    else:
        pending=collections.deque(j['key'] for j in jobs)
    pending=collections.deque(k for k in pending if k not in completed)
    active={}
    log(f'RUNNER_START jobs={len(jobs)} pending={len(pending)} gpus={GPUS} RR=true')
    while pending or active:
        started=False
        for gpu in GPUS:
            if gpu in active or not pending:
                continue
            key=pending[0]; job=by[key]
            if valid_output(job):
                pending.popleft(); completed.add(key)
                log(f'SKIP_VALID key={key}')
                started=True
                continue
            threshold=RETRY_FREE_MIB if failed.get(key,0) else NORMAL_FREE_MIB
            free=free_mib(gpu)
            if free < threshold:
                log(f'WAITING_FOR_GPU gpu={gpu} key={key} free_mib={free} required_mib={threshold}')
                continue
            pending.popleft()
            cfg=make_config(job)
            logpath=RUN_ROOT/'logs'/f'{job["label"]}_{job["case"]}.log'
            logpath.parent.mkdir(parents=True,exist_ok=True)
            cmd=['env',f'CUDA_VISIBLE_DEVICES={gpu}',PYTHON,'-u',str(INFERENCE),'--config_path',str(cfg)]
            fh=logpath.open('a')
            proc=subprocess.Popen(cmd,cwd=ROOT,stdout=fh,stderr=subprocess.STDOUT)
            active[gpu]={'key':key,'job':job,'proc':proc,'fh':fh,'logpath':logpath}
            log(f'START gpu={gpu} key={key} threshold_mib={threshold} command={" ".join(cmd)}')
            started=True
        for gpu,item in list(active.items()):
            rc=item['proc'].poll()
            if rc is None:
                continue
            item['fh'].close(); key=item['key']; job=item['job']
            ok=rc==0 and valid_output(job)
            log(f'END gpu={gpu} key={key} rc={rc} valid={ok}')
            del active[gpu]
            if ok:
                completed.add(key)
            else:
                failed[key]=failed.get(key,0)+1
                tail=item['logpath'].read_text(errors='replace').splitlines()[-20:]
                log(f'REQUEUE key={key} attempt={failed[key]} tail='+' | '.join(tail))
                pending.append(key)
        save_state(pending,active,completed,failed)
        if pending or active:
            time.sleep(20 if not started else 5)
    if len(completed) != len(jobs):
        raise RuntimeError(f'CANONICAL10_INFERENCE_GATE_FAILED: completed={len(completed)} expected={len(jobs)}')
    if not run_evaluation(manifest, completed, pending, active, failed):
        raise RuntimeError('evaluation failed')
    save_state(pending,active,completed,failed,stage='COMPLETE')
    log('RUNNER_COMPLETE')

if __name__=='__main__':
    main()
