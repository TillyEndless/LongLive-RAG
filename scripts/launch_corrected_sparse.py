#!/usr/bin/env python3
from pathlib import Path
import json, re, subprocess, os, time

ROOT=Path('/data/zxl/LongLive-RAG-group11_15_h200')
BASE=ROOT/'results/group12_15_persistent_campaign'
OUT=ROOT/'results/group14_15_corrected_sparse'
CFGROOT=OUT/'configs'; LOGROOT=OUT/'logs'
PY='/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'
GPUS=['0','1']

def make_jobs():
    manifest=json.loads((BASE/'campaign_manifest.json').read_text())
    src={(j['label'],j['case']):Path(j['config']) for j in manifest['jobs']}
    jobs=[]
    # Prioritize lower dense rate first for both groups: sparsity30 retains
    # 70%, then 80%, 90%, and 95%. Existing valid outputs are skipped.
    for removed in (30,20,10,5):
        for group in (14,15):
            keep=1.0-removed/100.0
            label=f'group{group}_sparsity{removed:02d}'
            for i in range(1,11):
                case=f'case{i:02d}'
                old=f'group{group}_sparse{removed:02d}'
                # Source configs use the same canonical prompt/case mapping.
                p=src[(old,case)]
                text=p.read_text()
                text=re.sub(r'^output_folder:.*$', f'output_folder: {OUT}/{label}/{case}', text, flags=re.M)
                text=re.sub(r'^  group_sparse_ratio:.*$', f'  group_sparse_ratio: {keep:.2f}', text, flags=re.M)
                text=re.sub(r'^ratio_semantics:.*$', 'ratio_semantics: retained_interaction_ratio', text, flags=re.M)
                text += f'\n# corrected sparse contract: sparsity={removed}% means remove {removed}%; retain={keep:.2f}\n'
                dst=CFGROOT/f'{label}_{case}.yaml'; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_text(text)
                jobs.append((label,case,dst,OUT/label/case))
    return jobs

def valid(out):
    return bool(list(out.glob('*.mp4'))) and any(p.name.endswith('_runtime.json') for p in out.glob('*.json'))

def worker(gpu, jobs):
    for label,case,cfg,out in jobs:
        out.mkdir(parents=True,exist_ok=True)
        if valid(out): continue
        log=LOGROOT/f'{label}_{case}.log'; log.parent.mkdir(parents=True,exist_ok=True)
        cmd=['env',f'CUDA_VISIBLE_DEVICES={gpu}',PY,'-u',str(ROOT/'inference.py'),'--config_path',str(cfg)]
        with log.open('a') as fh:
            fh.write(f'START gpu={gpu} label={label} case={case} command={cmd}\n'); fh.flush()
            rc=subprocess.run(cmd,cwd=ROOT,stdout=fh,stderr=subprocess.STDOUT).returncode
            fh.write(f'END rc={rc}\n')
        if rc:
            raise SystemExit(f'FAILED {label}/{case} rc={rc}')

if __name__=='__main__':
    jobs=make_jobs()
    (OUT/'corrected_sparse_manifest.json').write_text(json.dumps({
        'semantics':'sparsity is removal fraction; retained fraction = 1-sparsity',
        'seed':0,'window':12,'jobs':[{'label':a,'case':b,'config':str(c),'output':str(d)} for a,b,c,d in jobs]},indent=2)+'\n')
    # Deterministic RR partition; at most one inference per GPU.
    buckets=[[],[]]
    for idx,j in enumerate(jobs): buckets[idx%2].append(j)
    import multiprocessing as mp
    ps=[mp.Process(target=worker,args=(GPUS[i],buckets[i])) for i in range(2)]
    for p in ps: p.start()
    for p in ps: p.join()
    if any(p.exitcode for p in ps): raise SystemExit(1)
    ev=ROOT/'scripts/evaluate_corrected_sparse.py'
    evlog=OUT/'evaluation.log'
    with evlog.open('w') as fh:
        rc=subprocess.run(['env','CUDA_VISIBLE_DEVICES=0','/data/zxl/SolarWM/.env_vbench310/bin/python','-u',str(ev)],cwd=ROOT,stdout=fh,stderr=subprocess.STDOUT).returncode
    if rc: raise SystemExit(f'EVALUATION_FAILED rc={rc}')
    print('COMPLETE_WITH_EVALUATION', OUT)
