#!/usr/bin/env python3
"""FIFO, resumable canonical3 runner; independent of canonical10."""
import argparse, json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import yaml
from scripts.canonical3_config_inheritance import canonical3_template, deep_merge

ROOT=Path(__file__).resolve().parents[1]
PYTHON=os.environ.get('CANONICAL3_PYTHON','/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python')
OLD=Path('/data/zxl/LongLive-RAG-group11_15_h200')
OUT=ROOT/'results/canonical3_20260929'
PROMPTS=Path('/data/zxl/LongLive-RAG-profile/prompts10.txt')
SHA='db9462c0954186b9128543b58fb6d1d496a8b89e0754333beca698d3cf461fec'
CASES=['case_01','case_02','case_03']

def base(label):
    if label=='11.1': return ROOT/'configs/group11_15_reference/group11_draftmap_online_w12_case01.yaml',ROOT
    if label=='11.2': return ROOT/'configs/qprev_case01.yaml',ROOT
    if label=='11.3': return ROOT/'configs/flashfetch_case01.yaml',ROOT
    if label=='11.4': return ROOT/'configs/group11_4_next_layer_prefetch_case01.yaml',ROOT
    if label=='12': return ROOT/'configs/group12_15_v2/group12_v2_local_int8_fp8.yaml',ROOT
    if label=='13': return ROOT/'configs/group12_15_v2/group13_v2_local_nvfp4.yaml',ROOT
    if label.startswith('14.'): return ROOT/'configs/group12_15_v2/group14_v2_local_int8_fp8_sparse.yaml',ROOT
    if label.startswith('15.'): return ROOT/'configs/group12_15_v2/group15_v2_local_nvfp4_sparse.yaml',ROOT
    if label in ('4.1','4.2'): return OLD/f'results/night_campaign_serial_20260929_v2/configs/group{label}_1/case_01.yaml',OLD
    return None,None

def materialize(label,case):
    src,work=base(label)
    if not src or not src.exists(): raise FileNotFoundError(src)
    inherited = canonical3_template(ROOT, label)
    cfg = deep_merge(inherited, yaml.safe_load(src.read_text())) if inherited is not None else yaml.safe_load(src.read_text())
    cfg['data_path']=str(PROMPTS); cfg['inference_iter']=int(case[-2:])-1
    cfg['output_folder']=str(OUT/'outputs'/label/case); cfg['skip_existing']=True
    cfg['seed']=0; cfg['canonical_case_count']=3; cfg['canonical_case_selection']='first_n'; cfg['num_output_frames']=120
    if label.startswith('14.') or label.startswith('15.'):
        dense={'14.1':.7,'14.2':.6,'14.3':.5,'14.4':.3,'15.1':.7,'15.2':.6,'15.3':.5,'15.4':.3}[label]
        mk=cfg.setdefault('model_kwargs',{}); mk['q_sparse_ratio']=1.0-dense; mk['group_sparse_ratio']=1.0-dense; mk['v2_fetch_scheduler']='serial_two_wait'
    if label in ('12','13'): cfg.setdefault('model_kwargs',{})['v2_fetch_scheduler']='serial_two_wait'
    dst=OUT/'configs'/label/f'{case}.yaml'; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_text(yaml.safe_dump(cfg,sort_keys=False)); return dst,work

def state_io():
    p=OUT/'runner_state.json'
    if p.exists(): return p,json.loads(p.read_text())
    jobs=[]
    for label in ['4.1','4.2','11.1','11.2','11.3','11.4','12','13']+[f'14.{i}' for i in range(1,5)]+[f'15.{i}' for i in range(1,5)]:
        for case in CASES:
            try: cfg,work=materialize(label,case); jobs.append({'label':label,'case':case,'config':str(cfg),'workdir':str(work)})
            except FileNotFoundError: pass
    return p,{'status':'PENDING','queue':jobs,'completed':{},'failed':[],'active':{},'manifest_sha256':SHA,'case_ids':CASES}

def save(p,s):
    t=p.with_suffix('.tmp'); t.write_text(json.dumps(s,indent=2,sort_keys=True)); t.replace(p)

def run(job,gpu,attempt):
    log=OUT/'logs'/job['label']/(job['case']+f'.attempt{attempt}.log'); log.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu); env['GROUP14_15_MAIN_SCHEDULER']='SERIAL_EXACT'
    with log.open('a') as f:
        f.write(json.dumps({'gpu':gpu,'config':job['config'],'workdir':job['workdir']})+'\n')
        p=subprocess.run([PYTHON,'-u','inference.py','--config_path',job['config']],cwd=job['workdir'],env=env,stdout=f,stderr=subprocess.STDOUT)
        if p.returncode: return p.returncode
        p=subprocess.run([PYTHON,str(ROOT/'scripts/evaluate_canonical3_case.py'),'--label',job['label'],'--case',job['case'],'--output-root',str(OUT)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT); return p.returncode

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--gpus',type=int,nargs='+',default=[0,1]); ap.add_argument('--max-retries',type=int,default=2); a=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True); p,s=state_io(); s['status']='RUNNING'; save(p,s)
    with ThreadPoolExecutor(max_workers=len(a.gpus)) as pool:
      while s['queue'] or s.get('active'):
        futs={}
        for gpu in a.gpus:
          if not s['queue']: break
          j=s['queue'].pop(0); key=j['label']+'/'+j['case']; s.setdefault('active',{})[str(gpu)]={'job':key,'gpu':gpu}; futs[pool.submit(run,j,gpu,0)]=(j,gpu,key)
        save(p,s)
        for fut in as_completed(futs):
          j,gpu,key=futs[fut]
          try: rc=fut.result()
          except Exception: rc=99
          s.get('active',{}).pop(str(gpu),None)
          if rc==0: s['completed'][key]={'gpu':gpu,'time':time.time()}
          else:
            s['failed'].append({'job':key,'return_code':rc,'time':time.time()})
            if a.max_retries: s['queue'].append(j)
          save(p,s)
    s['status']='COMPLETE'; save(p,s)
if __name__=='__main__': main()
