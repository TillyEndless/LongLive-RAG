#!/usr/bin/env python3
"""Detached 33-GiB RR campaign for the H200 Group11-15 and RAG jobs.

This file is copied to the H200 repository and is intentionally self-contained:
it only creates a new output tree and never changes the source checkout.
"""
import argparse, collections, csv, hashlib, json, os, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

HROOT=Path('/data/zxl/LongLive-RAG-group11_15_h200')
RROOT=Path('/data/zxl/LongLive-RAG')
PY='/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'
EVAL_PY='/data/zxl/SolarWM/.env_vbench310/bin/python'
# New-run-only output root.  It is intentionally distinct from every prior
# campaign so no old-rule video/evaluation artifact can be reused.
OUT=HROOT/'results/night_campaign_serial_20260929_v2'
MANIFEST=Path('/data/zxl/LongLive-RAG-profile/prompts10.txt')
THRESH=33*1024
GPU_IDS=['0','1']; MAX_PER_GPU=1

def now(): return datetime.now(timezone.utc).isoformat()
def q(s): return str(s).replace('\\','/').replace('"','\\"')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def replace_line(t,key,val):
    pat=r'(?m)^'+re.escape(key)+r':.*$'
    line=key+': '+str(val)
    return re.sub(pat,line,t,count=1) if re.search(pat,t) else t+'\n'+line+'\n'

def replace_nested(t,key,val):
    pat=r'(?m)^(  '+re.escape(key)+r':).*?$'
    return re.sub(pat,lambda m:m.group(1)+' '+str(val),t,count=1) if re.search(pat,t) else t

def make_cfg(src,dst,data,out,extra=None):
    t=Path(src).read_text()
    t=replace_line(t,'data_path',data); t=replace_line(t,'output_folder',out)
    t=replace_line(t,'skip_existing','false')
    if extra:
        for k,v in extra.items():
            if k.startswith('model_kwargs.'):
                t=replace_nested(t,k.split('.',1)[1],v)
            else: t=replace_line(t,k,v)
    Path(dst).parent.mkdir(parents=True,exist_ok=True); Path(dst).write_text(t); return str(dst)

def prepare():
    OUT.mkdir(parents=True,exist_ok=True); (OUT/'configs').mkdir(exist_ok=True)
    cases=[HROOT/f'configs/group12_15_corrected_campaign/case{i:02d}.txt' for i in range(1,11)]
    jobs=[]
    specs=[]
    for g,src,extra in [
      ('group11_1',HROOT/'configs/g11_1.yaml',{}),('group11_2',HROOT/'configs/g11_2.yaml',{}),
      ('group11_3',HROOT/'configs/g11_3.yaml',{}),('group11_4',HROOT/'configs/g11_4.yaml',{}),
      ('group12',HROOT/'configs/g12_currentq_int8_fp8.yaml',{}),
      ('group13',HROOT/'configs/g13_currentq_nvfp4.yaml',{}),
    ]:
        specs.append((g,str(src),extra,str(HROOT)))
    for g,r in [('group14_1',.7),('group14_2',.5),('group14_3',.3),('group14_4',.6)]:
        specs.append((g,str(HROOT/'configs/g14_currentq_int8_fp8_promotion.yaml'),{'model_kwargs.q_sparse_ratio':r,'model_kwargs.kv_promotion_ratio':.2,'model_kwargs.group_sparse_ratio':0.0},str(HROOT)))
    for g,r in [('group15_1',.7),('group15_2',.5),('group15_3',.3),('group15_4',.6)]:
        specs.append((g,str(HROOT/'configs/g15_currentq_nvfp4_promotion.yaml'),{'model_kwargs.q_sparse_ratio':r,'model_kwargs.kv_promotion_ratio':.2,'model_kwargs.group_sparse_ratio':0.0},str(HROOT)))
    for g,src,extra,repo in specs:
        root=OUT/g; cfgs=[]
        for i,case in enumerate(cases,1):
            cfgs.append(make_cfg(src,OUT/'configs'/g/f'case_{i:02d}.yaml',case,root/f'case_{i:02d}',extra))
        jobs.append({'key':g,'label':g,'repo':repo,'root':str(root),'configs':cfgs,'kind':'h200'})
    # RAG configs live in their own checkout and are deliberately written to a new tree.
    rag_specs=[('group4_1',RROOT/'profiling_configs/rag_b_retrieval_w6.yaml'),('group4_2',RROOT/'results/canonical_longlive_rag_w12_10case/config_w12.yaml')]
    for g,src in rag_specs:
        root=RROOT/'results/night_campaign_20260929'/g; cfgs=[]
        for i,case in enumerate(cases,1):
            cfgs.append(make_cfg(src,OUT/'configs'/g/f'case_{i:02d}.yaml',case,root/f'case_{i:02d}',{}))
        jobs.append({'key':g,'label':g,'repo':str(RROOT),'root':str(root),'configs':cfgs,'kind':'rag'})
    (OUT/'jobs.json').write_text(json.dumps(jobs,indent=2))
    meta={'created_utc':now(),'threshold_mib':THRESH,'max_per_gpu':MAX_PER_GPU,'manifest':str(MANIFEST),'manifest_sha256':sha(MANIFEST),'jobs':[j['key'] for j in jobs], 'note':'Group9/10 intentionally excluded; request prose specified Group11-15 and 4.1/4.2.'}
    (OUT/'campaign_manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({'jobs':len(jobs),'cases_per_job':10,'manifest_sha256':meta['manifest_sha256'],'output':str(OUT)},indent=2))

def video_for(case_root):
    vs=sorted(Path(case_root).glob('**/*.mp4'))
    return vs[0] if vs else None

def evaluate(job):
    # Canonical lightweight evaluator: frozen frame pair, raw RGB SSIM/PSNR, optional DINO.
    code=r'''import csv,glob,json,sys
from pathlib import Path
import cv2,numpy as np
from skimage.metrics import structural_similarity,peak_signal_noise_ratio
root=Path(sys.argv[1]); out=root/'evaluation'; out.mkdir(exist_ok=True)
try:
 import torch
 from PIL import Image
 from transformers import AutoImageProcessor,AutoModel
 dev='cuda' if torch.cuda.is_available() else 'cpu'; pr=AutoImageProcessor.from_pretrained('/data/zxl/eval_models/dinov2-small'); mo=AutoModel.from_pretrained('/data/zxl/eval_models/dinov2-small').to(dev).eval()
 def ds(a,b):
  with torch.inference_mode():
   x=pr(images=[Image.fromarray(a),Image.fromarray(b)],return_tensors='pt'); x={k:v.to(dev) for k,v in x.items()}; z=torch.nn.functional.normalize(mo(**x).last_hidden_state[:,0],dim=-1); return float((z[0]*z[1]).sum())
except Exception: ds=lambda a,b:'NOT_AVAILABLE'
rows=[]
for c in sorted(root.glob('case_*')):
 v=sorted(c.glob('**/*.mp4'))
 if not v: continue
 p=v[0]; cap=cv2.VideoCapture(str(p)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,a=cap.read(); cap.set(cv2.CAP_PROP_POS_FRAMES,237); ok2,b=cap.read(); fps=cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
 if not(ok and ok2): continue
 a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB); b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
 rows.append({'case':c.name,'video':str(p),'frames':n,'fps':fps,'width':w,'height':h,'first_frame':0,'return_frame':237,'dino':ds(a,b),'ssim':float(structural_similarity(a,b,channel_axis=2,data_range=255)),'psnr':float(peak_signal_noise_ratio(a,b,data_range=255)),'lpips':'NOT_AVAILABLE'})
with (out/'quality.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['case']); w.writeheader(); w.writerows(rows)
def mean(k):
 z=[]
 for r in rows:
  try:z.append(float(r[k]))
  except:pass
 return sum(z)/len(z) if z else 'NA'
(out/'summary.csv').write_text(json.dumps({'cases':len(rows),'dino':mean('dino'),'ssim':mean('ssim'),'psnr':mean('psnr'),'lpips':'NOT_AVAILABLE','protocol':{'frames':474,'fps':16,'pair':[0,237]}},indent=2))
(out/'evaluation_manifest.json').write_text(json.dumps({'cases':len(rows),'valid_10':len(rows)==10,'protocol':{'frames':474,'fps':16,'resolution':'832x480','pair':'0->237'}},indent=2))
if len(rows)!=10: raise SystemExit('evaluation did not find 10 readable videos')
'''
    p=Path(job['root'])/'evaluation_runner.py'; p.write_text(code)
    elog=Path(job['root'])/'evaluation.log'
    with elog.open('a') as ef:
        r=subprocess.run([EVAL_PY,str(p),job['root']],cwd=job['repo'],env={**os.environ,'CUDA_VISIBLE_DEVICES':job['gpu'],'TMPDIR':'/data/zxl/tmp','HF_HOME':'/data/zxl/tmp/hf'},stdout=ef,stderr=subprocess.STDOUT)
    return r.returncode

def worker(jobfile):
    job=json.loads(Path(jobfile).read_text()); root=Path(job['root']); log=root/'worker.log'; root.mkdir(parents=True,exist_ok=True)
    with log.open('a') as lf:
        for i,cfg in enumerate(job['configs'],1):
            # Preserve completed case outputs after a controlled runner stop/restart.
            case_root=Path(job['root'])/f'case_{i:02d}'
            if list(case_root.glob('**/*.mp4')):
                lf.write(f'[{now()}] SKIP_EXISTING_CASE {i}/10 {case_root}\n'); lf.flush(); continue
            lf.write(f'[{now()}] CASE {i}/10 {cfg}\\n'); lf.flush()
            e=os.environ.copy(); e['CUDA_VISIBLE_DEVICES']=job['gpu']; e['PYTHONUNBUFFERED']='1'
            p=subprocess.run([PY,'inference.py','--config_path',cfg],cwd=job['repo'],env=e,stdout=lf,stderr=subprocess.STDOUT)
            if p.returncode: return p.returncode
        return evaluate(job)

def free_mib(g):
    try:return int(subprocess.check_output(['nvidia-smi','-i',g,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip().splitlines()[0])
    except:return 0

def run():
    jobs=json.loads((OUT/'jobs.json').read_text()); pending=collections.deque(jobs); active={}; complete=[]; failed=[]
    state={'status':'RUNNING','started_utc':now(),'threshold_mib':THRESH,'max_per_gpu':MAX_PER_GPU,'pending':[j['key'] for j in pending],'active':{},'completed':[],'retry_count':{}}
    sf=OUT/'runner_state.json'; lf=(OUT/'runner.log').open('a')
    def save():
        state['pending']=[j['key'] for j in pending]; state['active']={k:{'gpu':v['gpu'],'pid':v['p'].pid} for k,v in active.items()}; state['completed']=complete; sf.write_text(json.dumps(state,indent=2))
    while pending or active:
        # reap
        for k,v in list(active.items()):
            rc=v['p'].poll()
            if rc is None: continue
            del active[k]; lf.write(f'[{now()}] DONE {k} gpu={v["gpu"]} rc={rc}\\n'); lf.flush()
            if rc==0: complete.append(k)
            else:
                state['retry_count'][k]=state['retry_count'].get(k,0)+1; pending.append(next(j for j in jobs if j['key']==k)); lf.write(f'[{now()}] RETRY {k} attempt={state["retry_count"][k]}\\n')
        # fill slots, RR order; each launch checks current free VRAM.
        for g in GPU_IDS:
            if sum(v['gpu']==g for v in active.values())>=MAX_PER_GPU or free_mib(g)<THRESH: continue
            for _ in range(len(pending)):
                j=pending.popleft()
                if free_mib(g)<THRESH: pending.appendleft(j); break
                jf=OUT/'jobs_runtime'/f'{j["key"]}.json'; jf.parent.mkdir(exist_ok=True); j={**j,'gpu':g}; jf.write_text(json.dumps(j))
                p=subprocess.Popen([PY,str(Path(__file__).resolve()),'--worker','--job-file',str(jf)],cwd=j['repo'],env={**os.environ,'CUDA_VISIBLE_DEVICES':g},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                active[j['key']]={'p':p,'gpu':g}; lf.write(f'[{now()}] START {j["key"]} gpu={g} pid={p.pid} free_mib={free_mib(g)}\\n'); break
        save(); time.sleep(20)
    state['status']='COMPLETE'; state['finished_utc']=now(); save(); lf.write(f'[{now()}] COMPLETE\\n'); lf.close()

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--prepare-only',action='store_true'); ap.add_argument('--worker',action='store_true'); ap.add_argument('--job-file'); a=ap.parse_args()
    if a.prepare_only: prepare()
    elif a.worker: sys.exit(worker(a.job_file))
    else:
        if not (OUT/'jobs.json').exists(): prepare()
        run()
