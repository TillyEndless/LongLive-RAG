#!/usr/bin/env python3
"""Post-run Group4 queue. It never starts while canonical3 has active work."""
import json, os, subprocess, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'results/canonical3_20260929'; PY='/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'; MAIN_PID=1085202
PENDING=OUT/'group4_pending_append.json'; STATE=OUT/'group4_postrun_state.json'
def save(d):
 t=STATE.with_suffix('.tmp'); t.write_text(json.dumps(d,indent=2)); t.replace(STATE)
def run(j,gpu):
 log=OUT/'logs'/j['label']/(j['case']+'.postrun.log'); log.parent.mkdir(parents=True,exist_ok=True); e=os.environ.copy(); e['CUDA_VISIBLE_DEVICES']=str(gpu)
 with log.open('a') as f:
  rc=subprocess.run([PY,'-u','inference.py','--config_path',j['config']],cwd=j['workdir'],env=e,stdout=f,stderr=subprocess.STDOUT).returncode
  if rc: return rc
  return subprocess.run([PY,str(ROOT/'scripts/evaluate_canonical3_case.py'),'--label',j['label'],'--case',j['case'],'--output-root',str(OUT)],cwd=ROOT,env=e,stdout=f,stderr=subprocess.STDOUT).returncode
def main():
 d={'status':'WAITING_FOR_MAIN','pending':json.loads(PENDING.read_text())['jobs'],'completed':{},'failed':[]}; save(d)
 while True:
  try: s=json.loads((OUT/'runner_state.json').read_text())
  except Exception: time.sleep(20); continue
  if s.get('status')=='COMPLETE' and not s.get('active'): break
  time.sleep(20)
 d['status']='RUNNING'; save(d); jobs=d['pending']
 with ThreadPoolExecutor(max_workers=2) as ex:
  fs={ex.submit(run,j,g): (j,g) for g,j in zip((0,1)*3,jobs)}
  for f in as_completed(fs):
   j,g=fs[f]; rc=f.result(); key=j['label']+'/'+j['case']
   (d['completed'] if rc==0 else d['failed'])[key]={'gpu':g,'return_code':rc}; save(d)
 d['status']='COMPLETE'; save(d)
if __name__=='__main__': main()
