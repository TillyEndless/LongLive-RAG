import json, subprocess, time
from pathlib import Path
ROOT=Path('/data/zxl/LongLive-RAG-group11_qprev_h200')
PY='/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'
OUT=ROOT/'results/latency_instrumented/group11_2/case_01'; OUT.mkdir(parents=True,exist_ok=True)
PROMPT=ROOT/'results/latency_instrumented/group11_2/case_01.txt'; PROMPT.write_text(Path('/data/zxl/LongLive-RAG-profile/prompts10.txt').read_text().splitlines()[0]+'\n')
cfg0=ROOT/'configs/qprev_case01.yaml'; cfg=ROOT/'results/latency_instrumented/group11_2/case_01.yaml'
from omegaconf import OmegaConf
c=OmegaConf.load(cfg0); c.data_path=str(PROMPT); c.output_folder=str(OUT); c.inference_iter=0; c.skip_existing=False; OmegaConf.save(c,cfg)
log=OUT/'inference.log'; start=time.perf_counter()
with log.open('w') as f:
    rc=subprocess.run(['env','CUDA_VISIBLE_DEVICES=1',PY,'-u',str(ROOT/'inference.py'),'--config_path',str(cfg)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
wall=time.perf_counter()-start
videos=sorted(OUT.glob('*.mp4')); runtime=next(iter(OUT.glob('*_runtime.json')),None)
if rc or not videos or runtime is None: raise SystemExit(f'FAILED rc={rc} video={videos} runtime={runtime}')
d=json.loads(runtime.read_text()); prof=d.get('group11_profile',{}); h2d=sum(float(x.get('copy_time_ms',0)) for x in prof.get('h2d_rows',[]))/1000.0
ph=prof.get('model_phase_ms',{}); transformer=sum(float(v) for v in ph.values())/1000.0
metrics={'group':'11.2','case':'case_01','e2e_latency_s':wall,'exposed_h2d_s':h2d,'wrapper_latency_s':max(wall-transformer,0.0),'transformer_latency_s':transformer,'h2d_measurement':'sum of recorded host-side copy_time_ms','source_runtime':str(runtime)}
(OUT/'latency_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n'); print(json.dumps(metrics,indent=2))
