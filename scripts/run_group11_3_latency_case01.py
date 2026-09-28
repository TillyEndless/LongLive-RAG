import json, os, subprocess, time
from pathlib import Path
ROOT=Path('/data/zxl/LongLive-RAG-group11_flashfetch_h200')
PY='/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'
OUT=ROOT/'results/latency_instrumented/group11_3/case_01'; OUT.mkdir(parents=True,exist_ok=True)
PROMPT=ROOT/'results/latency_instrumented/group11_3/case_01.txt'; PROMPT.write_text(Path('/data/zxl/LongLive-RAG-profile/prompts10.txt').read_text().splitlines()[0]+'\n')
from omegaconf import OmegaConf
c=OmegaConf.load(ROOT/'configs/flashfetch_case01.yaml'); c.data_path=str(PROMPT); c.output_folder=str(OUT); c.inference_iter=0; c.skip_existing=False
cfg=ROOT/'results/latency_instrumented/group11_3/case_01.yaml'; OmegaConf.save(c,cfg)
start=time.perf_counter(); log=OUT/'inference.log'
with log.open('w') as f: rc=subprocess.run(['env',f'CUDA_VISIBLE_DEVICES={os.environ.get("PROBE_GPU", "1")}',PY,'-u',str(ROOT/'inference.py'),'--config_path',str(cfg)],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
wall=time.perf_counter()-start; runtime=next(iter(OUT.glob('*_runtime.json')),None); videos=sorted(OUT.glob('*.mp4'))
if rc or not runtime or not videos: raise SystemExit(f'FAILED rc={rc} runtime={runtime} videos={videos}')
d=json.loads(runtime.read_text()); prof=d.get('group11_profile',{}); h2d=sum(float(x.get('copy_time_ms',0)) for x in prof.get('h2d_rows',[]))/1000.0; transformer=sum(float(v) for v in prof.get('model_phase_ms',{}).values())/1000.0
m={'group':'11.3','case':'case_01','e2e_latency_s':wall,'exposed_h2d_s':h2d,'wrapper_latency_s':max(wall-transformer,0.0),'transformer_latency_s':transformer,'h2d_measurement':'sum of recorded host-side copy_time_ms','source_runtime':str(runtime)}
(OUT/'latency_metrics.json').write_text(json.dumps(m,indent=2)+'\n'); print(json.dumps(m,indent=2))
