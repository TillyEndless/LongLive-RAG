import json
import os
import subprocess
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
PY = '/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'

def run(name, source):
    out = ROOT / 'results' / 'group11_validity_smoke_10block' / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(str(ROOT / source))
    cfg.denoising_step_list = [1000]
    cfg.num_output_frames = 30
    cfg.smoke_10block = True
    cfg.inference_iter = 1
    cfg.skip_existing = False
    cfg.data_path = '/data/zxl/strict_latency_case01_20260928/case_01.txt'
    cfg.output_folder = str(out / 'inference')
    cfg.model_kwargs.memory_size = 2
    cfg.model_kwargs.recent_exclude = 0
    cfg_path = out / 'config.yaml'
    OmegaConf.save(cfg, str(cfg_path))
    env = os.environ.copy()
    env.update({'CUDA_VISIBLE_DEVICES': '1', 'RAG_STRATEGY_PROFILE': '1',
                'RAG_PROFILE_CASE_ID': 'case_01', 'PYTHONPATH': str(ROOT),
                'QPREV_ALIGNMENT_ASSERT': '1' if name == 'group11_2_qprev' else '0'})
    log = out / 'runtime.log'
    with log.open('w') as f:
        p = subprocess.run([PY, '-u', 'inference.py', '--config_path', str(cfg_path)],
                           cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT)
    (out / 'exit_status.json').write_text(json.dumps({'returncode': p.returncode}, indent=2))
    if p.returncode:
        raise SystemExit(f'{name} smoke failed; see {log}')

run('group11_2_qprev', 'configs/g11_2.yaml')
run('group11_4_prefetch', 'configs/g11_4.yaml')
print('validity smoke PASS')
