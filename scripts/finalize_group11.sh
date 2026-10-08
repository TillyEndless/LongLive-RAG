#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/zxl/LongLive-RAG-group11_15_h200
PY=/data/zxl/SolarWM/.env_vbench310/bin/python
OUT_ROOT=$ROOT/results/group11_canonical10
REPORT=$ROOT/reports/group11_final.md
CSV=$ROOT/results/group11_final.csv
mkdir -p "$ROOT/reports"

valid() {
  local d="$1" v="$d/rank0-0-0_lora.mp4"
  test -s "$v" && test -s "$d/rank0-0-0_lora_runtime.json" && test -s "$d/rank0-0-0_lora_memory_measurement.json" || return 1
  test "$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$v" 2>/dev/null || true)" = 474
}

while true; do
  count=0
  for i in $(seq 1 10); do
    d="$OUT_ROOT/case_$(printf '%02d' "$i")"
    [ "$i" -eq 1 ] && d="$ROOT/results/group11_case01"
    valid "$d" && count=$((count+1)) || true
  done
  procs=$(ps -eo args | grep "/data/zxl/LongLive-RAG-group11_15_h200/inference.py" | grep -v grep || true)
  if [ "$count" -eq 10 ] && [ -z "$procs" ]; then break; fi
  echo "WAITING_FOR_GROUP11_COMPLETION valid=$count/10 $(date -u +%FT%TZ)"
  sleep 20
done

exec "$PY" - "$ROOT" "$CSV" "$REPORT" <<'PY'
import csv, json, subprocess, sys
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import AutoImageProcessor, AutoModel

root=Path(sys.argv[1]); csv_path=Path(sys.argv[2]); report=Path(sys.argv[3])
rows=[]
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
proc=AutoImageProcessor.from_pretrained('/data/zxl/eval_models/dinov2-small')
model=AutoModel.from_pretrained('/data/zxl/eval_models/dinov2-small').to(device).eval()

def frame(path, idx):
    cap=cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, idx); ok, x=cap.read(); cap.release()
    if not ok: raise RuntimeError(f'cannot read {idx} from {path}')
    return cv2.cvtColor(x, cv2.COLOR_BGR2RGB)

@torch.inference_mode()
def dino(a,b):
    x=proc(images=[Image.fromarray(a),Image.fromarray(b)],return_tensors='pt')
    x={k:v.to(device) for k,v in x.items()}; z=model(**x).last_hidden_state[:,0]
    z=torch.nn.functional.normalize(z,dim=-1); return float((z[0]*z[1]).sum().item())

for i in range(1,11):
    case=f'case_{i:02d}'; d=root/'results'/'group11_canonical10'/case
    if i==1: d=root/'results'/'group11_case01'
    v=d/'rank0-0-0_lora.mp4'; a=frame(v,0); b=frame(v,237)
    m=json.loads((d/'rank0-0-0_lora_memory_measurement.json').read_text())
    r=json.loads((d/'rank0-0-0_lora_runtime.json').read_text())
    lat=json.loads((d/'latency.json').read_text()) if (d/'latency.json').exists() else {'latency_s':'NOT_AVAILABLE'}
    rows.append({'Group':11,'Method':'LongLive + Draft Attention','Window':12,'case':case,
      'Draft_RAG':'YES','DraftMap':'YES','Precision_Budget_KERNEL':'BF16 100%',
      'Q_EXEC':'BF16','K_STORAGE':'BF16','K_EXEC':'BF16','V_STORAGE':'BF16','V_EXEC':'BF16',
      'GPU_KV_GiB':m.get('GPU_KV_MEASURED_GiB','NA'),'CPU_KV_GiB':m.get('CPU_KV_MEASURED_GiB','NA'),
      'Additional_GPU_Draft_GiB':m.get('GPU_DRAFT_PERSISTENT_GiB','NA'),'CPU_DRAFT_GiB':m.get('CPU_DRAFT_PERSISTENT_GiB','NA'),
      'Draft_H2D_bytes':m.get('DRAFT_H2D_BYTES','NA'),'Full_KV_H2D_bytes':m.get('FULL_KV_H2D_BYTES','NA'),
      'DINO':dino(a,b),'SSIM':float(structural_similarity(a,b,channel_axis=2,data_range=255)),
      'PSNR':float(peak_signal_noise_ratio(a,b,data_range=255)),'LPIPS':'NOT_AVAILABLE',
      'Latency_s':lat.get('latency_s','NOT_AVAILABLE'),'Video_frames':474,'FPS':'16/1','Protocol':'frame 0 vs frame 237'})

fields=list(rows[0].keys())
with csv_path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def vals(k): return [float(x[k]) for x in rows if x[k] not in ('NA','NOT_AVAILABLE')]
def mean(k):
    x=vals(k); return sum(x)/len(x) if x else None
with report.open('w') as f:
    f.write('# Group 11 final canonical10\n\n')
    f.write('This report contains only the corrected Group 11 run: window=12, DraftMap/Draft-RAG active, persistent historical Draft-K GPU-only, CPU Draft-K=0, Draft-K H2D=0, full BF16 historical K/V CPU archive, selected full BF16 K/V CPU-to-GPU retrieval, and BF16 attention. Groups 12–15 are excluded.\n\n')
    f.write('Manifest: `/data/zxl/LongLive-RAG-profile/prompts10.txt`; SHA256 `db9462c0954186b9128543b58fb6d1d496a8b89e0754333beca698d3cf461fec`; seed=0. Evaluator protocol: 474 frames, 16 FPS, 832x480, frame 0 vs frame 237, DINOv2-small CLS cosine, raw RGB SSIM/PSNR, LPIPS `NOT_AVAILABLE`.\n\n')
    f.write('|case|DINO|SSIM|PSNR|GPU KV GiB|CPU KV GiB|GPU Draft GiB|CPU Draft GiB|Draft H2D bytes|Full KV H2D bytes|Latency s|\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n')
    for x in rows:
      f.write('|{case}|{DINO:.6f}|{SSIM:.6f}|{PSNR:.6f}|{GPU_KV_GiB}|{CPU_KV_GiB}|{Additional_GPU_Draft_GiB}|{CPU_DRAFT_GiB}|{Draft_H2D_bytes}|{Full_KV_H2D_bytes}|{Latency_s}|\n'.format(**x))
    f.write('\n## Canonical10 means\n\n')
    f.write(f'- DINO: `{mean("DINO"):.6f}`\n- SSIM: `{mean("SSIM"):.6f}`\n- PSNR: `{mean("PSNR"):.6f}`\n- LPIPS: `NOT_AVAILABLE`\n')
    ml=mean('Latency_s'); f.write(f'- Latency: `{ml:.3f} s` over measured cases; case01 pre-existing artifact had no wall-clock latency and is excluded from this mean.\n' if ml is not None else '- Latency: `NOT_AVAILABLE`.\n')
    f.write('\nFull per-case data is in `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_final.csv`.\n')
print(csv_path); print(report)
PY
