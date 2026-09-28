#!/usr/bin/env python3
import csv, json
from pathlib import Path
import cv2
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

ROOT=Path('/data/zxl/LongLive-RAG-group11_15_h200')
IN=ROOT/'results/group11_canonical10'
OUT=IN/'evaluation_rerun_20260928'
MODEL='/data/zxl/eval_models/dinov2-small'
OUT.mkdir(parents=True,exist_ok=True)

def read_frame(path,idx):
    cap=cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,img=cap.read(); cap.release()
    if not ok: raise RuntimeError(f'cannot read frame {idx}: {path}')
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

processor=AutoImageProcessor.from_pretrained(MODEL)
model=AutoModel.from_pretrained(MODEL).to('cuda').eval()
@torch.inference_mode()
def dino(a,b):
    x=processor(images=[Image.fromarray(a),Image.fromarray(b)],return_tensors='pt'); x={k:v.to('cuda') for k,v in x.items()}
    f=torch.nn.functional.normalize(model(**x).last_hidden_state[:,0],dim=-1); return float((f[0]*f[1]).sum().item())

rows=[]
TIMING_KEYS = ['E2E_LATENCY_S', 'EXPOSED_H2D_S', 'WRAPPER_LATENCY_S']
for i in range(1,11):
    case=f'case_{i:02d}'; videos=sorted((IN/case).glob('*.mp4'))
    if not videos: raise RuntimeError(f'missing video: {case}')
    video=videos[0]; cap=cv2.VideoCapture(str(video)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    if n!=474 or abs(fps-16)>0.01 or (w,h)!=(832,480): raise RuntimeError(f'protocol mismatch {case}: {(n,fps,w,h)}')
    a,b=read_frame(video,0),read_frame(video,237)
    rows.append({'group':'11.1','case':case,'video':str(video),'first_frame':0,'return_frame':237,'frames':n,'fps':fps,'DINO':dino(a,b),'SSIM':float(structural_similarity(a,b,channel_axis=2,data_range=255)),'PSNR':float(peak_signal_noise_ratio(a,b,data_range=255)),'LPIPS':'NOT_AVAILABLE','protocol':'frame 0 vs frame 237; DINOv2-small CLS cosine; raw RGB SSIM/PSNR'})
for row in rows:
    meta_path = Path(row['video']).with_name(Path(row['video']).stem + '_runtime.json')
    try: meta = json.loads(meta_path.read_text())
    except Exception: meta = {}
    row.update({k: meta.get(k, 'NOT_AVAILABLE') for k in TIMING_KEYS})
with (OUT/'quality.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary={'group':'11.1','window':12,'cases_inferred':10,'cases_evaluated':10,'DINO':sum(r['DINO'] for r in rows)/10,'SSIM':sum(r['SSIM'] for r in rows)/10,'PSNR':sum(r['PSNR'] for r in rows)/10,'LPIPS':'NOT_AVAILABLE','protocol':'474 frames; 16 FPS; 832x480; frame 0 vs frame 237; DINOv2-small CLS; raw RGB SSIM/PSNR','note':'fresh 10-case evaluation; no inference was run'}
for key in TIMING_KEYS:
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, 'NOT_AVAILABLE')]
    summary[key] = sum(vals) / len(vals) if vals else 'NOT_AVAILABLE'
with (OUT/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
(OUT/'evaluation_manifest.json').write_text(json.dumps({'cases':[r['case'] for r in rows],'protocol':summary['protocol']},indent=2)+'\n')
print(json.dumps(summary,indent=2))
