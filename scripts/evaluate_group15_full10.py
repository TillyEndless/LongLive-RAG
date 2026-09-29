#!/usr/bin/env python3
import csv, json
from pathlib import Path
import cv2, torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import AutoImageProcessor, AutoModel

ROOT=Path('/data/zxl/LongLive-RAG-group11_15_h200')
BASE=ROOT/'results/group12_15_persistent_campaign'
OUT=ROOT/'results/group15_first3_evaluation'
OUT.mkdir(parents=True,exist_ok=True)
processor=AutoImageProcessor.from_pretrained('/data/zxl/eval_models/dinov2-small')
model=AutoModel.from_pretrained('/data/zxl/eval_models/dinov2-small').to('cuda').eval()

def read_frame(path,idx):
    cap=cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,img=cap.read(); cap.release()
    if not ok: raise RuntimeError(f'cannot read {path} frame {idx}')
    return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
@torch.inference_mode()
def dino(a,b):
    x=processor(images=[Image.fromarray(a),Image.fromarray(b)],return_tensors='pt')
    x={k:v.to('cuda') for k,v in x.items()}
    z=torch.nn.functional.normalize(model(**x).last_hidden_state[:,0],dim=-1)
    return float((z[0]*z[1]).sum().item())
def load(p): return json.loads(p.read_text())

rows=[]
for removed in (30,20,10,5):
    root=BASE/f'group15_sparse{removed:02d}'
    for i in range(1,4):
        case=f'case{i:02d}'
        video=root/case/'rank0-0-0_lora.mp4'
        runtime=root/case/'rank0-0-0_lora_runtime.json'
        if not video.exists() or not runtime.exists(): raise FileNotFoundError(str(video))
        meta=load(runtime)
        cap=cv2.VideoCapture(str(video)); frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
        if (frames,fps,w,h)!=(474,16.0,832,480): raise RuntimeError(f'protocol mismatch {video}: {(frames,fps,w,h)}')
        a,b=read_frame(video,0),read_frame(video,237)
        rows.append({'removed_percent':removed,'retained_ratio':1.0-removed/100.0,'case':case,'dino':dino(a,b),'ssim':float(structural_similarity(a,b,channel_axis=2,data_range=255)),'psnr':float(peak_signal_noise_ratio(a,b,data_range=255)),'actual_sparse_ratio':meta.get('SPARSE_RATIO'),'actual_retained_blocks':meta.get('ROUTE_BLOCKS_RETAINED'),'total_blocks':meta.get('ROUTE_BLOCKS_TOTAL'),'actual_retained_fraction':meta.get('ACTUAL_BF16_FRACTION'),'actual_zero_fraction':meta.get('ACTUAL_ZERO_FRACTION'),'sparse_execution_status':meta.get('SPARSE_EXECUTION_STATUS'),'video':str(video),'frames':frames,'fps':fps,'first_frame':0,'return_frame':237,'protocol':'frame 0 vs frame 237; DINOv2-small CLS cosine; raw RGB SSIM/PSNR'})

with (OUT/'per_case.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary=[]
for removed in (30,20,10,5):
    rr=[r for r in rows if r['removed_percent']==removed]
    summary.append({'removed_percent':removed,'retained_ratio':1.0-removed/100.0,'cases':len(rr),'DINO':sum(r['dino'] for r in rr)/len(rr),'SSIM':sum(r['ssim'] for r in rr)/len(rr),'PSNR':sum(r['psnr'] for r in rr)/len(rr),'actual_sparse_ratio':sorted(set(r['actual_sparse_ratio'] for r in rr)),'actual_retained_blocks':sorted(set(r['actual_retained_blocks'] for r in rr)),'total_blocks':sorted(set(r['total_blocks'] for r in rr)),'actual_retained_fraction':sorted(set(r['actual_retained_fraction'] for r in rr)),'actual_zero_fraction':sorted(set(r['actual_zero_fraction'] for r in rr)),'sparse_execution_status':sorted(set(r['sparse_execution_status'] for r in rr))})
with (OUT/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
(OUT/'README.md').write_text('# Group15 full canonical3 evaluation\n\nExisting videos only; no inference. Protocol: 474 frames, 16 FPS, 832x480, frame 0 vs 237, DINOv2-small CLS cosine, raw RGB SSIM/PSNR.\n')
print(OUT)
