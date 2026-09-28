#!/usr/bin/env python3
import csv, json
from pathlib import Path
import cv2, torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import AutoImageProcessor, AutoModel

ROOT=Path('/data/zxl/LongLive-RAG-group11_15_h200')
OUT=ROOT/'results/group14_full10_audit'
MANIFEST=Path('/data/zxl/LongLive-RAG-profile/prompts10.txt')
OUT.mkdir(parents=True,exist_ok=True)
processor=AutoImageProcessor.from_pretrained('/data/zxl/eval_models/dinov2-small')
model=AutoModel.from_pretrained('/data/zxl/eval_models/dinov2-small').to('cuda').eval()

def frame(path,idx):
    cap=cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,x=cap.read(); cap.release()
    if not ok: raise RuntimeError(f'cannot read {path} frame {idx}')
    return cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
@torch.inference_mode()
def dino(a,b):
    x=processor(images=[Image.fromarray(a),Image.fromarray(b)],return_tensors='pt')
    x={k:v.to('cuda') for k,v in x.items()}
    z=torch.nn.functional.normalize(model(**x).last_hidden_state[:,0],dim=-1)
    return float((z[0]*z[1]).sum().item())
def one(path,group,case,removed_percent):
    cap=cv2.VideoCapture(str(path)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
    if (n,fps,w,h)!=(474,16.0,832,480): raise RuntimeError(f'protocol mismatch {path}: {(n,fps,w,h)}')
    a,b=frame(path,0),frame(path,237)
    retained_ratio = 1.0 - removed_percent / 100.0
    return {'group':group,'case':case,'removed_percent':removed_percent,'retained_ratio':retained_ratio,'video':str(path),'num_frames':n,'fps':fps,'first_frame':0,'return_frame':237,'DINO':dino(a,b),'SSIM':float(structural_similarity(a,b,channel_axis=2,data_range=255)),'PSNR':float(peak_signal_noise_ratio(a,b,data_range=255)),'LPIPS':'NOT_AVAILABLE','protocol':'frame 0 vs frame 237; DINOv2-small CLS cosine; raw RGB SSIM/PSNR'}

rows=[]
for removed in (30,20,10,5):
    root=ROOT/'results/group12_15_persistent_campaign'/f'group14_sparse{removed:02d}'
    for i in range(1,11):
        p=root/f'case{i:02d}'/'rank0-0-0_lora.mp4'
        if not p.exists(): raise FileNotFoundError(p)
        rows.append(one(p,14,f'case_{i:02d}',removed))
with (OUT/'per_case.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summ=[]
for removed in (30,20,10,5):
    rr=[r for r in rows if r['removed_percent']==removed]
    summ.append({'group':14,'removed_percent':removed,'retained_ratio':1.0-removed/100.0,'cases':len(rr),'DINO':sum(r['DINO'] for r in rr)/len(rr),'SSIM':sum(r['SSIM'] for r in rr)/len(rr),'PSNR':sum(r['PSNR'] for r in rr)/len(rr),'LPIPS':'NOT_AVAILABLE'})
with (OUT/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)
(OUT/'audit_metadata.json').write_text(json.dumps({'manifest':str(MANIFEST),'manifest_sha256':'db9462c0954186b9128543b58fb6d1d496a8b89e0754333beca698d3cf461fec','seed':0,'protocol':'same canonical evaluator; frame 0 vs 237; 474 frames; 16 FPS; 832x480'},indent=2)+'\n')
print(OUT)
