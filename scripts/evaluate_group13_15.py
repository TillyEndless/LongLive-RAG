#!/usr/bin/env python
import argparse,csv,cv2,json,math
from pathlib import Path
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
root=Path(args.root); out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
proc=AutoImageProcessor.from_pretrained('/data/zxl/eval_models/dinov2-small')
model=AutoModel.from_pretrained('/data/zxl/eval_models/dinov2-small').to(device).eval()
def frame(p,idx):
 c=cv2.VideoCapture(str(p)); c.set(cv2.CAP_PROP_POS_FRAMES,idx); ok,x=c.read(); c.release()
 if not ok: raise RuntimeError(f'frame {idx} failed {p}')
 return cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
@torch.inference_mode()
def dino(a,b):
 z=proc(images=[Image.fromarray(a),Image.fromarray(b)],return_tensors='pt')
 z={k:v.to(device) for k,v in z.items()}
 h=torch.nn.functional.normalize(model(**z).last_hidden_state[:,0],dim=-1)
 return float((h[0]*h[1]).sum())
rows=[]
for case in sorted(root.glob('case_*')):
 p=case/'rank0-0-0_lora.mp4'
 if not p.is_file(): continue
 cap=cv2.VideoCapture(str(p)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
 a=frame(p,0); b=frame(p,237)
 rows.append({'case':case.name,'video':str(p),'num_frames':n,'fps':fps,'width':w,'height':h,'frame0':0,'return_frame':237,'DINO':dino(a,b),'SSIM':float(structural_similarity(a,b,channel_axis=2,data_range=255)),'PSNR':float(peak_signal_noise_ratio(a,b,data_range=255))})
with (out/'quality.csv').open('w',newline='') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['case']); wr.writeheader(); wr.writerows(rows)
if rows:
 summ={k:sum(float(r[k]) for r in rows)/len(rows) for k in ['DINO','SSIM','PSNR']}
 summ['cases']=len(rows); summ['root']=str(root)
 (out/'summary.json').write_text(json.dumps(summ,indent=2))
print('evaluated',len(rows),'root',root)
