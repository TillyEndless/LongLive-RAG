import csv, json
from pathlib import Path
import cv2, torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import AutoImageProcessor, AutoModel

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
SRC = ROOT / 'results/group14_15_corrected_sparse/group15_sparsity30'
OUT = ROOT / 'results/group14_15_corrected_sparse/evaluation_sparsity30/group15'
OUT.mkdir(parents=True, exist_ok=True)
processor = AutoImageProcessor.from_pretrained('/data/zxl/eval_models/dinov2-small')
model = AutoModel.from_pretrained('/data/zxl/eval_models/dinov2-small').to('cuda').eval()

@torch.inference_mode()
def dino(a, b):
    x = processor(images=[Image.fromarray(a), Image.fromarray(b)], return_tensors='pt')
    x = {k: v.to('cuda') for k, v in x.items()}
    f = torch.nn.functional.normalize(model(**x).last_hidden_state[:, 0], dim=-1)
    return float((f[0] * f[1]).sum().item())

rows=[]
for i in range(1, 11):
    case=f'case{i:02d}'; out=SRC/case
    videos=sorted(out.glob('*.mp4'))
    if len(videos)!=1: raise RuntimeError(f'{case}: expected one video, found {len(videos)}')
    video=videos[0]
    cap=cv2.VideoCapture(str(video)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS)); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_POS_FRAMES,0); ok,a=cap.read(); cap.set(cv2.CAP_PROP_POS_FRAMES,237); ok2,b=cap.read(); cap.release()
    if not (ok and ok2 and n==474 and abs(fps-16)<.01 and (w,h)==(832,480)): raise RuntimeError(f'{case}: protocol mismatch {(n,fps,w,h,ok,ok2)}')
    a=cv2.cvtColor(a,cv2.COLOR_BGR2RGB); b=cv2.cvtColor(b,cv2.COLOR_BGR2RGB)
    rows.append({'group':15,'label':'group15_sparsity30','case':case,'retained_fraction':0.7,'video':str(video),'frames':n,'fps':fps,'first_frame':0,'return_frame':237,'DINO':dino(a,b),'SSIM':float(structural_similarity(a,b,channel_axis=2,data_range=255)),'PSNR':float(peak_signal_noise_ratio(a,b,data_range=255)),'LPIPS':'NOT_AVAILABLE','protocol':'474 frames; 16 FPS; 832x480; frame 0 vs 237; DINOv2-small CLS; raw RGB SSIM/PSNR'})

with (OUT/'per_case.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
summary={'group':15,'label':'group15_sparsity30','retained_fraction':0.7,'cases':10,'DINO':sum(r['DINO'] for r in rows)/10,'SSIM':sum(r['SSIM'] for r in rows)/10,'PSNR':sum(r['PSNR'] for r in rows)/10,'LPIPS':'NOT_AVAILABLE','protocol':'474 frames; 16 FPS; 832x480; frame 0 vs 237; DINOv2-small CLS; raw RGB SSIM/PSNR'}
with (OUT/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary)); w.writeheader(); w.writerow(summary)
(OUT/'provenance.json').write_text(json.dumps({'source':str(SRC),'cases':10,'corrected_semantics':'sparsity30 means remove 30%, retain 70%','seed':0,'protocol':summary['protocol']},indent=2)+'\n')
print(json.dumps(summary, indent=2))
