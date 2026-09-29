#!/usr/bin/env python3
"""Per-case evaluator entry point; intentionally has no canonical10 gate."""
from __future__ import annotations
import argparse, json, glob, csv
from pathlib import Path
import cv2
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import AutoImageProcessor, AutoModel

def read_frame(path, idx):
    cap=cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame=cap.read(); cap.release()
    if not ok: raise RuntimeError(f"cannot read frame {idx}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

@torch.inference_mode()
def dino_score(a,b):
    proc=AutoImageProcessor.from_pretrained('/data/zxl/eval_models/dinov2-small')
    model=AutoModel.from_pretrained('/data/zxl/eval_models/dinov2-small').to('cuda').eval()
    x=proc(images=[Image.fromarray(a),Image.fromarray(b)],return_tensors='pt')
    x={k:v.to('cuda') for k,v in x.items()}; f=torch.nn.functional.normalize(model(**x).last_hidden_state[:,0],dim=-1)
    return float((f[0]*f[1]).sum().item())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--label",required=True); ap.add_argument("--case",required=True); ap.add_argument("--output-root",required=True); a=ap.parse_args()
    root=Path(a.output_root)/"outputs"/a.label/a.case
    videos=sorted(glob.glob(str(root/"*.mp4")))
    result={"label":a.label,"case":a.case,"video":videos[0] if videos else None,
            "protocol":{"frames":474,"fps":16,"resolution":"832x480","pair":[0,237]},
            "dino":None,"ssim":None,"psnr":None,"lpips":"NOT_AVAILABLE",
            "status":"NOT_AVAILABLE"}
    if videos:
        video=Path(videos[0]); cap=cv2.VideoCapture(str(video)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
        result.update({'frames':n,'fps':fps,'width':w,'height':h})
        if n==474 and abs(fps-16)<.01 and (w,h)==(832,480):
            x,y=read_frame(video,0),read_frame(video,237)
            result.update({'dino':dino_score(x,y),'ssim':float(structural_similarity(x,y,channel_axis=2,data_range=255)),'psnr':float(peak_signal_noise_ratio(x,y,data_range=255)),'status':'OK'})
        else: result['status']='PROTOCOL_MISMATCH'
        for suffix in ('_runtime.json','_memory_measurement.json'):
            p=video.with_name(video.stem+suffix)
            if p.exists(): result[suffix[1:-5]]=json.loads(p.read_text())
    out=Path(a.output_root)/"evaluation"/a.label/a.case; out.mkdir(parents=True,exist_ok=True)
    (out/"evaluation.json").write_text(json.dumps(result,indent=2))
    return 0
if __name__ == "__main__": raise SystemExit(main())
