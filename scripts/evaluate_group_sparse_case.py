#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import AutoImageProcessor, AutoModel


parser = argparse.ArgumentParser()
parser.add_argument('--video', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--group', type=int, required=True)
parser.add_argument('--ratio', type=float, required=True)
args = parser.parse_args()

video = Path(args.video)
cap = cv2.VideoCapture(str(video))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()


def read_frame(index):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'cannot read frame {index} from {video}')
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


first = read_frame(0)
returned = read_frame(frames // 2)
processor = AutoImageProcessor.from_pretrained('/data/zxl/eval_models/dinov2-small')
model = AutoModel.from_pretrained('/data/zxl/eval_models/dinov2-small').to('cuda').eval()
with torch.inference_mode():
    batch = processor(images=[Image.fromarray(first), Image.fromarray(returned)], return_tensors='pt')
    batch = {key: value.to('cuda') for key, value in batch.items()}
    features = torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0], dim=-1)
    dino = float((features[0] * features[1]).sum())

result = {
    'group': args.group,
    'retained_ratio_target': args.ratio,
    'video': str(video),
    'frames': frames,
    'fps': fps,
    'resolution': f'{width}x{height}',
    'first_frame': 0,
    'return_frame': frames // 2,
    'DINO': dino,
    'SSIM': float(structural_similarity(first, returned, channel_axis=2, data_range=255)),
    'PSNR': float(peak_signal_noise_ratio(first, returned, data_range=255)),
    'LPIPS': 'NOT_AVAILABLE',
    'protocol': 'canonical frame 0 vs midpoint; DINOv2-small CLS; raw RGB SSIM/PSNR',
}
Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
