#!/usr/bin/env python3
import csv
import hashlib
import json
from pathlib import Path

import cv2
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import AutoImageProcessor, AutoModel

# This legacy helper previously evaluated Group14/15 case_01 only.  Refuse to
# run it so it cannot silently create a non-canonical aggregate.  The active
# canonical runner uses evaluate_group12_15_corrected.py, which enforces ten
# cases per label before writing a summary.
raise RuntimeError(
    'LEGACY_EVALUATOR_DISABLED: use scripts/evaluate_group12_15_corrected.py '
    'for strict canonical10 evaluation'
)

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
OUT = ROOT / 'results/canonical_aligned_evaluation'
OUT.mkdir(parents=True, exist_ok=True)

processor = AutoImageProcessor.from_pretrained('/data/zxl/eval_models/dinov2-small')
model = AutoModel.from_pretrained('/data/zxl/eval_models/dinov2-small').to('cuda').eval()


def read_frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f'cannot read frame {index} from {path}')
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


@torch.inference_mode()
def dino_similarity(first, returned):
    batch = processor(images=[Image.fromarray(first), Image.fromarray(returned)], return_tensors='pt')
    batch = {key: value.to('cuda') for key, value in batch.items()}
    features = model(**batch).last_hidden_state[:, 0]
    features = torch.nn.functional.normalize(features, dim=-1)
    return float((features[0] * features[1]).sum().item())


def evaluate_video(group, case, video, ratio=None):
    cap = cv2.VideoCapture(str(video))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if frames != 474 or abs(fps - 16.0) > 0.01 or (width, height) != (832, 480):
        raise RuntimeError(f'protocol mismatch for {video}: {frames=} {fps=} {width=} {height=}')
    first = read_frame(video, 0)
    returned = read_frame(video, 237)
    return {
        'group': group,
        'case': case,
        'retained_ratio': ratio if ratio is not None else 'NA',
        'video': str(video),
        'num_frames': frames,
        'fps': fps,
        'first_frame': 0,
        'return_frame': 237,
        'DINO': dino_similarity(first, returned),
        'SSIM': float(structural_similarity(first, returned, channel_axis=2, data_range=255)),
        'PSNR': float(peak_signal_noise_ratio(first, returned, data_range=255)),
        'LPIPS': 'NOT_AVAILABLE',
        'protocol': 'frame 0 vs frame 237; DINOv2-small CLS cosine; raw RGB SSIM/PSNR',
    }


def collect_group(group, pairs):
    rows = []
    for case, video, ratio in pairs:
        rows.append(evaluate_video(group, case, video, ratio))
    target = OUT / f'group{group}'
    target.mkdir(parents=True, exist_ok=True)
    with (target / 'per_case.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def mean(key):
        values = [float(row[key]) for row in rows]
        return sum(values) / len(values)
    summary = {
        'group': group,
        'cases': len(rows),
        'DINO': mean('DINO'),
        'SSIM': mean('SSIM'),
        'PSNR': mean('PSNR'),
        'LPIPS': 'NOT_AVAILABLE',
        'protocol': rows[0]['protocol'],
    }
    (target / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return rows, summary


def case10(prefix, case01):
    pairs = [('case_01', ROOT / 'results' / case01 / 'rank0-0-0_lora.mp4', None)]
    pairs += [
        (f'case_{index:02d}', ROOT / 'results' / prefix / f'case_{index:02d}' / 'rank0-0-0_lora.mp4', None)
        for index in range(2, 11)
    ]
    return pairs


all_rows = []
all_rows += collect_group(12, case10('group12_corrected_canonical10', 'group12_corrected_case01_rerun'))[0]
all_rows += collect_group(13, case10('group13_corrected_canonical10', 'group13_corrected_case01_rerun4'))[0]

for group in (14, 15):
    pairs = []
    for percent in (30, 20, 10, 5):
        video = ROOT / 'results' / f'group{group}_case01_retained{percent:02d}' / 'rank0-0-0_lora.mp4'
        if video.exists():
            pairs.append(('case_01', video, percent))
    if pairs:
        all_rows += collect_group(group, pairs)[0]

with (OUT / 'all_metrics.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
    writer.writeheader()
    writer.writerows(all_rows)

manifest = Path('/data/zxl/LongLive-RAG-profile/prompts10.txt')
manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
(OUT / 'evaluation_manifest.json').write_text(json.dumps({
    'manifest': str(manifest),
    'manifest_sha256': manifest_hash,
    'seed': 0,
    'protocol': 'Group11/Group5 canonical metric implementation',
    'frame_pair': [0, 237],
    'video_format': {'frames': 474, 'fps': 16, 'resolution': '832x480'},
}, indent=2) + '\n')
print(f'WROTE {OUT}')
