"""Portable schema helpers for the unified evaluation pipeline."""
import csv
import json
from pathlib import Path

QUALITY_FIELDS = ['DINO', 'SSIM', 'PSNR', 'LPIPS']
LATENCY_FIELDS = [
    'E2E_INFERENCE_S', 'TRANSFORMER_S', 'SELF_ATTN_WRAPPER_S',
    'ATTENTION_KERNEL_S', 'NON_TRANSFORMER_E2E_S', 'FETCH_WORK_S',
    'FETCH_EXPOSED_S', 'FETCH_HIDDEN_S', 'VAE_DECODE_S',
    'VIDEO_ENCODE_S', 'PROCESS_WALL_S',
]
MEMORY_FIELDS = [
    'GPU_KV_GIB', 'CPU_KV_GIB', 'DRAFT_GPU_GIB', 'TRANSIENT_GPU_GIB',
    'PEAK_GPU_ALLOCATED_GIB', 'PEAK_GPU_RESERVED_GIB',
    'KV_COMPRESSION_RATIO',
]

def number(value):
    if value is None or value == '' or value in {'NA', 'N/A', 'NOT_AVAILABLE'}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def val(d, *keys, default='NOT_AVAILABLE'):
    for k in keys:
        if k in d and d[k] not in (None, '', 'NA', 'N/A', 'NOT_AVAILABLE'):
            return d[k]
    return default

def load_json(path):
    return json.loads(Path(path).read_text())

def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def as_csv_row(summary):
    q, m, l, p, v = summary['quality'], summary['memory'], summary['latency'], summary['provenance'], summary['validity']
    return {
        'Method': p.get('method', 'NOT_AVAILABLE'), 'Window': p.get('window', 'NOT_AVAILABLE'),
        'KV Strategy': p.get('kv_storage_policy', 'NOT_AVAILABLE'), 'Precision': p.get('attention_precision', 'NOT_AVAILABLE'),
        'GPU KV': m.get('GPU_KV_GIB', 'NOT_AVAILABLE'), 'CPU KV': m.get('CPU_KV_GIB', 'NOT_AVAILABLE'),
        'Draft GPU': m.get('DRAFT_GPU_GIB', 'NOT_AVAILABLE'), 'Transient GPU': m.get('TRANSIENT_GPU_GIB', 'NOT_AVAILABLE'),
        'Compression': m.get('KV_COMPRESSION_RATIO', 'NOT_AVAILABLE'),
        'DINO': q.get('DINO', 'NOT_AVAILABLE'), 'SSIM': q.get('SSIM', 'NOT_AVAILABLE'),
        'PSNR': q.get('PSNR', 'NOT_AVAILABLE'), 'LPIPS': q.get('LPIPS', 'NOT_AVAILABLE'),
        'E2E': l.get('E2E_INFERENCE_S', 'NOT_AVAILABLE'), 'Transformer': l.get('TRANSFORMER_S', 'NOT_AVAILABLE'),
        'Self-Attention Wrapper': l.get('SELF_ATTN_WRAPPER_S', 'NOT_AVAILABLE'),
        'Attention Kernel': l.get('ATTENTION_KERNEL_S', 'NOT_AVAILABLE'),
        'Fetch Work': l.get('FETCH_WORK_S', 'NOT_AVAILABLE'), 'Fetch Exposed': l.get('FETCH_EXPOSED_S', 'NOT_AVAILABLE'),
        'Fetch Hidden': l.get('FETCH_HIDDEN_S', 'NOT_AVAILABLE'), 'Process Wall': l.get('PROCESS_WALL_S', 'NOT_AVAILABLE'),
        'Quality Source': p.get('quality_source', 'NOT_AVAILABLE'), 'Latency Source': p.get('latency_source', 'NOT_AVAILABLE'),
        'Memory Source': p.get('memory_source', 'NOT_AVAILABLE'), 'Quality Valid': v.get('QUALITY_VALID', False),
        'Latency Valid': v.get('LATENCY_VALID', False), 'Memory Valid': v.get('MEMORY_VALID', False),
        'Final Row Valid': v.get('FINAL_ROW_VALID', False),
    }

def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
