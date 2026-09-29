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
    'H2D_CUDA_WORK_S', 'H2D_EXPOSED_WAIT_S', 'H2D_HOST_ENQUEUE_S', 'H2D_HIDDEN_S',
]
MEMORY_FIELDS = [
    'GPU_KV_GIB', 'CPU_KV_GIB', 'DRAFT_GPU_GIB', 'TRANSIENT_GPU_GIB',
    'PEAK_GPU_ALLOCATED_GIB', 'PEAK_GPU_RESERVED_GIB',
    'GPU_LOCAL_PERSISTENT_KV_GIB', 'GPU_HISTORICAL_PERSISTENT_RESIDENT_KV_GIB',
    'GPU_KV_PERSISTENT_MEAN_GIB', 'GPU_KV_PERSISTENT_PEAK_MEAN_GIB',
    'GPU_KV_PERSISTENT_PEAK_MAX_GIB', 'DRAFT_GPU_PERSISTENT_GIB',
    'GPU_METHOD_PERSISTENT_MEAN_GIB', 'GPU_METHOD_PERSISTENT_PEAK_GIB',
    'TRANSIENT_DEQUANT_GPU_PEAK_GIB', 'TRANSIENT_PROMOTION_GPU_PEAK_GIB',
    'TRANSIENT_HISTORICAL_GPU_PEAK_GIB',
    'KV_COMPRESSION_RATIO', 'KV_COMPRESSION_AT_PERSISTENT_PEAK',
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

def normalize_h2d_runtime(record):
    """Normalize modern and legacy H2D records without conflating semantics."""
    out = dict(record)
    def first(*keys, default='NOT_AVAILABLE'):
        for key in keys:
            value = out.get(key)
            if value not in (None, '', 'NA', 'N/A', 'NOT_AVAILABLE'):
                return value
        return default

    if 'H2D_CUDA_WORK_S' not in out:
        legacy = first('h2d_latency_s', 'legacy_historical_fetch_h2d_cuda_interval_s')
        out['H2D_CUDA_WORK_S'] = legacy
        if legacy != 'NOT_AVAILABLE':
            out.setdefault('H2D_SOURCE_CLASS', 'LEGACY_HISTORICAL_FETCH')
            out.setdefault('H2D_TIMING_CLASS', 'legacy_cuda_interval_including_stack_or_reshape')
    out.setdefault('H2D_EXPOSED_WAIT_S', first('h2d_exposed_wait_s', default='NOT_AVAILABLE'))
    out.setdefault('H2D_HOST_ENQUEUE_S', first('h2d_host_enqueue_s', default='NOT_AVAILABLE'))
    out.setdefault('H2D_HIDDEN_S', first('h2d_hidden_s', default='NOT_AVAILABLE'))
    out['H2D_TOTAL_BYTES'] = first('H2D_TOTAL_BYTES', 'h2d_total_bytes', 'H2D_BYTES', default='NOT_AVAILABLE')
    out['H2D_TOTAL_CALLS'] = first('H2D_TOTAL_CALLS', 'h2d_total_calls', 'H2D_CALLS', default='NOT_AVAILABLE')
    out['H2D_WORK_VALID'] = out.get('H2D_WORK_VALID', out.get('h2d_work_valid', False))
    out['H2D_EXPOSED_WAIT_VALID'] = out.get('H2D_EXPOSED_WAIT_VALID', out.get('h2d_exposed_valid', False))
    out['H2D_BYTES_VALID'] = out.get('H2D_BYTES_VALID', out.get('h2d_bytes_valid', False))
    if not (out['H2D_WORK_VALID'] and out['H2D_EXPOSED_WAIT_VALID']):
        out['H2D_HIDDEN_S'] = 'NOT_AVAILABLE'
    out.setdefault('H2D_SOURCE_CLASS', 'NOT_AVAILABLE')
    out.setdefault('H2D_TIMING_CLASS', 'NOT_AVAILABLE')
    return out

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
        'GPU Persistent KV Mean': m.get('GPU_KV_PERSISTENT_MEAN_GIB', 'NOT_AVAILABLE'),
        'GPU Persistent KV Peak Mean': m.get('GPU_KV_PERSISTENT_PEAK_MEAN_GIB', 'NOT_AVAILABLE'),
        'GPU Persistent KV Peak Max': m.get('GPU_KV_PERSISTENT_PEAK_MAX_GIB', 'NOT_AVAILABLE'),
        'GPU Local Persistent KV': m.get('GPU_LOCAL_PERSISTENT_KV_GIB', 'NOT_AVAILABLE'),
        'GPU Historical Persistent Resident KV': m.get('GPU_HISTORICAL_PERSISTENT_RESIDENT_KV_GIB', 'NOT_AVAILABLE'),
        'Draft GPU Persistent': m.get('DRAFT_GPU_PERSISTENT_GIB', 'NOT_AVAILABLE'),
        'GPU Method Persistent Mean': m.get('GPU_METHOD_PERSISTENT_MEAN_GIB', 'NOT_AVAILABLE'),
        'GPU Method Persistent Peak': m.get('GPU_METHOD_PERSISTENT_PEAK_GIB', 'NOT_AVAILABLE'),
        'Transient Dequant Peak': m.get('TRANSIENT_DEQUANT_GPU_PEAK_GIB', 'NOT_AVAILABLE'),
        'Transient Promotion Peak': m.get('TRANSIENT_PROMOTION_GPU_PEAK_GIB', 'NOT_AVAILABLE'),
        'Transient Historical Peak': m.get('TRANSIENT_HISTORICAL_GPU_PEAK_GIB', 'NOT_AVAILABLE'),
        'KV Compression @ Persistent Peak': m.get('KV_COMPRESSION_AT_PERSISTENT_PEAK', 'NOT_AVAILABLE'),
        'DINO': q.get('DINO', 'NOT_AVAILABLE'), 'SSIM': q.get('SSIM', 'NOT_AVAILABLE'),
        'PSNR': q.get('PSNR', 'NOT_AVAILABLE'), 'LPIPS': q.get('LPIPS', 'NOT_AVAILABLE'),
        'E2E': l.get('E2E_INFERENCE_S', 'NOT_AVAILABLE'), 'Transformer': l.get('TRANSFORMER_S', 'NOT_AVAILABLE'),
        'Self-Attention Wrapper': l.get('SELF_ATTN_WRAPPER_S', 'NOT_AVAILABLE'),
        'Attention Kernel': l.get('ATTENTION_KERNEL_S', 'NOT_AVAILABLE'),
        'Fetch Work': l.get('FETCH_WORK_S', 'NOT_AVAILABLE'), 'Fetch Exposed': l.get('FETCH_EXPOSED_S', 'NOT_AVAILABLE'),
        'Fetch Hidden': l.get('FETCH_HIDDEN_S', 'NOT_AVAILABLE'), 'Process Wall': l.get('PROCESS_WALL_S', 'NOT_AVAILABLE'),
        'H2D CUDA Work': l.get('H2D_CUDA_WORK_S', 'NOT_AVAILABLE'),
        'H2D Exposed Wait': l.get('H2D_EXPOSED_WAIT_S', 'NOT_AVAILABLE'),
        'H2D Host Enqueue': l.get('H2D_HOST_ENQUEUE_S', 'NOT_AVAILABLE'),
        'H2D Hidden': l.get('H2D_HIDDEN_S', 'NOT_AVAILABLE'),
        'H2D Bytes': l.get('H2D_TOTAL_BYTES', 'NOT_AVAILABLE'),
        'H2D Calls': l.get('H2D_TOTAL_CALLS', 'NOT_AVAILABLE'),
        'H2D Work Valid': l.get('H2D_WORK_VALID', False),
        'H2D Exposed Valid': l.get('H2D_EXPOSED_WAIT_VALID', False),
        'H2D Bytes Valid': l.get('H2D_BYTES_VALID', False),
        'H2D Source Class': l.get('H2D_SOURCE_CLASS', 'NOT_AVAILABLE'),
        'H2D Timing Class': l.get('H2D_TIMING_CLASS', 'NOT_AVAILABLE'),
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
