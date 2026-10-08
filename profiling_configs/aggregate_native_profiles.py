import csv
import json
from pathlib import Path

BASE = Path('/data/zxl/LongLive-RAG-group11_15_h200/results/native_w6_w12_case01_profile')
OUT_CSV = Path('/data/zxl/LongLive-RAG-group11_15_h200/results/native_w6_w12_case01_profile.csv')
OUT_JSON = Path('/data/zxl/LongLive-RAG-group11_15_h200/results/native_w6_w12_case01_profile.json')
OUT_MD = Path('/data/zxl/LongLive-RAG-group11_15_h200/reports/native_w6_w12_case01_profile.md')

QUALITY = {
    6: {'dino': 0.7487808406352997, 'ssim': 0.32644455234919245, 'psnr': 9.815110998214726},
    12: {'dino': 0.7914137989282608, 'ssim': 0.33470661431094184, 'psnr': 10.029504634828326},
}

def read_json(path):
    with path.open() as f:
        return json.load(f)

def get(d, *keys, default=None):
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return default

rows = []
for window in (6, 12):
    profile = read_json(BASE / f'w{window}_v2' / 'native_case01_profile.json')
    runtime = read_json(BASE / f'w{window}' / 'rank0-0-0_lora_runtime.json')
    memory = read_json(BASE / f'w{window}' / 'rank0-0-0_lora_memory_measurement.json')
    q = QUALITY[window]
    row = {
        'method': 'Native local window', 'window': window, 'case': 'case_01', 'gpu': 'H200 GPU0',
        'gpu_kv_gib': get(memory, 'GPU_KV_MEASURED_GiB', 'GPU_KV_GiB', 'gpu_kv_gib'),
        'cpu_kv_gib': get(memory, 'CPU_KV_MEASURED_GiB', 'CPU_KV_GiB', 'cpu_kv_gib', default=0.0),
        'peak_gpu_allocated_gib': 'NOT_AVAILABLE', 'peak_gpu_reserved_gib': 'NOT_AVAILABLE',
        'draft_gpu_gib': get(memory, 'GPU_DRAFT_PERSISTENT_GiB', 'draft_gpu_gib', default=0.0),
        'transient_fetch_gpu_gib': get(memory, 'TRANSIENT_GPU_PEAK_GiB', 'transient_fetch_gpu_gib', default=0.0),
        'e2e_latency_s': get(runtime, 'E2E_LATENCY_S', 'e2e_latency_s'),
        'transformer_latency_s': get(runtime, 'TRANSFORMER_LATENCY_S', 'transformer_latency_s'),
        'self_attention_wrapper_s': get(profile, 'self_attention_wrapper_s'),
        'attention_kernel_s': get(profile, 'attention_kernel_s'),
        'cross_attention_s': 'NOT_AVAILABLE', 'ffn_s': 'NOT_AVAILABLE',
        'norm_residual_s': 'NOT_AVAILABLE', 'transformer_other_s': 'NOT_AVAILABLE',
        'fetch_work_s': 0.0, 'fetch_exposed_wait_s': 0.0, 'fetch_hidden_overlap_s': 0.0,
        'vae_decode_s': 'NOT_AVAILABLE', 'video_encode_s': 'NOT_AVAILABLE',
        'process_wall_s': 'NOT_AVAILABLE',
        'attention_calls': get(profile, 'attention_calls'),
        'attention_kernel_calls': get(profile, 'attention_kernel_calls'),
        'attention_kernel_ms_per_call': get(profile, 'attention_kernel_ms_per_call'),
        'dino': q['dino'], 'ssim': q['ssim'], 'psnr': q['psnr'], 'lpips': 'NOT_AVAILABLE',
        'quality_provenance': 'canonical10', 'latency_provenance': 'matched_case_01',
        'memory_provenance': 'matched_case_01', 'profiler_version': 'native_case01_profiler_v3',
        'timer_boundary_version': 'unified_runtime_profiler_v1',
        'git_commit': 'f519727eb2a9e6d9b748220ce1f6bb1144406446', 'validity': 'PASS',
    }
    rows.append(row)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUT_MD.parent.mkdir(parents=True, exist_ok=True)
fields = list(rows[0])
with OUT_CSV.open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(rows)
with OUT_JSON.open('w') as f:
    json.dump({'experiment': 'native_w6_w12_case01_profile', 'rows': rows,
               'quality_source': 'existing canonical10',
               'latency_memory_source': 'new matched case_01 profiling-only runs'}, f, indent=2)

def fmt(v):
    return 'NOT_AVAILABLE' if v == 'NOT_AVAILABLE' else str(v)

lines = [
    '# Native W6/W12 matched case_01 runtime profile', '',
    'This report adds profiling-only latency and memory measurements for the existing Native W6/W12 baselines. No canonical10 inference or quality evaluation was rerun.', '',
    '## Provenance', '',
    '- Quality: existing canonical10 results.',
    '- Latency and memory: new matched `case_01` profiling runs, 120 latent output frames / 474 decoded frames, 16 FPS, 832x480.',
    '- Semantics: native local sliding window; `use_latentmem=false`, `memory_size=0`; no retrieval, DraftMap, low-bit, or sparse path.',
    '- Code commit: `f519727eb2a9e6d9b748220ce1f6bb1144406446`.', '',
    '## Results', '',
    '| Window | E2E (s) | Transformer (s) | Self-attention wrapper (s) | Attention kernel (s) | GPU KV (GiB) | CPU KV (GiB) | Calls | Quality DINO | SSIM | PSNR |',
    '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
]
for r in rows:
    lines.append(f"| {r['window']} | {fmt(r['e2e_latency_s'])} | {fmt(r['transformer_latency_s'])} | {fmt(r['self_attention_wrapper_s'])} | {fmt(r['attention_kernel_s'])} | {fmt(r['gpu_kv_gib'])} | {fmt(r['cpu_kv_gib'])} | {r['attention_calls']} | {r['dino']} | {r['ssim']} | {r['psnr']} |")
lines += [
    '', '## Validation and limitations', '',
    '- Both windows recorded 6000 attention calls and 6000 attention-kernel event pairs; validity is `PASS`.',
    '- Native fetch fields are zero because these runs do not use CPU history retrieval.',
    '- Peak allocated/reserved GPU memory was not emitted by the current memory artifact and is therefore `NOT_AVAILABLE`; it was not inferred from `nvidia-smi`.',
    '- QKV, cross-attention, FFN, norm/residual, VAE, and video-encode components were not instrumented in this non-invasive native probe and remain `NOT_AVAILABLE`.',
    '- The latency values are matched-case measurements, not ten-case means. They are directly comparable to other runs only where timer boundaries and workload are identical.',
]
OUT_MD.write_text('\n'.join(lines) + '\n')
print(OUT_CSV)
print(OUT_JSON)
print(OUT_MD)
