"""Unified evaluation entry point.

Reuse mode consumes existing artifacts only. Full mode optionally executes an
explicit command from the config, then consumes the resulting artifacts. This
module never changes inference/profiler semantics.
"""
import argparse, json, subprocess, sys
from pathlib import Path
from unified_evaluation_schema import (LATENCY_FIELDS, MEMORY_FIELDS, QUALITY_FIELDS,
    as_csv_row, load_json, normalize_h2d_runtime, number, sha256, val, write_csv)

def _source(config, key, default=None):
    p = config.get(key)
    return Path(p) if p else default

def _bytes_to_gib(value):
    if value in (None, '', 'NA', 'N/A', 'NOT_AVAILABLE'):
        return 'NOT_AVAILABLE'
    try:
        # Preserve already-normalized GiB values used by older adapters.
        x = float(value)
        return x / (2 ** 30) if abs(x) >= 1024 * 1024 else x
    except (TypeError, ValueError):
        return 'NOT_AVAILABLE'

def build_summary(cfg):
    profile = load_json(_source(cfg, 'profile_json')) if cfg.get('profile_json') else {}
    runtime = load_json(_source(cfg, 'runtime_json')) if cfg.get('runtime_json') else {}
    memory = load_json(_source(cfg, 'memory_json')) if cfg.get('memory_json') else {}
    runtime = normalize_h2d_runtime(runtime)
    profile = normalize_h2d_runtime({**runtime, **profile})
    quality = dict(cfg.get('quality', {}))
    for k in QUALITY_FIELDS:
        quality.setdefault(k, 'NOT_AVAILABLE')
    e2e = val(profile, 'e2e_s', default=val(runtime, 'E2E_LATENCY_S', 'e2e_latency_s'))
    transformer = val(profile, 'transformer_s', default=val(runtime, 'TRANSFORMER_LATENCY_S', 'transformer_latency_s'))
    wrapper = val(profile, 'self_attention_wrapper_s', default='NOT_AVAILABLE')
    latency = {
        'E2E_INFERENCE_S': e2e, 'TRANSFORMER_S': transformer,
        'SELF_ATTN_WRAPPER_S': wrapper,
        'ATTENTION_KERNEL_S': val(profile, 'attention_kernel_s', default='NOT_AVAILABLE'),
        'NON_TRANSFORMER_E2E_S': (number(e2e) - number(transformer)) if number(e2e) is not None and number(transformer) is not None else 'NOT_AVAILABLE',
        'FETCH_WORK_S': val(profile, 'fetch_work_s', default=0.0),
        'FETCH_EXPOSED_S': val(profile, 'fetch_exposed_s', default=0.0),
        'FETCH_HIDDEN_S': val(profile, 'fetch_hidden_s', default='NOT_AVAILABLE'),
        'H2D_CUDA_WORK_S': val(profile, 'H2D_CUDA_WORK_S', default='NOT_AVAILABLE'),
        'H2D_EXPOSED_WAIT_S': val(profile, 'H2D_EXPOSED_WAIT_S', default='NOT_AVAILABLE'),
        'H2D_HOST_ENQUEUE_S': val(profile, 'H2D_HOST_ENQUEUE_S', default='NOT_AVAILABLE'),
        'H2D_HIDDEN_S': val(profile, 'H2D_HIDDEN_S', default='NOT_AVAILABLE'),
        'H2D_TOTAL_BYTES': val(profile, 'H2D_TOTAL_BYTES', default='NOT_AVAILABLE'),
        'H2D_TOTAL_CALLS': val(profile, 'H2D_TOTAL_CALLS', default='NOT_AVAILABLE'),
        'H2D_WORK_VALID': profile.get('H2D_WORK_VALID', False),
        'H2D_EXPOSED_WAIT_VALID': profile.get('H2D_EXPOSED_WAIT_VALID', False),
        'H2D_BYTES_VALID': profile.get('H2D_BYTES_VALID', False),
        'H2D_SOURCE_CLASS': profile.get('H2D_SOURCE_CLASS', 'NOT_AVAILABLE'),
        'H2D_TIMING_CLASS': profile.get('H2D_TIMING_CLASS', 'NOT_AVAILABLE'),
        'VAE_DECODE_S': val(profile, 'vae_decode_s', default='NOT_AVAILABLE'),
        'VIDEO_ENCODE_S': val(profile, 'video_encode_s', default='NOT_AVAILABLE'),
        'PROCESS_WALL_S': val(profile, 'process_wall_s', default='NOT_AVAILABLE'),
    }
    memory_out = {
        'GPU_KV_GIB': val(profile, 'gpu_kv_gib', default=val(memory, 'GPU_KV_MEASURED_GiB', 'GPU_KV_GiB')),
        'CPU_KV_GIB': val(profile, 'cpu_kv_gib', default=val(memory, 'CPU_KV_MEASURED_GiB', 'CPU_KV_GiB', default=0.0)),
        'DRAFT_GPU_GIB': val(profile, 'draft_gpu_gib', default=val(memory, 'GPU_DRAFT_PERSISTENT_GiB', default=0.0)),
        'TRANSIENT_GPU_GIB': val(profile, 'transient_fetch_gpu_gib', default=val(memory, 'TRANSIENT_PREFETCH_GPU_PEAK_GiB', default=0.0)),
        'PEAK_GPU_ALLOCATED_GIB': val(profile, 'peak_gpu_allocated_gib', default='NOT_AVAILABLE'),
        'PEAK_GPU_RESERVED_GIB': val(profile, 'peak_gpu_reserved_gib', default='NOT_AVAILABLE'),
        'KV_COMPRESSION_RATIO': val(memory, 'GPU_KV_COMPRESSION_RATIO', 'KV_COMPRESSION_RATIO', default='NOT_AVAILABLE'),
        'GPU_LOCAL_PERSISTENT_KV_GIB': _bytes_to_gib(val(memory, 'GPU_LOCAL_PERSISTENT_KV_BYTES', default='NOT_AVAILABLE')),
        'GPU_HISTORICAL_PERSISTENT_RESIDENT_KV_GIB': _bytes_to_gib(val(memory, 'GPU_HISTORICAL_PERSISTENT_RESIDENT_KV_BYTES', default='NOT_AVAILABLE')),
        'GPU_KV_PERSISTENT_MEAN_GIB': _bytes_to_gib(val(memory, 'GPU_KV_ACTUAL_PERSISTENT_BYTES', default='NOT_AVAILABLE')),
        'GPU_KV_PERSISTENT_PEAK_MEAN_GIB': _bytes_to_gib(val(memory, 'GPU_KV_PERSISTENT_PEAK_BYTES', default='NOT_AVAILABLE')),
        'GPU_KV_PERSISTENT_PEAK_MAX_GIB': _bytes_to_gib(val(memory, 'GPU_KV_PERSISTENT_PEAK_BYTES', default='NOT_AVAILABLE')),
        'DRAFT_GPU_PERSISTENT_GIB': _bytes_to_gib(val(memory, 'GPU_DRAFT_PERSISTENT_BYTES', default=memory_out.get('DRAFT_GPU_GIB', 'NOT_AVAILABLE'))),
        'GPU_METHOD_PERSISTENT_MEAN_GIB': _bytes_to_gib(val(memory, 'GPU_METHOD_PERSISTENT_BYTES', default='NOT_AVAILABLE')),
        'GPU_METHOD_PERSISTENT_PEAK_GIB': _bytes_to_gib(val(memory, 'GPU_METHOD_PERSISTENT_PEAK_BYTES', default='NOT_AVAILABLE')),
        'TRANSIENT_DEQUANT_GPU_PEAK_GIB': _bytes_to_gib(val(memory, 'TRANSIENT_DEQUANT_GPU_PEAK_BYTES', default='NOT_AVAILABLE')),
        'TRANSIENT_PROMOTION_GPU_PEAK_GIB': _bytes_to_gib(val(memory, 'TRANSIENT_PROMOTION_GPU_PEAK_BYTES', default='NOT_AVAILABLE')),
        'TRANSIENT_HISTORICAL_GPU_PEAK_GIB': _bytes_to_gib(val(memory, 'TRANSIENT_HISTORICAL_GPU_PEAK_BYTES', default='NOT_AVAILABLE')),
        'KV_COMPRESSION_AT_PERSISTENT_PEAK': val(memory, 'KV_COMPRESSION_AT_PERSISTENT_PEAK', default='NOT_AVAILABLE'),
    }
    manifest = cfg.get('manifest_path')
    manifest_hash = cfg.get('manifest_sha256')
    if manifest and Path(manifest).exists(): manifest_hash = sha256(manifest)
    provenance = {
        'experiment_id': cfg.get('experiment_id'), 'method': cfg.get('method'), 'group': cfg.get('group', 'NOT_AVAILABLE'),
        'window': cfg.get('window'), 'case_protocol': cfg.get('case_protocol', 'NOT_AVAILABLE'),
        'quality_source': cfg.get('quality_source', 'NOT_AVAILABLE'), 'latency_source': cfg.get('latency_source', 'NOT_AVAILABLE'),
        'memory_source': cfg.get('memory_source', 'NOT_AVAILABLE'), 'manifest_path': manifest or 'NOT_AVAILABLE',
        'manifest_sha256': manifest_hash or 'NOT_AVAILABLE', 'prompt_case_id': cfg.get('case_id', 'NOT_AVAILABLE'),
        'seed': cfg.get('seed', 'NOT_AVAILABLE'), 'checkpoint': cfg.get('checkpoint', 'NOT_AVAILABLE'),
        'lora': cfg.get('lora', 'NOT_AVAILABLE'), 'git_commit': cfg.get('git_commit', 'NOT_AVAILABLE'),
        'gpu_model': cfg.get('gpu_model', 'NOT_AVAILABLE'), 'gpu_id': cfg.get('gpu_id', 'NOT_AVAILABLE'),
        'profiler_version': val(profile, 'profiler_version', default='NOT_AVAILABLE'),
        'timer_boundary_version': val(profile, 'timer_boundary_version', default='NOT_AVAILABLE'),
        'evaluator_version': cfg.get('evaluator_version', 'NOT_AVAILABLE'),
        'attention_call_count': val(profile, 'attention_calls', default='NOT_AVAILABLE'),
        'kv_storage_policy': cfg.get('kv_storage_policy', 'NOT_AVAILABLE'),
        'attention_precision': cfg.get('attention_precision', 'NOT_AVAILABLE'),
        'retrieval_strategy': cfg.get('retrieval_strategy', 'NOT_APPLICABLE'),
        'fetch_strategy': cfg.get('fetch_strategy', 'NOT_APPLICABLE'),
    }
    quality_valid = all(number(quality.get(k)) is not None for k in ('DINO', 'SSIM', 'PSNR'))
    latency_valid = all(number(latency.get(k)) is not None for k in ('E2E_INFERENCE_S', 'TRANSFORMER_S', 'SELF_ATTN_WRAPPER_S', 'ATTENTION_KERNEL_S'))
    memory_valid = number(memory_out.get('GPU_KV_GIB')) is not None and number(memory_out.get('CPU_KV_GIB')) is not None
    provenance_valid = provenance['manifest_sha256'] != 'NOT_AVAILABLE' and provenance['git_commit'] != 'NOT_AVAILABLE'
    validity = {'QUALITY_VALID': quality_valid, 'MEMORY_VALID': memory_valid, 'LATENCY_VALID': latency_valid,
                'PROVENANCE_VALID': provenance_valid, 'INFERENCE_VALID': bool(cfg.get('inference_valid', True)),
                'FINAL_ROW_VALID': all((quality_valid, memory_valid, latency_valid, provenance_valid, bool(cfg.get('inference_valid', True))))}
    return {'experiment': {'id': cfg.get('experiment_id'), 'mode': cfg.get('_mode')}, 'config': cfg,
            'quality': quality, 'memory': memory_out, 'latency': latency,
            'rag_strategy': cfg.get('rag_strategy', {}), 'provenance': provenance, 'validity': validity}

def run(cfg_path, mode):
    cfg = load_json(cfg_path); cfg['_mode'] = mode
    if mode == 'full' and cfg.get('inference_command'):
        subprocess.run(cfg['inference_command'], shell=True, check=True)
    elif mode == 'full' and not cfg.get('inference_command'):
        raise SystemExit('full mode requires explicit inference_command; no inference was launched')
    summary = build_summary(cfg)
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    (out / 'evaluation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    write_csv(out / 'evaluation_summary.csv', [as_csv_row(summary)])
    (out / 'evaluation_report.md').write_text(report(summary))
    print(json.dumps({'output_dir': str(out), 'validity': summary['validity']}, indent=2))
    return summary

def report(s):
    p, v = s['provenance'], s['validity']; q, m, l = s['quality'], s['memory'], s['latency']
    return f'''# Unified evaluation: {p.get("experiment_id")}\n\n| Section | Source | Valid |\n|---|---|---|\n| Quality | {p.get("quality_source")} | {v["QUALITY_VALID"]} |\n| Latency | {p.get("latency_source")} | {v["LATENCY_VALID"]} |\n| Memory | {p.get("memory_source")} | {v["MEMORY_VALID"]} |\n| Provenance | {p.get("manifest_sha256")} | {v["PROVENANCE_VALID"]} |\n| Final row | mixed-source aggregate | {v["FINAL_ROW_VALID"]} |\n\n## Metrics\n\n- DINO/SSIM/PSNR/LPIPS: {q}\n- Latency: {l}\n- Memory: {m}\n\n`NON_TRANSFORMER_E2E_S` is derived as `E2E_INFERENCE_S - TRANSFORMER_S`; it is not Self-Attention Wrapper time.\n'''

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--mode', choices=['full','reuse'], required=True)
    ap.add_argument('--quality-only', action='store_true'); ap.add_argument('--profile-only', action='store_true')
    a = ap.parse_args(); run(a.config, a.mode)
if __name__ == '__main__': main()
