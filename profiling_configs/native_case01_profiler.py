#!/usr/bin/env python3
import csv, json, os, runpy, sys, time
from pathlib import Path
import torch
import wan.modules.causal_model as cm

events, forward_rows = [], []
layer_counter = {'n': 0}
old_init = cm.CausalWanSelfAttention.__init__
def profiled_init(self, *args, **kwargs):
    old_init(self, *args, **kwargs)
    self._native_profile_layer = layer_counter['n']; layer_counter['n'] += 1
cm.CausalWanSelfAttention.__init__ = profiled_init
old_forward = cm.CausalWanSelfAttention.forward
def profiled_forward(self, *args, **kwargs):
    t = time.perf_counter(); out = old_forward(self, *args, **kwargs)
    forward_rows.append({'layer': int(getattr(self, '_native_profile_layer', -1)), 'host_s': time.perf_counter() - t})
    return out
cm.CausalWanSelfAttention.forward = profiled_forward
old_flex = cm.flex_attention
def profiled_flex(*args, **kwargs):
    a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
    a.record(torch.cuda.current_stream()); out = old_flex(*args, **kwargs); b.record(torch.cuda.current_stream())
    events.append((a, b)); return out
cm.flex_attention = profiled_flex

config = os.environ['PROFILE_CONFIG']; out = Path(os.environ['PROFILE_OUT']); out.mkdir(parents=True, exist_ok=True)
sys.argv = ['inference.py', '--config_path', config]
os.chdir(os.environ['PROFILE_REPO'])
runpy.run_path(str(Path(os.environ['PROFILE_REPO']) / 'inference.py'), run_name='__main__')
torch.cuda.synchronize()
kernel_ms = [float(a.elapsed_time(b)) for a, b in events]
runtime_files = sorted(out.glob('*_runtime.json')); memory_files = sorted(out.glob('*_memory_measurement.json'))
runtime = json.loads(runtime_files[0].read_text()) if runtime_files else {}
memory = json.loads(memory_files[0].read_text()) if memory_files else {}
summary = {
    'method': 'Native local-window baseline', 'window': int(os.environ['PROFILE_WINDOW']), 'case': 'case_01',
    'gpu_visible': os.environ.get('CUDA_VISIBLE_DEVICES', 'unset'), 'git_commit': os.environ.get('PROFILE_GIT_COMMIT', 'NOT_RECORDED'),
    'profiler_version': 'native_case01_profiler_v1 + existing unified E2E/memory contract',
    'timer_boundary_version': 'Group11 matched case01: E2E / transformer / self-attention CUDA flex kernel',
    'attention_calls': len(forward_rows), 'attention_kernel_calls': len(kernel_ms),
    'attention_kernel_s': sum(kernel_ms) / 1000.0, 'attention_kernel_ms_per_call': sum(kernel_ms) / len(kernel_ms) if kernel_ms else 'NA',
    'self_attention_wrapper_s': sum(x['host_s'] for x in forward_rows),
    'e2e_s': runtime.get('E2E_LATENCY_S', runtime.get('e2e_latency_s', 'NA')), 'transformer_s': runtime.get('TRANSFORMER_LATENCY_S', runtime.get('transformer_latency_s', 'NA')),
    'process_wall_s': runtime.get('PROCESS_WALL_S', 'NA'), 'vae_decode_s': runtime.get('VAE_DECODE_S', 'NA'), 'video_encode_s': runtime.get('VIDEO_ENCODE_S', 'NA'),
    'fetch_work_s': 0.0, 'fetch_exposed_s': 0.0, 'fetch_hidden_s': 0.0,
    'qkv_projection_s': 'NOT_AVAILABLE', 'cross_attention_s': 'NOT_AVAILABLE', 'ffn_mlp_s': 'NOT_AVAILABLE', 'norm_residual_s': 'NOT_AVAILABLE', 'transformer_other_s': 'NOT_AVAILABLE', 'rope_s': 'NOT_AVAILABLE', 'mask_layout_s': 'NOT_AVAILABLE',
    'gpu_kv_gib': memory.get('GPU_KV_MEASURED_GiB', runtime.get('GPU_KV_MEASURED_GiB', 'NA')), 'cpu_kv_gib': memory.get('CPU_KV_MEASURED_GiB', runtime.get('CPU_KV_MEASURED_GiB', 0.0)),
    'draft_gpu_gib': memory.get('GPU_DRAFT_PERSISTENT_GiB', runtime.get('GPU_DRAFT_PERSISTENT_GiB', 0.0)), 'transient_fetch_gpu_gib': memory.get('TRANSIENT_FETCH_GPU_GiB', runtime.get('TRANSIENT_FETCH_GPU_GiB', 0.0)),
    'peak_gpu_allocated_gib': memory.get('PEAK_GPU_ALLOCATED_GiB', runtime.get('PEAK_GPU_ALLOCATED_GiB', runtime.get('peak_allocated_gib', 'NA'))), 'peak_gpu_reserved_gib': memory.get('PEAK_GPU_RESERVED_GiB', runtime.get('PEAK_GPU_RESERVED_GiB', runtime.get('peak_reserved_gib', 'NA'))),
    'runtime_metadata': str(runtime_files[0]) if runtime_files else 'NOT_FOUND', 'memory_metadata': str(memory_files[0]) if memory_files else 'NOT_FOUND',
    'validity': 'PASS' if len(forward_rows) == 6000 and len(kernel_ms) == 6000 else 'FAIL_CALL_COUNT',
}
(out / 'native_case01_profile.json').write_text(json.dumps(summary, indent=2))
with (out / 'attention_kernel_events.csv').open('w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['call_id', 'cuda_ms']); w.writeheader(); w.writerows({'call_id': i, 'cuda_ms': x} for i, x in enumerate(kernel_ms))
print(json.dumps(summary))
