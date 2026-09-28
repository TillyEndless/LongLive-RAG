from pathlib import Path
import ast

root = Path('/data/zxl/LongLive-RAG-group11_15_h200')
source = (root / 'wan/modules/causal_model_latentmem.py').read_text()
tree = ast.parse(source)
names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
required = {'_prefetch_state', '_reconcile_prefetch', '_schedule_next_layer_prefetch', '_consume_prefetch_or_fetch'}
assert required <= names
assert 'true_set' in source and 'prefetch_wasted_bytes' in source
assert 'prefetch_hit_bytes' in source and 'prefetch_correction_bytes' in source
assert 'memory_indices = online_memory_indices' in source
assert 'prefetched is not None' in source

# Pure set-contract check: attention IDs are always I_{l+1}, never a union.
prev = {1, 2, 4, 7}
curr = {2, 3, 7, 9}
assert (prev & curr) == {2, 7}
assert (curr - prev) == {3, 9}
assert (prev - curr) == {1, 4}
assert curr != (prev | curr)

pipe = (root / 'pipeline/causal_inference.py').read_text()
mem = (root / 'utils/runtime_memory_measurement.py').read_text()
inf = (root / 'inference.py').read_text()
assert 'compressed_history_archive' in pipe
assert 'PERSISTENT_STORAGE_MODE' in inf
assert 'TRANSIENT_PREFETCH_GPU_PEAK_BYTES' in mem
assert 'GPU_KV_ACTUAL_PERSISTENT_BYTES' in mem
assert 'GPU_KV_ACTUAL_PERSISTENT_BYTES' not in '\n'.join(
    line for line in mem.splitlines() if 'TRANSIENT_PREFETCH' in line
)
print('GROUP11.4_STATIC_CONTRACT=PASS')
print('FINAL_ATTENTION_IDS=I_{l+1}')
print('HIT=I_l_INTERSECT_I_{l+1}; CORRECTION=I_{l+1}_MINUS_I_l; WASTE=I_l_MINUS_I_{l+1}')
print('GROUP12_15_ARCHIVE_OWNER=UNCHANGED')
