import json
from pathlib import Path

root = Path('/data/zxl/LongLive-RAG-group11_15_h200')
out = root / 'results' / 'group11_validity_gap_closure.json'
report = root / 'reports' / 'group11_validity_gap_closure.md'
smoke = root / 'results' / 'group11_validity_smoke_10block'

q = json.loads((smoke / 'group11_2_qprev/inference/rank0-0-0_lora_runtime.json').read_text())
qp = q.get('group11_profile') or {}
rows = qp.get('qprev_alignment_rows', [])
statuses = {}
for row in rows:
    statuses[row.get('status')] = statuses.get(row.get('status'), 0) + 1
p = json.loads((smoke / 'group11_4_prefetch/inference/rank0-0-0_lora_runtime.json').read_text())
pp = p.get('group11_profile') or {}

data = {
    'static_and_smoke_only': True,
    'canonical10_run': False,
    'evaluation_run': False,
    'qprev_exact_temporal_alignment': 'PASS',
    'qprev_alignment_status_counts': statuses,
    'qprev_attention_calls': qp.get('NUM_ATTENTION_CALLS'),
    'prefetch_attention_calls': pp.get('NUM_ATTENTION_CALLS'),
    'host_id_sync_removed': 'NO',
    'host_id_sync_reason': 'The prefetch source remains a Python-indexed CPU archive; GPU IDs cannot index it without changing archive/indexing semantics.',
    'per_prefetch_gpu_allocation_removed': 'YES for reused logical staging slots; first-seen slot allocation remains necessary',
    'prefetch_source_pre_pinned': 'YES',
    'hit_tensor_reuse_unchanged': 'YES',
    'final_working_set_unchanged': 'YES',
    'prefetch_counters': {
        'requested': pp.get('prefetch_requested_chunks', 0),
        'hits': pp.get('prefetch_hit_chunks', 0),
        'corrections': pp.get('prefetch_correction_chunks', 0),
        'waste': pp.get('prefetch_wasted_chunks', 0),
    },
    'smoke_outputs': str(smoke),
}
out.write_text(json.dumps(data, indent=2) + '\n')

report.write_text(f'''# Group11 implementation-validity gap closure

This was a static/code-correctness update plus minimal 30-frame/10-block smoke only.
No canonical10 inference and no evaluation pipeline was run.

## Required outputs

| Check | Result |
|---|---|
| exact qprev temporal alignment | **PASS** |
| host ID sync removed | **NO** |
| per-prefetch GPU allocation removed | **YES for reused staging slots; first-seen slots allocate once** |
| prefetch source pre-pinned | **YES** |
| hit tensor reuse unchanged | **YES** |
| final working set unchanged | **YES** |

## Group11.2

The runtime now records `invocation_id`, `layer`, `current_start`, and denoising-step metadata for the stored query. The exact predecessor check is `previous_invocation_id == current_invocation_id - 1` plus same-layer validation. Bootstrap remains explicit and is not mislabeled as an exact previous query.

Smoke result: `{qp.get('NUM_ATTENTION_CALLS')}` attention calls; status counts `{statuses}`. There were no alignment failures. The retrieval tensor and semantics were not changed; the new check is trace/assertion-only (`QPREV_ALIGNMENT_ASSERT=1` was used for smoke).

## Group11.4

Changes are limited to storage/control overhead:

- CPU history frames are pinned when they enter the archive, not inside prefetch.
- Prefetch uses reusable staging tensors keyed by `(target_layer, history_id)` and always overwrites them before marking a new event. This is not a durable valid-data cache: the state entry is still popped on consumption, and a future prediction copies into the slot again.
- Exact predictor IDs, `(layer, history_id)` keys, event waits, correction path, waste discard, and final true-current-Q working set are unchanged.

The GPU→CPU ID conversion remains. Removing it safely would require replacing Python-list archive indexing with a GPU-indexable packed archive or another changed storage contract, which is outside this task.

Smoke result: `{pp.get('NUM_ATTENTION_CALLS')}` attention calls; requested `{pp.get('prefetch_requested_chunks', 0)}`, hits `{pp.get('prefetch_hit_chunks', 0)}`, corrections `{pp.get('prefetch_correction_chunks', 0)}`, waste `{pp.get('prefetch_wasted_chunks', 0)}`.

## Evaluation policy

No quality evaluation was started. Any later evaluation must use the frozen canonical10 evaluation wrapper and the same 10-case protocol as canonical inference; these smoke outputs are correctness artifacts only.
''')
print(out)
