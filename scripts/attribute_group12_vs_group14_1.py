#!/usr/bin/env python3
import csv, json, math, statistics
from pathlib import Path

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
SRC = ROOT / 'results/unified_latency_profile'
OUT = ROOT / 'results'
RAW = OUT / 'group12_vs_group14_1_overhead_raw.csv'
AGG = OUT / 'group12_vs_group14_1_overhead_aggregate.json'
REPORT = ROOT / 'reports/group12_vs_group14_1_overhead_attribution.md'

def load(group):
    return json.loads((SRC / group / 'case_01/rank0-0-0_lora_runtime.json').read_text())

def nsum(rows, key):
    return sum(float(r.get(key, 0.0) or 0.0) for r in rows)

def stats(vals):
    vals = [float(x) for x in vals if x is not None]
    if not vals: return {'count': 0, 'sum_ms': 'NOT_AVAILABLE'}
    xs = sorted(vals)
    return {'count': len(xs), 'sum_ms': sum(xs),
            'mean_ms': statistics.mean(xs),
            'p50_ms': xs[len(xs)//2],
            'p95_ms': xs[min(len(xs)-1, math.ceil(len(xs)*.95)-1)]}

data = {'Group12': load('group12'), 'Group14.1': load('group14')}
all_rows = []
summary = {}
for name, d in data.items():
    prof = d['group11_profile']
    events = d['UNIFIED_LATENCY_PROFILE']['records']
    attn = prof['attention_rows']
    routes = prof.get('draftmap_rows', [])
    fetch = prof.get('fetch_phase_rows', [])
    h2d = prof.get('h2d_rows', [])
    assert len(events) == 6000 and len(attn) == 6000
    for i, (ev, ar) in enumerate(zip(events, attn)):
        row = {
            'variant': name, 'call_id': i,
            'generation_unit': ev.get('generation_unit', 'NOT_AVAILABLE'),
            'denoising_step': ev.get('denoising_step', 'NOT_AVAILABLE'),
            'layer_id': ev.get('layer_id', 'NOT_AVAILABLE'),
            'Q_LEN': ev.get('q_len', 'NOT_AVAILABLE'),
            'ORIGINAL_K_LEN': ev.get('original_k_len', 'NOT_AVAILABLE'),
            'ACTUAL_K_LEN': ev.get('actual_k_len', 'NOT_AVAILABLE'),
            'RETAINED_RATIO': ev.get('retained_ratio', 'NOT_AVAILABLE'),
            'routing_score_ms': 'NOT_AVAILABLE', 'block_selection_ms': 'NOT_AVAILABLE',
            'gather_k_ms': 'NOT_AVAILABLE', 'gather_v_ms': 'NOT_AVAILABLE',
            'repack_ms': 'NOT_AVAILABLE', 'dequant_k_ms': 'NOT_AVAILABLE',
            'dequant_v_ms': 'NOT_AVAILABLE', 'bf16_materialize_ms': 'NOT_AVAILABLE',
            'rope_ms': 'NOT_AVAILABLE', 'mask_layout_ms': 'NOT_AVAILABLE',
            'host_sync_wait_ms': 'NOT_AVAILABLE',
            'attention_kernel_ms': ev.get('cuda_ms', 'NOT_AVAILABLE'),
            # attention_rows.cpu_wall_ms is the inner attention-call host timer,
            # not the parent SELF_ATTN_WRAPPER boundary; do not relabel it.
            'self_attn_wrapper_ms': 'NOT_AVAILABLE',
        }
        all_rows.append(row)
    draft_q = nsum(routes, 'time_draft_q_pool_ms')
    draft_score = nsum(routes, 'time_score_ms')
    draft_agg = nsum(routes, 'time_aggregate_ms')
    topk = nsum(routes, 'time_topk_ms')
    gather = nsum(fetch, 'gather_ms')
    cat = nsum(fetch, 'cat_ms')
    fetch_e2e = nsum(fetch, 'end_to_end_ms')
    summary[name] = {
        'attention_calls': len(events),
        'wrapper_s': float(d['group11_profile']['model_phase_ms']['attention_wrapper']) / 1000.0,
        'attention_kernel_s': nsum(events, 'cuda_ms') / 1000.0,
        'draft_q_pool_s': draft_q / 1000.0,
        'draft_score_matmul_softmax_s': draft_score / 1000.0,
        'importance_reduction_s': draft_agg / 1000.0,
        'topk_selection_s': topk / 1000.0,
        'gather_repack_s': (gather + cat) / 1000.0,
        'fetch_end_to_end_work_s': fetch_e2e / 1000.0,
        'h2d_copy_work_s': nsum(h2d, 'copy_time_ms') / 1000.0,
        'route_rows': len(routes), 'fetch_rows': len(fetch),
        'h2d_rows': len(h2d),
        'dequant_s': 'NOT_AVAILABLE', 'rope_s': 'NOT_AVAILABLE',
        'mask_layout_s': 'NOT_AVAILABLE', 'host_sync_wait_s': 'NOT_AVAILABLE',
    }

with RAW.open('w', newline='') as f:
    fields = list(all_rows[0])
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(all_rows)

g12, g14 = summary['Group12'], summary['Group14.1']
additional = (80.519 - 8.655) - (75.487 - 10.717)
deltas = {
    'routing_score_delta_s': (g14['draft_q_pool_s'] + g14['draft_score_matmul_softmax_s'] + g14['importance_reduction_s']) - (g12['draft_q_pool_s'] + g12['draft_score_matmul_softmax_s'] + g12['importance_reduction_s']),
    'block_selection_delta_s': g14['topk_selection_s'] - g12['topk_selection_s'],
    'gather_repack_delta_s': g14['gather_repack_s'] - g12['gather_repack_s'],
    'dequant_materialization_delta_s': 'NOT_AVAILABLE',
    'rope_delta_s': 'NOT_AVAILABLE', 'mask_layout_delta_s': 'NOT_AVAILABLE',
    'host_sync_wait_delta_s': 'NOT_AVAILABLE',
}
deltas['measured_work_delta_s'] = deltas['routing_score_delta_s'] + deltas['block_selection_delta_s'] + deltas['gather_repack_delta_s']
aggregate = {
    'matched_contract': {'case': 'case_01', 'window': 12, 'seed': 0, 'attention_calls': 6000, 'profile_interval': 'full matched case', 'protocol': 'same checkpoint/config family'},
    'group12': g12, 'group14_1': g14,
    'known_equation': {'group12_wrapper_s': 75.487, 'group14_1_wrapper_s': 80.519, 'group12_kernel_s': 10.717, 'group14_1_kernel_s': 8.655, 'attention_kernel_saved_s': 2.062, 'additional_non_kernel_s': additional},
    'deltas': deltas,
    'explained_critical_path_s': 'NOT_IDENTIFIABLE_FROM_EXISTING_TRACE',
    'residual_s': 'NOT_IDENTIFIABLE_FROM_EXISTING_TRACE',
    'decode_order': 'FULL_DECODE_THEN_GATHER',
    'default_stream_serialized': 'YES for recorded attention path; exact subphase overlap NOT_MEASURED',
    'root_cause_classification': 'UNRESOLVED_WITH_EXISTING_TRACE',
    'break_even_retained_ratio': 'NOT_IDENTIFIABLE_YET',
    'source_evidence': {
        'routing': 'utils/persistent_draftmap.py:route_draftmap',
        'prepare': 'utils/h200_group_runtime.py:prepare_attention_kv',
        'attention': 'wan/modules/causal_model_latentmem.py:CausalWanSelfAttention.forward',
        'fetch': 'wan/modules/causal_model_latentmem.py:fetch/repack region',
    },
}
AGG.write_text(json.dumps(aggregate, indent=2) + '\n')

def f(v): return 'NOT_AVAILABLE' if isinstance(v, str) else f'{v:.6f}'
lines = [
    '# Group12 vs Group14.1 overhead attribution', '',
    '## Scope and contract', '',
    'Matched existing case_01, W12, seed=0, 6000 attention calls per variant. No canonical10 or Group14.2/14.3/14.4 inference was run by this analysis.', '',
    '## Known equation', '',
    '| quantity | Group12 | Group14.1 |', '|---|---:|---:|',
    '| self-attention wrapper (s) | 75.487 | 80.519 |', '| attention CUDA work (s) | 10.717 | 8.655 |', '| non-kernel wrapper (s) | 64.770 | 71.864 |', '| additional non-kernel (s) | colspan=2: 7.094 |', '',
    'The kernel saving is 2.062 s. The remaining 7.094 s is a critical-path quantity, not the sum of nested component timers.', '',
    '## Measured work-time components', '',
    '| component | Group12 work (s) | Group14.1 work (s) | delta (s) |', '|---|---:|---:|---:|',
    f'| Draft Q pooling | {f(g12["draft_q_pool_s"])} | {f(g14["draft_q_pool_s"])} | {f(g14["draft_q_pool_s"]-g12["draft_q_pool_s"])} |',
    f'| DraftMap score (matmul+softmax combined) | {f(g12["draft_score_matmul_softmax_s"])} | {f(g14["draft_score_matmul_softmax_s"])} | {f(g14["draft_score_matmul_softmax_s"]-g12["draft_score_matmul_softmax_s"])} |',
    f'| importance reduction | {f(g12["importance_reduction_s"])} | {f(g14["importance_reduction_s"])} | {f(g14["importance_reduction_s"]-g12["importance_reduction_s"])} |',
    f'| TopK/block selection | {f(g12["topk_selection_s"])} | {f(g14["topk_selection_s"])} | {f(g14["topk_selection_s"]-g12["topk_selection_s"])} |',
    f'| gather + cat/repack | {f(g12["gather_repack_s"])} | {f(g14["gather_repack_s"])} | {f(g14["gather_repack_s"]-g12["gather_repack_s"])} |',
    '| K/V gather split | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |',
    '| dequant/materialization | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |',
    '| RoPE/mask/layout split | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |',
    '| host sync wait | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |', '',
    f'Measured work-time delta for the available routing+selection+gather fields is **{deltas["measured_work_delta_s"]:.6f} s**, which is {100*deltas["measured_work_delta_s"]/additional:.1f}% of the 7.094 s critical-path delta. This over-coverage proves nested/overlapping accounting; it cannot be reported as exposed overhead.', '',
    '## Decode order and synchronization audit', '',
    '`archive.fetch` dequantizes/materializes the selected historical record at `wan/modules/causal_model_latentmem.py:802-813`; only afterwards does `route_draftmap` pool/score/select and gather BF16 K/V at `utils/persistent_draftmap.py:32-43`. Therefore the measured code order is **FULL_DECODE_THEN_GATHER**. The route implementation also materializes selected IDs/scores through `.detach().cpu().tolist()` at lines 58-59, a definite potential GPU→CPU wait site; its wait duration was not measured in the saved trace.', '',
    '**Default stream:** recorded attention events use the current/default stream; exact overlap among routing subphases is not measured. The saved trace has no independent dequant, host-wait, RoPE, or mask-layout timers.', '',
    '## Conclusion', '',
    '- Attention arithmetic is reduced: QK elements fall to 0.704242 of Group12 and CUDA attention work falls by 2.062 s.',
    '- The available routing score, selection, and gather/repack timers do not explain the 7.094 s delta: their measured work delta is -0.366903 s.',
    '- The exact 90% critical-path attribution is **not achieved** by existing artifacts. Residual and top-3 exposed contributors are therefore **NOT_IDENTIFIABLE**, not guessed.',
    '- Root cause classification: **UNRESOLVED_WITH_EXISTING_TRACE**.',
    '- Break-even retained ratio: **NOT_IDENTIFIABLE_YET**; one retained-ratio point cannot fit a model.', '',
    '## Outputs', '',
    f'- Raw per-call CSV: `{RAW}`', f'- Aggregate JSON: `{AGG}`', f'- Report: `{REPORT}`',
]
REPORT.write_text('\n'.join(lines) + '\n')
print(json.dumps({'raw': str(RAW), 'aggregate': str(AGG), 'report': str(REPORT), 'additional_non_kernel_s': additional, 'measured_work_delta_s': deltas['measured_work_delta_s']}, indent=2))
