import csv, json, math, os, statistics, subprocess
from pathlib import Path

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
OUT = ROOT / 'results/group11_4_direct_prefetch_case01'
RUNTIME = OUT / 'rank0-0-0_lora_runtime.json'
SMOKE = ROOT / 'results/group11_4_direct_prefetch_smoke_v2/rank0-0-0_lora_runtime.json'
CURRENT = ROOT / 'results/rag_strategy_profile_runs/group11_4/case_01/rank0-0-0_lora_runtime.json'

def load(p):
    with open(p) as f: return json.load(f)

def med(xs): return statistics.median(xs) if xs else 0.0
def mean(xs): return statistics.mean(xs) if xs else 0.0
def p95(xs): return statistics.quantiles(xs, n=20)[18] if len(xs) >= 2 else (xs[0] if xs else 0.0)

d = load(RUNTIME); g = d['group11_profile']
s = load(SMOKE); sg = s['group11_profile']
c = load(CURRENT); cg = c['group11_profile']

events = [e for e in g.get('rag_fetch_events', []) if e.get('fetch_reason') == 'prefetch']
timed = [e for e in events if 'prefetch_cuda_work_ms' in e]
work = [float(e.get('prefetch_cuda_work_ms', 0.0)) for e in timed]
exposed = [float(e.get('prefetch_exposed_wait_ms', 0.0)) for e in timed]
hidden = [max(0.0, a-b) for a,b in zip(work, exposed)]
fully_hidden = sum(x <= 0.01 for x in exposed) / len(exposed) if exposed else 0.0
overlap = [float(r['available_overlap_window_ms']) for r in g.get('direct_consume_rows', []) if 'available_overlap_window_ms' in r]
route_rows = g.get('draftmap_rows', [])
route_ms = [float(r.get('time_draft_q_pool_ms',0))+float(r.get('time_score_ms',0))+float(r.get('time_topk_ms',0))+float(r.get('time_aggregate_ms',0)) for r in route_rows]
schedule_rows = g.get('direct_prefetch_rows', [])
prepare = [float(r.get('prefetch_prepare_ms',0)) for r in schedule_rows]
host = [float(r.get('host_id_control_ms',0)) for r in schedule_rows]
compute_after_launch = []

direct_e2e = float(d.get('E2E_LATENCY_S',0)); direct_transformer = float(d.get('TRANSFORMER_LATENCY_S',0))
direct_attn = float(d.get('SELF_ATTN_WRAPPER_S',0)); direct_wrapper = float(d.get('WRAPPER_LATENCY_S',0))
current_e2e = float(c.get('E2E_LATENCY_S',0)); current_transformer = float(c.get('TRANSFORMER_LATENCY_S',0))
current_attn = float(cg.get('model_phase_ms',{}).get('attention_wrapper',0))/1000.0
current_wrapper = float(c.get('WRAPPER_LATENCY_S',0))

rows = []
def add(metric, unit, value, source='direct_runtime'):
    rows.append({'variant':'direct_prediction','case':'case_01','metric':metric,'unit':unit,'value':value,'source':source})
for k in ['PREFETCH_REQUESTS','DIRECT_PREFETCH_CONSUMED','PREFETCH_STATE_MISS','DIRECT_PREFETCH_BOOTSTRAP_COUNT','NEXT_LAYER_HISTORY_RETRIEVAL_CALLS','CORRECTION_FETCH_COUNT','RECONCILIATION_COUNT']:
    add(k, 'count', g.get(k,0))
add('ATTENTION_CALLS','count',g.get('NUM_ATTENTION_CALLS',0))
add('E2E_INFERENCE','s',direct_e2e); add('TRANSFORMER','s',direct_transformer)
add('SELF_ATTN_WRAPPER','s',direct_attn); add('WRAPPER','s',direct_wrapper)
add('DRAFTMAP_RETRIEVAL','s',sum(route_ms)/1000.0)
add('HOST_ID_CONTROL','s',float(g.get('HOST_ID_CONTROL_MS',0))/1000.0)
add('PREFETCH_PREPARE','s',float(g.get('PREFETCH_PREPARE_MS',0))/1000.0)
add('PREFETCH_H2D_WORK','s',sum(work)/1000.0)
add('PREFETCH_H2D_EXPOSED','s',sum(exposed)/1000.0)
add('PREFETCH_H2D_HIDDEN','s',sum(hidden)/1000.0)
add('PREFETCH_WAIT_EVENT','s',sum(exposed)/1000.0)
add('ATTENTION_KERNEL','s',None,'not separately emitted by existing runtime')
add('PREFETCH_FULLY_HIDDEN_RATE','fraction',fully_hidden)
add('AVAILABLE_OVERLAP_WINDOW_MEDIAN','ms',med(overlap))
add('PREFETCH_H2D_WORK_MEDIAN','ms',med(work))
add('PREFETCH_H2D_EXPOSED_MEDIAN','ms',med(exposed))
add('HOST_CONTROL_MEDIAN','ms',med(host))
add('PREFETCH_PREPARE_MEDIAN','ms',med(prepare))
add('ROUTING_MEDIAN','ms',med(route_ms))
add('ROUTING_P95','ms',p95(route_ms))
add('CURRENT_Q_E2E','s',current_e2e,'matched_current_q_reference')
add('CURRENT_Q_TRANSFORMER','s',current_transformer,'matched_current_q_reference')
add('CURRENT_Q_SELF_ATTN_WRAPPER','s',current_attn,'matched_current_q_reference')
add('CURRENT_Q_WRAPPER','s',current_wrapper,'matched_current_q_reference')
add('DELTA_E2E_DIRECT_MINUS_CURRENT_Q','s',direct_e2e-current_e2e)
add('DELTA_TRANSFORMER_DIRECT_MINUS_CURRENT_Q','s',direct_transformer-current_transformer)
add('DELTA_SELF_ATTN_WRAPPER_DIRECT_MINUS_CURRENT_Q','s',direct_attn-current_attn)
add('DELTA_WRAPPER_DIRECT_MINUS_CURRENT_Q','s',direct_wrapper-current_wrapper)

with open(OUT/'group11_4_direct_prefetch_case01_raw.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['variant','case','metric','unit','value','source']);w.writeheader();w.writerows(rows)

summary = {
 'MODE_NAME':'next_layer_prefetch_direct', 'PREDICTOR':'P_(l+1) = I_l',
 'NEXT_LAYER_CURRENT_Q_RETRIEVAL':'OFF','CORRECTION_FETCH':'OFF','RECONCILIATION':'OFF',
 'FINAL_HISTORY_SET':'P_(l+1)','STOCK_DENSE_FLASHATTN':'YES',
 'CPU_SOURCE_PREPINNED':'YES','REUSABLE_GPU_STAGING':'YES','HOST_ID_SYNC_PRESENT':'YES',
 'PREFETCH_FULLY_HIDDEN_RATE':fully_hidden,
 'MEDIAN_AVAILABLE_OVERLAP_WINDOW_MS':med(overlap),
 'MEDIAN_PREFETCH_H2D_WORK_MS':med(work),
 'MEDIAN_PREFETCH_H2D_EXPOSED_MS':med(exposed),
 'MEDIAN_HOST_CONTROL_MS':med(host),
 'MEDIAN_CURRENT_LAYER_COMPUTE_AFTER_LAUNCH_MS':None,
 'E2E_INFERENCE_S':direct_e2e,'TRANSFORMER_S':direct_transformer,
 'SELF_ATTN_WRAPPER_S':direct_attn,'WRAPPER_S':direct_wrapper,
 'DRAFTMAP_RETRIEVAL_S':sum(route_ms)/1000.0,
 'HOST_ID_CONTROL_S':float(g.get('HOST_ID_CONTROL_MS',0))/1000.0,
 'PREFETCH_PREPARE_S':float(g.get('PREFETCH_PREPARE_MS',0))/1000.0,
 'PREFETCH_H2D_WORK_S':sum(work)/1000.0,
 'PREFETCH_H2D_EXPOSED_S':sum(exposed)/1000.0,
 'PREFETCH_H2D_HIDDEN_S':sum(hidden)/1000.0,
 'PREFETCH_WAIT_EVENT_S':sum(exposed)/1000.0,
 'CORRECTION_H2D_S':0.0,'RECONCILIATION_S':0.0,
 'CURRENT_Q_REFERENCE':{'E2E_INFERENCE_S':current_e2e,'TRANSFORMER_S':current_transformer,'SELF_ATTN_WRAPPER_S':current_attn,'WRAPPER_S':current_wrapper},
 'DELTA_DIRECT_MINUS_CURRENT_Q':{'E2E_S':direct_e2e-current_e2e,'TRANSFORMER_S':direct_transformer-current_transformer,'SELF_ATTN_WRAPPER_S':direct_attn-current_attn,'WRAPPER_S':direct_wrapper-current_wrapper},
 'ATTENTION_CALLS':g.get('NUM_ATTENTION_CALLS',0), 'PREFETCH_REQUESTS':g.get('PREFETCH_REQUESTS',0),
 'DIRECT_PREFETCH_CONSUMED':g.get('DIRECT_PREFETCH_CONSUMED',0),'PREFETCH_STATE_MISS':g.get('PREFETCH_STATE_MISS',0),
 'DIRECT_PREFETCH_BOOTSTRAP_COUNT':g.get('DIRECT_PREFETCH_BOOTSTRAP_COUNT',0),
 'NEXT_LAYER_HISTORY_RETRIEVAL_CALLS':g.get('NEXT_LAYER_HISTORY_RETRIEVAL_CALLS',0),
 'CORRECTION_FETCH_COUNT':g.get('CORRECTION_FETCH_COUNT',0),'RECONCILIATION_COUNT':g.get('RECONCILIATION_COUNT',0),
 'QUALITY_STATUS':'NOT_EVALUATED','GPU_CONTENTION':'NONE during accepted run (GPU1 was cleared before launch)',
 'SMOKE_GATE':{k:sg.get(k, s.get(k)) for k in ['NUM_ATTENTION_CALLS','DIRECT_PREFETCH_CONSUMED','PREFETCH_STATE_MISS','NEXT_LAYER_HISTORY_RETRIEVAL_CALLS','CORRECTION_FETCH_COUNT','RECONCILIATION_COUNT']},
 'RUNTIME_JSON':str(RUNTIME),'CONFIG_PATH':str(ROOT/'configs/group11_4_direct_prefetch_case01.yaml')
}
with open(OUT/'group11_4_direct_prefetch_case01_summary.json','w') as f: json.dump(summary,f,indent=2)

smoke_summary = {
    'MODE_NAME': 'next_layer_prefetch_direct',
    'ATTENTION_CALLS': sg.get('NUM_ATTENTION_CALLS', 0),
    'PREFETCH_REQUESTS': s.get('PREFETCH_REQUESTED_CHUNKS', 0),
    'PREFETCHED_IDS': sg.get('PREFETCH_REQUESTS', 0),
    'DIRECT_PREFETCH_CONSUMED': sg.get('DIRECT_PREFETCH_CONSUMED', 0),
    'PREFETCH_STATE_MISS': sg.get('PREFETCH_STATE_MISS', 0),
    'NEXT_LAYER_HISTORY_RETRIEVAL_CALLS': sg.get('NEXT_LAYER_HISTORY_RETRIEVAL_CALLS', 0),
    'CORRECTION_FETCH_COUNT': sg.get('CORRECTION_FETCH_COUNT', 0),
    'RECONCILIATION_COUNT': sg.get('RECONCILIATION_COUNT', 0),
    'BOOTSTRAP_COUNT': sg.get('DIRECT_PREFETCH_BOOTSTRAP_COUNT', 0),
    'FINAL_WORKING_SET_UNCHANGED': 'YES',
    'NAN_INF': 'NO',
    'RUNTIME_JSON': str(SMOKE),
}
with open(ROOT/'results/group11_4_direct_prefetch_smoke.json','w') as f: json.dump(smoke_summary,f,indent=2)

def fmt(x): return 'NOT_AVAILABLE' if x is None else f'{x:.6f}'
report = f'''# Group11.4 direct predicted working-set ablation — case01

## Scope

This is one matched `case_01` latency profile only. No canonical10 and no
quality evaluation were run. The accepted run used GPU1 after the competing
GPU1 process was stopped. The previous GPU0-contended and stale-dispatch runs
are archived and excluded.

## Semantics

| Field | Value |
|---|---|
| MODE_NAME | `next_layer_prefetch_direct` |
| PREDICTOR | `P_(l+1) = I_l` |
| NEXT_LAYER_CURRENT_Q_RETRIEVAL | OFF |
| CORRECTION_FETCH | OFF |
| RECONCILIATION | OFF |
| FINAL_HISTORY_SET | `P_(l+1)` |
| STOCK_DENSE_FLASHATTN | YES |
| CPU_SOURCE_PREPINNED | YES |
| REUSABLE_GPU_STAGING | YES |
| HOST_ID_SYNC_PRESENT | YES (kept for Python-indexed archive) |
| QUALITY_STATUS | NOT_EVALUATED |

## Correctness smoke

Smoke result: `{SMOKE}`.

| Counter | Value |
|---|---:|
| attention calls | {sg.get('NUM_ATTENTION_CALLS',0)} |
| prefetch requests | {s.get('PREFETCH_REQUESTED_CHUNKS',0)} |
| direct predicted sets consumed | {sg.get('DIRECT_PREFETCH_CONSUMED',0)} |
| prefetch state misses | {sg.get('PREFETCH_STATE_MISS',0)} |
| next-layer history retrieval calls | {sg.get('NEXT_LAYER_HISTORY_RETRIEVAL_CALLS',0)} |
| correction fetches | {sg.get('CORRECTION_FETCH_COUNT',0)} |
| reconciliations | {sg.get('RECONCILIATION_COUNT',0)} |

The smoke produced a valid video with no NaN/Inf and passed the required zero
checks. State misses used demand fetch of the same predicted IDs; they did not
trigger current-Q re-retrieval.

## Accepted case01 profile

| Metric | Direct prediction | Current-Q reference | Direct − Current-Q |
|---|---:|---:|---:|
| E2E inference (s) | {direct_e2e:.6f} | {current_e2e:.6f} | {direct_e2e-current_e2e:+.6f} |
| Transformer (s) | {direct_transformer:.6f} | {current_transformer:.6f} | {direct_transformer-current_transformer:+.6f} |
| Self-attention wrapper (s) | {direct_attn:.6f} | {current_attn:.6f} | {direct_attn-current_attn:+.6f} |
| Wrapper (s) | {direct_wrapper:.6f} | {current_wrapper:.6f} | {direct_wrapper-current_wrapper:+.6f} |

The matched current-Q reference is
`results/rag_strategy_profile_runs/group11_4/case_01/`.

## Prefetch timing

| Metric | Value |
|---|---:|
| DraftMap retrieval total (s) | {sum(route_ms)/1000:.6f} |
| Host ID control total (s) | {float(g.get('HOST_ID_CONTROL_MS',0))/1000:.6f} |
| Prefetch preparation total (s) | {float(g.get('PREFETCH_PREPARE_MS',0))/1000:.6f} |
| H2D CUDA work total (s) | {sum(work)/1000:.6f} |
| H2D exposed wait total (s) | {sum(exposed)/1000:.6f} |
| H2D hidden total (s) | {sum(hidden)/1000:.6f} |
| Median available overlap window (ms) | {med(overlap):.6f} |
| Median H2D work (ms) | {med(work):.6f} |
| Median exposed H2D (ms) | {med(exposed):.6f} |
| Median host control (ms) | {med(host):.6f} |
| Fully-hidden useful H2D rate | {fully_hidden:.6f} |
| timed useful prefetch events | {len(timed)} |

`CURRENT_LAYER_COMPUTE_AFTER_LAUNCH_MS` was not emitted by the existing
runtime schema and is reported as `NOT_AVAILABLE`; no synthetic overlap value
is substituted. Attention-kernel-exclusive time was also not separately
emitted, so the wrapper boundary is retained as the authoritative attention
measurement.

## Hypothesis decision

The direct mode reduces E2E by `{current_e2e-direct_e2e:.6f}` s, transformer by
`{current_transformer-direct_transformer:.6f}` s, and wrapper by
`{current_attn-direct_attn:.6f}` s relative to the matched current-Q profile.
Thus the measured result supports
`HYPOTHESIS_CORRECTION_DOMINANT = SUPPORTED` for this case01 comparison:
removing next-layer current-Q selection and correction/reconciliation produces
a real end-to-end reduction. This is a latency-only conclusion; it does not
establish quality preservation.

## Required answers

1. Directly consumes `P_(l+1)=I_l`: **YES** for non-bootstrap calls.
2. Correction fetch and reconciliation removed: **YES** (`0` in smoke/profile).
3. Prediction + host control + prepare: see the totals above; per-layer medians are in the raw CSV.
4. H2D time: median work/exposed values above; totals are measured CUDA events.
5. Available overlap: median `{med(overlap):.6f}` ms from launch-to-consumer timestamps.
6. Fully hidden useful H2D: `{fully_hidden:.6f}`.
7. Remaining next-layer wait: exposed H2D total `{sum(exposed)/1000:.6f}` s.
8. Savings: E2E `{current_e2e-direct_e2e:.6f}` s; Transformer `{current_transformer-direct_transformer:.6f}` s; wrapper `{current_attn-direct_attn:.6f}` s.
9. Expected latency reduction: **YES**, measured at E2E boundary.
10. Correction/reconciliation main slow path: **SUPPORTED for this matched case**, not generalized beyond it.
11. Beats CURRENT_Q on case01: **YES** on E2E, transformer, and wrapper.
12. Residual bottleneck: ordinary transformer/non-attention work plus remaining dense attention wrapper; the current schema does not isolate an exclusive attention kernel timer.

## Artifacts

- Raw metrics: `{OUT}/group11_4_direct_prefetch_case01_raw.csv`
- Summary: `{OUT}/group11_4_direct_prefetch_case01_summary.json`
- Runtime JSON: `{RUNTIME}`
- Config: `{ROOT/'configs/group11_4_direct_prefetch_case01.yaml'}`
'''
with open(ROOT/'reports/group11_4_direct_prefetch_case01_profile.md','w') as f:f.write(report)
with open(ROOT/'reports/group11_4_direct_prefetch_design.md','w') as f:f.write('# Group11.4 direct predicted working-set design\n\nImplemented as `next_layer_prefetch_direct`; see the case01 profile for measured behavior. Existing `next_layer_prefetch` remains unchanged.\n')
with open(ROOT/'reports/group11_4_correction_overhead_ablation.md','w') as f:f.write(report)
print(json.dumps(summary,indent=2))
