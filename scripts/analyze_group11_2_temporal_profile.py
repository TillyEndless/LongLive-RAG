#!/usr/bin/env python3
import csv, json, math, statistics
from pathlib import Path

ROOT=Path('/data/zxl/LongLive-RAG-group11_15_h200')
OUT=ROOT/'results/group11_2_temporal_prefetch_case01_profile'
RUNTIME=OUT/'rank0-0-0_lora_runtime.json'
CSV_OUT=ROOT/'results/group11_2_temporal_prefetch_case01_events.csv'
SUMMARY_OUT=ROOT/'results/group11_2_temporal_prefetch_case01_summary.json'
REPORT_OUT=ROOT/'reports/group11_2_temporal_prefetch_case01_profile.md'

def n(v):
    return None if v is None else float(v)
def ok(v):
    return v is not None and math.isfinite(float(v))
def dist(xs):
    xs=sorted(float(x) for x in xs if ok(x))
    if not xs: return {'count':0,'mean_ms':None,'median_ms':None,'p90_ms':None,'p95_ms':None,'min_ms':None,'max_ms':None}
    def pct(q):
        if len(xs)==1: return xs[0]
        x=(len(xs)-1)*q; lo=int(x); hi=min(lo+1,len(xs)-1)
        return xs[lo]+(xs[hi]-xs[lo])*(x-lo)
    return {'count':len(xs),'mean_ms':statistics.fmean(xs),'median_ms':statistics.median(xs),'p90_ms':pct(.9),'p95_ms':pct(.95),'min_ms':xs[0],'max_ms':xs[-1]}

d=json.loads(RUNTIME.read_text()); g=d['group11_profile']; events=g.get('TEMPORAL_PREFETCH_EVENTS',[]); rows=[]
for e in events:
    a,b=n(e.get('predict_start_host')),n(e.get('predict_end_host')); enq,need=n(e.get('h2d_enqueue_host')),n(e.get('target_consumer_need_host'))
    predict=n(e.get('predict_work_ms'))
    if predict is None and ok(a) and ok(b): predict=(b-a)*1000
    available=(need-enq)*1000 if ok(need) and ok(enq) else None
    h2d=n(e.get('h2d_work_ms')); wait=n(e.get('target_wait_event_ms'))
    host_wait=n(e.get('target_wait_call_host_ms'))
    rows.append({'case':'case_01','window':12,'layer_id':e.get('layer_id'),'target_valid_invocation_id':e.get('target_valid_invocation_id'),'target_invocation_id':e.get('target_invocation_id'),'source_invocation_id':e.get('source_invocation_id'),'requested_ids':e.get('requested_ids'),'selected_ids':json.dumps(e.get('selected_ids',[]),separators=(',',':')),'qpredict_work_ms':predict,'host_id_control_ms':n(e.get('host_control_ms')),'archive_lookup_ms':n(e.get('archive_lookup_ms')),'prefetch_prepare_ms':n(e.get('prefetch_prepare_ms')),'h2d_work_ms':h2d,'h2d_exposed_wait_ms':host_wait,'target_wait_event_ms':wait,'kv_assembly_ms':n(e.get('kv_assembly_ms')),'available_overlap_window_ms':available,'overlap_margin_ms':available-h2d if ok(available) and ok(h2d) else None,'fully_hidden_h2d':host_wait is not None and host_wait<=.01})

CSV_OUT.parent.mkdir(parents=True,exist_ok=True)
fields=list(rows[0]) if rows else ['case','window']
with CSV_OUT.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def vals(k): return [r[k] for r in rows if ok(r.get(k))]
summary={'case':'case_01','window':12,'result_dir':str(OUT),'runtime_json':str(RUNTIME),'process_wall_s':None,'e2e_latency_s':float(g.get('E2E_LATENCY_MS',0))/1000,'transformer_latency_s':float(g.get('TRANSFORMER_LATENCY_MS',0))/1000,'wrapper_latency_s':float(g.get('WRAPPER_LATENCY_MS',0))/1000,'self_attention_wrapper_s':float(g['SELF_ATTN_WRAPPER_MS'])/1000 if g.get('SELF_ATTN_WRAPPER_MS') is not None else None,'events':len(rows),'invariants':{'NON_BOOTSTRAP_PREFETCH_MISS':g.get('NON_BOOTSTRAP_PREFETCH_MISS',0),'NON_BOOTSTRAP_FALLBACK_COUNT':g.get('NON_BOOTSTRAP_FALLBACK_COUNT',0),'CURRENT_INVOCATION_QPREV_RERETRIEVAL':g.get('CURRENT_INVOCATION_QPREV_RETRIEVAL_COUNT',0),'DEMAND_H2D_ON_VALID_PREFETCH':g.get('DEMAND_H2D_ON_VALID_PREFETCH_HIT',0),'EXACT_Q_T_MINUS_1_ALIGNMENT':g.get('EXACT_Q_T_MINUS_1_ALIGNMENT'),'FINAL_WORKING_SET_EQUALS_QPREV_PREDICTION':g.get('FINAL_WORKING_SET_EQUALS_QPREV_PREDICTION')},'counts':{k:g.get(k) for k in ['NUM_ATTENTION_CALLS','BOOTSTRAP_CALLS','QPREFETCH_PREDICTIONS','QPREFETCH_REQUESTED_IDS','QPREFETCH_H2D_COUNT','QPREFETCH_HITS','QPREFETCH_MISSES','QPREFETCH_STALE_REJECTS','QPREFETCH_TEMPORAL_ALIGNMENT_PASS','QPREFETCH_TEMPORAL_ALIGNMENT_FAIL']},'timing_summary_ms':{k:dist(vals(k)) for k in ['qpredict_work_ms','host_id_control_ms','archive_lookup_ms','prefetch_prepare_ms','h2d_work_ms','h2d_exposed_wait_ms','target_wait_event_ms','kv_assembly_ms','available_overlap_window_ms','overlap_margin_ms']},'unavailable_phase_breakdown':['output_projection_after_launch','ffn_after_launch','norm_residual_after_launch']}
wall=OUT/'process_wall.txt'
if wall.exists():
    for line in wall.read_text().splitlines():
        if line.startswith('real '): summary['process_wall_s']=float(line.split()[1])
SUMMARY_OUT.write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
def f(v): return 'NA' if v is None else f'{v:.3f}'
ts=summary['timing_summary_ms']; labels=[('qpredict_work_ms','Q-predict work'),('host_id_control_ms','host ID control'),('archive_lookup_ms','archive lookup'),('prefetch_prepare_ms','prefetch prepare'),('h2d_work_ms','H2D CUDA work'),('h2d_exposed_wait_ms','exposed/event wait'),('kv_assembly_ms','KV assembly'),('available_overlap_window_ms','available overlap window'),('overlap_margin_ms','overlap margin')]
lines=['# Group11.2 previous-Q temporal-prefetch case01 profile','','Matched case01 latency profile only; no smoke, canonical10, or evaluator.','',f'- result directory: `{OUT}`',f'- process wall time: **{f(summary["process_wall_s"])} s**',f'- E2E: **{summary["e2e_latency_s"]:.3f} s**; Transformer: **{summary["transformer_latency_s"]:.3f} s**; Wrapper: **{summary["wrapper_latency_s"]:.3f} s**','', '## Correctness invariants','',f'- attention calls: `{g.get("NUM_ATTENTION_CALLS")}`; bootstrap: `{g.get("BOOTSTRAP_CALLS")}`; temporal predictions: `{g.get("QPREFETCH_PREDICTIONS")}`',f'- requested/H2D IDs: `{g.get("QPREFETCH_REQUESTED_IDS")}` / `{g.get("QPREFETCH_H2D_COUNT")}`; valid prefetch hits: `{g.get("QPREFETCH_HITS")}`',f'- non-bootstrap miss: `{g.get("NON_BOOTSTRAP_PREFETCH_MISS",0)}`; fallback: `{g.get("NON_BOOTSTRAP_FALLBACK_COUNT",0)}`; current-Q re-retrieval: `{g.get("CURRENT_INVOCATION_QPREV_RETRIEVAL_COUNT",0)}`',f'- demand H2D on valid prefetch: `{g.get("DEMAND_H2D_ON_VALID_PREFETCH_HIT",0)}`',f'- exact Q(t−1) alignment: **{g.get("EXACT_Q_T_MINUS_1_ALIGNMENT","NA")}**',f'- final working set equals Q(t−1) prediction: **{g.get("FINAL_WORKING_SET_EQUALS_QPREV_PREDICTION","NA")}**','', '## Component timing','', '| component | count | mean ms | p50 ms | p95 ms |','|---|---:|---:|---:|---:|']
for k,label in labels:
    z=ts[k]; lines.append(f'| {label} | {z["count"]} | {f(z["mean_ms"])} | {f(z["median_ms"])} | {f(z["p95_ms"])} |')
lines += ['', 'H2D CUDA work and consumer-stream wait are reported separately. Positive overlap margin means measured H2D work fit within the available window.', '', '## Limits', '', '- Output projection, FFN, and norm/residual were not separately instrumented; they remain `NA` rather than being inferred from the parent wrapper timer.', '- This is one matched case and is not a multi-case speed claim.', '', '## Artifacts', '', f'- events CSV: `{CSV_OUT}`', f'- summary JSON: `{SUMMARY_OUT}`', f'- report: `{REPORT_OUT}`']
REPORT_OUT.parent.mkdir(parents=True,exist_ok=True); REPORT_OUT.write_text('\n'.join(lines)+'\n')
print(CSV_OUT); print(SUMMARY_OUT); print(REPORT_OUT)
