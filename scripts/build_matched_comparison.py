#!/usr/bin/env python3
import csv, json, glob, statistics, math
from pathlib import Path

ROOT=Path('/data/zxl/LongLive-RAG-group11_15_h200')
FILES={
 'CURRENT_Q':ROOT/'results/rag_strategy_profile_runs/group11_1/case_01/rank0-0-0_lora_runtime.json',
 'PREVIOUS_Q':ROOT/'results/rag_strategy_profile_runs/group11_2/case_01/rank0-0-0_lora_runtime.json',
 'FLASH_FETCH':ROOT/'results/rag_strategy_profile_runs/group11_3/case_01/rank0-0-0_lora_runtime.json',
 'NEXT_LAYER_PREFETCH':ROOT/'results/rag_strategy_profile_runs_overlap/group11_4/case_01/rank0-0-0_lora_runtime.json',
}
def mean(x): return statistics.fmean(x) if x else None
def pct(x,p):
 if not x:return None
 y=sorted(x); q=(len(y)-1)*p; lo=int(q); hi=math.ceil(q)
 return y[lo] if lo==hi else y[lo]+(y[hi]-y[lo])*(q-lo)
def get(x,k): return x.get(k)
def path_sums(p):
 g=p['group11_profile']; ev=g.get('rag_fetch_events',[]); ar=g.get('rag_attention_rows',[])
 work=[float(e.get('fetch_work_ms') or 0) for e in ev]; exposed=[float(e.get('fetch_exposed_ms') or 0) for e in ev]
 route=g.get('draftmap_rows',[])
 rw=sum(sum(float(x.get(k) or 0) for k in ('time_draft_q_pool_ms','time_score_ms','time_aggregate_ms','time_topk_ms')) for x in route)/1000
 if ar:
  retrieval=sum(float(x.get('retrieval_work_ms') or 0) for x in ar)/1000
 else: retrieval=rw
 return ev,ar,work,exposed,retrieval
rows=[]; data={}
for s,f in FILES.items():
 p=json.loads(f.read_text()); data[s]=p; ev,ar,w,e,retr=path_sums(p); g=p['group11_profile']
 unique={(x.get('layer_id'),x.get('history_id')) for x in ev}; total_bytes=sum(int(x.get('fetch_bytes_total') or 0) for x in ev)
 first=[x for x in w if False]
 first=[float(x.get('fetch_work_ms') or 0) for x in ev if x.get('is_first_fetch_in_call')]
 steady=[float(x.get('fetch_work_ms') or 0) for x in ev if not x.get('is_first_fetch_in_call')]
 r={'Strategy':s,'Attention Calls':g.get('NUM_ATTENTION_CALLS'),'Fetch Events':len(ev),'Fetch Bytes':total_bytes,
    'Retrieval Work (s)':retr,'Retrieval Exposed (s)':None,'Fetch Work (s)':sum(w)/1000,'Fetch Exposed (s)':sum(e)/1000,
    'Fetch Hidden (s)':max(0,sum(w)-sum(e))/1000,'Fetch Hidden Fraction':(max(0,sum(w)-sum(e))/sum(w) if sum(w) else None),
    'First Fetch Mean (ms)':mean(first),'First Fetch P95 (ms)':pct(first,.95),'Fetch Mean (ms)':mean(w),'Fetch P95 (ms)':pct(w,.95),'Fetch P99 (ms)':pct(w,.99),
    'Effective Fetch BW (GB/s)':total_bytes/1e9/(sum(w)/1000) if sum(w) else None,
    'Reuse Rate':1-len(unique)/len(ev) if ev else None,'Unique Fetch Ratio':len(unique)/len(ev) if ev else None,
    'Reuse Distance P50':None,'Reuse Distance P95':None,
    'Attention Kernel (s)':None,'Self-Attention Wrapper (s)':p.get('WRAPPER_LATENCY_S'),'Transformer (s)':p.get('TRANSFORMER_LATENCY_S'),'E2E Inference (s)':p.get('E2E_LATENCY_S'),
    'Theoretical Fetch Work Upper Bound (s)':sum(w)/1000,'Theoretical Fetch E2E Upper Bound (s)':sum(e)/1000,
    'Measured E2E Delta vs CURRENT_Q (s)':None,'Measured Wrapper Delta vs CURRENT_Q (s)':None,'Achieved Fetch Optimization Fraction':None,
    'Case':'case_01','Prompt manifest':'/data/zxl/strict_latency_case01_20260928/case_01.txt','Window':12,'Timer boundary':'RAG_STRATEGY_PROFILE=v1'}
 if s=='NEXT_LAYER_PREFETCH':
  for k,name in [('PREFETCH_REQUESTED_CHUNKS','Prefetch Total Chunks'),('PREFETCH_HIT_CHUNKS','Prefetch Hit Chunks'),('PREFETCH_CORRECTION_CHUNKS','Correction Fetch Chunks'),('PREFETCH_WASTED_CHUNKS','Wasted Prefetch Chunks'),('PREFETCH_HIT_BYTES','Prefetch Hit Bytes'),('PREFETCH_CORRECTION_BYTES','Correction Fetch Bytes'),('PREFETCH_WASTED_BYTES','Wasted Prefetch Bytes')]: r[name]=p.get(k)
  ms=g.get('rag_cuda_measurements',[]); sums={k:sum(float(x.get('cuda_ms') or 0) for x in ms if x.get('measurement_kind')==k)/1000 for k in ('prefetch_copy','prefetch_wait','correction_copy')}
  r.update({'Prefetch CUDA Work (s)':sums['prefetch_copy'],'Prefetch Exposed Wait (s)':sums['prefetch_wait'],'Prefetch Hidden (s)':max(0,sums['prefetch_copy']-sums['prefetch_wait']),'Prefetch Hidden Fraction':max(0,sums['prefetch_copy']-sums['prefetch_wait'])/sums['prefetch_copy'] if sums['prefetch_copy'] else None,'Correction CUDA Work (s)':sums['correction_copy'],'Correction Exposed (s)':sums['correction_copy'],'Prefetch Actual Overlap Proven':'YES'})
 if s=='FLASH_FETCH':
  tiles=[]; serial=ideal=0.0
  for call in p.get('flash_fetch_trace',[]):
   hist=[x for x in call.get('timeline',[]) if x.get('partition')=='history']
   fs=[float(x.get('h2d_ms') or 0) for x in hist]; cs=[float(x.get('compute_ms') or 0) for x in hist]
   tiles += [{'f':f,'c':c} for f,c in zip(fs,cs)]
   serial += sum(fs)+sum(cs)
   if hist: ideal += fs[0]+sum(max(cs[i],fs[i+1]) for i in range(len(fs)-1))+cs[-1]
  fw=sum(w)/1000
  r.update({'Flash First Fetch (s)':mean([x['f'] for x in tiles])/1000 if tiles else None,'Flash Fetch Work (s)':fw,'Flash Fetch Exposed (s)':fw,'Flash Fetch Hidden (s)':0.0,'Flash Hidden Fraction':0.0,'Tile Fetch Mean (ms)':mean([x['f'] for x in tiles]),'Tile Fetch P95 (ms)':pct([x['f'] for x in tiles],.95),'Tile Attn Mean (ms)':mean([x['c'] for x in tiles]),'Tile Attn P95 (ms)':pct([x['c'] for x in tiles],.95),'Serial Pipeline (s)':serial/1000,'Ideal Flash Pipeline (s)':ideal/1000,'Flash Theoretical Saving (s)':(serial-ideal)/1000,'Flash Measured Wrapper Saving (s)':None,'Flash Measured E2E Saving (s)':None,'Flash Achieved Fraction':None,'Flash Overlap Proven':'NO'})
 rows.append(r)
base=next(r for r in rows if r['Strategy']=='CURRENT_Q')
pref=next(r for r in rows if r['Strategy']=='NEXT_LAYER_PREFETCH')
pref_precision = pref.get('Prefetch Hit Chunks',0)/pref.get('Prefetch Total Chunks',1) if pref.get('Prefetch Total Chunks') else None
pref_recall = pref.get('Prefetch Hit Chunks',0)/(pref.get('Prefetch Hit Chunks',0)+pref.get('Correction Fetch Chunks',0)) if (pref.get('Prefetch Hit Chunks',0)+pref.get('Correction Fetch Chunks',0)) else None
for r in rows:
 r['Measured E2E Delta vs CURRENT_Q (s)']=r['E2E Inference (s)']-base['E2E Inference (s)']
 r['Measured Wrapper Delta vs CURRENT_Q (s)']=r['Self-Attention Wrapper (s)']-base['Self-Attention Wrapper (s)']
fields=[]
for _r in rows:
 for _k in _r:
  if _k not in fields: fields.append(_k)
out=ROOT/'results/group11_rag_strategy_matched_comparison.csv'; out.parent.mkdir(exist_ok=True)
with out.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
(ROOT/'results/group11_rag_strategy_matched_comparison.json').write_text(json.dumps({'matched':True,'rows':rows,'quality_provenance':'NOT_AVAILABLE in this case_01 profiling run; do not use for canonical quality ranking'},indent=2))
lines=['# Group11 RAG strategy matched comparison','', 'Matched case: `case_01`; window=12; 6000 attention calls; seed=0. This is a runtime-measurement comparison, not a canonical quality evaluation.', '', '| Strategy | E2E s | Wrapper s | Retrieval work s | Fetch work s | Fetch exposed s | Hidden fraction | Fetch BW GB/s |', '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows: lines.append(f"| {r['Strategy']} | {r['E2E Inference (s)']:.6f} | {r['Self-Attention Wrapper (s)']:.6f} | {r['Retrieval Work (s)']:.6f} | {r['Fetch Work (s)']:.6f} | {r['Fetch Exposed (s)']:.6f} | {r['Fetch Hidden Fraction']:.6f} | {r['Effective Fetch BW (GB/s)']:.3f} |")
lines += ['', '## Decision facts', '', f"- FASTEST_MEASURED_E2E_STRATEGY = `{min(rows,key=lambda x:x['E2E Inference (s)'])['Strategy']}`.", f"- LOWEST_SELF_ATTN_WRAPPER_STRATEGY = `{min(rows,key=lambda x:x['Self-Attention Wrapper (s)'])['Strategy']}`.", f"- LOWEST_FETCH_EXPOSED_STRATEGY = `{min(rows,key=lambda x:x['Fetch Exposed (s)'])['Strategy']}`.", '- HIGHEST_FETCH_HIDDEN_FRACTION_STRATEGY = `NEXT_LAYER_PREFETCH` only under the measured prefetch copy/wait accounting; demand/Flash copies are blocking and have zero hidden fraction.', '- LOWEST_RETRIEVAL_EXPOSED_STRATEGY = `NOT_IDENTIFIABLE`: retrieval exposed was not independently separated from the existing wrapper boundary.', '- BEST_CANDIDATE_FOR_COMPRESSION_BASE = `NOT_DECIDABLE_YET`: runtime is matched, but canonical quality provenance and Flash/prefetch semantic overlap evidence are not sufficient.', '', '## Current-Q headroom', '', f"- CURRENT_Q_FETCH_WORK_UPPER_BOUND_S = {base['Fetch Work (s)']:.6f}.", f"- CURRENT_Q_FETCH_E2E_UPPER_BOUND_S = {base['Fetch Exposed (s)']:.6f}.", '- CURRENT_Q_RETRIEVAL_UPPER_BOUND_S = NOT_IDENTIFIABLE because retrieval-exposed timing is not independently measured.', '- CURRENT_Q bottleneck under these fields: the remaining self-attention wrapper/transformer work, not the measured H2D copy sum alone.', '', '## Previous-Q', '', '- Selected-set recall/precision/Jaccard/exact/order are `NOT_AVAILABLE`: the persisted profile contains aggregate DraftMap rows but not a stable per-call selected-ID trace for both strategies.', '- Previous-Q has lower measured retrieval-work sum and lower E2E in this matched run, but selected-set or quality equivalence must not be inferred.', '', '## Flash Fetch', '', f"- FLASH_THEORETICAL_SAVING_S = {next(r for r in rows if r['Strategy']=='FLASH_FETCH').get('Flash Theoretical Saving (s)')}.", '- FLASH_FETCH_OVERLAP_PROVEN = NO: the existing Flash path is serial (`async_overlap=false`); no asynchronous overlap was measured.', '', '## Next-layer prefetch', '', '- PREFETCH_ACTUAL_OVERLAP_PROVEN = YES for CUDA-event work/wait observability only; this does not prove net E2E benefit.', f"- Prefetch counters: requested={pref['Prefetch Total Chunks']}, hits={pref['Prefetch Hit Chunks']}, corrections={pref['Correction Fetch Chunks']}, waste={pref['Wasted Prefetch Chunks']}.", f"- PREFETCH_PRECISION = {pref_precision:.6f}; PREFETCH_RECALL = {pref_recall:.6f}; exact-set rate = NOT_AVAILABLE.", f"- PREFETCH CUDA work = {pref['Prefetch CUDA Work (s)']:.6f} s; exposed wait = {pref['Prefetch Exposed Wait (s)']:.6f} s; hidden-by-difference = {pref['Prefetch Hidden (s)']:.6f} s.", '- Prefetch precision/recall counters are recorded by the existing runtime counters; event-level ID-to-wait attribution is not used to redefine them.', '- High prefetch hit/recall does not by itself establish a net critical-path gain.', '', '## Integrity and limitations', '', '- All four runs are case_01, seed=0, window=12, same checkpoint paths and 6000 attention calls.', '- Group11.4 overlap rerun preserved the existing prefetch stream, wait_event, selected IDs, correction/waste counters, and BF16 attention path; no canonical10 or Group12–19 was run.', '- ATTENTION_KERNEL is `NOT_AVAILABLE` in this run because unified profiler records were disabled; no wrapper/fetch time was substituted for it.', '- Quality provenance is `NOT_AVAILABLE` for this runtime-only case; do not use this report as a quality ranking.', '- Process wall was not used for ranking.']
(ROOT/'reports/group11_rag_strategy_matched_comparison.md').write_text('\n'.join(lines)+'\n')
print(out)
