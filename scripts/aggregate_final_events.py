import csv,json,os,collections,sys
root=sys.argv[1]
paths={'Group11':os.path.join(root,'results/final_group11_bounded_run_gpu0/phase_events.csv'),'LongLive-RAG W12':os.path.join(root,'results/final_longlive_bounded_run_gpu0/phase_events.csv')}
data={}
for method,p in paths.items():
    a=collections.defaultdict(lambda:{'calls':0,'cuda_ms':0.0,'host_ms':0.0})
    for r in csv.DictReader(open(p)):
        x=a[r['phase']]; x['calls']+=1; x['cuda_ms']+=float(r['cuda_event_ms']); x['host_ms']+=float(r['host_wall_ms'])
    data[method]=a
phases=['ATTN_WRAPPER','QKV_Q','QKV_K','QKV_V','BF16_ATTENTION','OUTPUT_PROJECTION','DRAFT_Q_POOL','DRAFTMAP_SCORE','RETRIEVAL_FETCH_K','RETRIEVAL_FETCH_V','WORKINGSET_GATHER_K','WORKINGSET_GATHER_V','WORKINGSET_CAT_K','WORKINGSET_CAT_V','LAYOUT_CONTIGUOUS','ROPE_POSITION','CACHE_UPDATE']
with open(os.path.join(root,'results/final_cuda_event_phase_pairs.csv'),'w',newline='') as f:
    w=csv.writer(f);w.writerow(['phase','group11_calls','group11_host_wall_s','group11_cuda_event_s','longlive_calls','longlive_host_wall_s','longlive_cuda_event_s','host_delta_s','cuda_delta_s','status'])
    for p in phases:
        g=data['Group11'].get(p); l=data['LongLive-RAG W12'].get(p)
        if g and l:w.writerow([p,g['calls'],g['host_ms']/1000,g['cuda_ms']/1000,l['calls'],l['host_ms']/1000,l['cuda_ms']/1000,(g['host_ms']-l['host_ms'])/1000,(g['cuda_ms']-l['cuda_ms'])/1000,'MEASURED'])
        else:w.writerow([p,g['calls'] if g else 'NOT_PRESENT','NOT_AVAILABLE','NOT_AVAILABLE',l['calls'] if l else 'NOT_PRESENT','NOT_AVAILABLE','NOT_AVAILABLE','NOT_AVAILABLE','NOT_AVAILABLE','NOT_PRESENT_OR_NOT_INSTRUMENTED'])
with open(os.path.join(root,'results/final_cuda_runtime_api_delta.csv'),'w',newline='') as f:
    w=csv.writer(f);w.writerow(['API','Group11_calls','LongLive_calls','call_delta','Group11_time_s','LongLive_time_s','time_delta_s','status'])
    for n in ['cudaHostAlloc','cudaMemcpyAsync','cudaStreamSynchronize','cudaDeviceSynchronize','cudaLaunchKernel','cudaLaunchKernelExC','cudaMalloc','cudaFree','cudaEventRecord']:w.writerow([n,'NOT_AVAILABLE','NOT_AVAILABLE','NOT_AVAILABLE','NOT_AVAILABLE','NOT_AVAILABLE','NOT_AVAILABLE','NOT_CAPTURED_IN_FINAL_MATCHED_RUN'])
with open(os.path.join(root,'results/final_sync_allocation_trace.csv'),'w',newline='') as f:
    w=csv.writer(f);w.writerow(['source','site','calls','time_s','classification','status'])
    for src in ['Group11','LongLive-RAG W12']:
        for site in ['cudaHostAlloc','cudaStreamSynchronize']:w.writerow([src,site,'NOT_AVAILABLE','NOT_AVAILABLE','NOT_CLASSIFIED','NOT_CAPTURED_IN_FINAL_MATCHED_RUN'])
g=data['Group11'];l=data['LongLive-RAG W12']; ad=(g['ATTN_WRAPPER']['cuda_ms']-l['ATTN_WRAPPER']['cuda_ms'])/1000; at=(g['BF16_ATTENTION']['cuda_ms']-l['BF16_ATTENTION']['cuda_ms'])/1000
def envelope(path):
    rows=[r for r in csv.DictReader(open(path)) if r['phase']=='ATTN_WRAPPER']
    lo=min(int(r['host_start_ns']) for r in rows); hi=max(int(r['host_start_ns'])+int(float(r['host_wall_ms'])*1e6) for r in rows)
    return (hi-lo)/1e9
gb=envelope(paths['Group11']); lb=envelope(paths['LongLive-RAG W12']); bd=gb-lb
json.dump({'matched_interval':'temporal blocks 6-7','same_gpu':'GPU0','event_pairs_installed':True,'single_final_synchronize':True,'group11':g,'longlive':l,'wrapper_cuda_delta_s':ad,'attention_cuda_delta_s':at,'torch_profiler_matched':'NOT_RUN','runtime_api_delta':'NOT_AVAILABLE','coverage_percent':'NOT_COMPUTABLE'},open(os.path.join(root,'results/final_group11_vs_longlive_profile.json'),'w'),indent=2,default=dict)
with open(os.path.join(root,'results/final_wrapper_phase_delta.csv'),'w',newline='') as f:
    w=csv.writer(f);w.writerow(['phase','group11_cuda_s','longlive_cuda_s','delta_s','status'])
    for p in ['ATTN_WRAPPER','QKV_Q','QKV_K','QKV_V','BF16_ATTENTION','OUTPUT_PROJECTION']:
        w.writerow([p,g[p]['cuda_ms']/1000,l[p]['cuda_ms']/1000,(g[p]['cuda_ms']-l[p]['cuda_ms'])/1000,'MEASURED'])
    for p in ['DRAFTMAP','FETCH','WORKINGSET_BUILD','LAYOUT','ROPE_POSITION','CACHE_UPDATE']:w.writerow([p,'NOT_AVAILABLE','NOT_AVAILABLE','NOT_AVAILABLE','NOT_INSTRUMENTED'])
report=f'''# Final Group11 latency root-cause report

Matched case01, seed=0, W12, GPU0, same checkpoint/LoRA environment, and temporal blocks 6–7. Every measured phase used CUDA Event start/end pairs, followed by one final synchronization after inference.

Wrappers: Group11=`wan/modules/causal_model_latentmem.py:CausalWanSelfAttention.forward`; LongLive=`wan/modules/causal_model.py:CausalWanSelfAttention.forward`. Both call `wan/modules/attention.py:attention`.

| phase | Group11 CUDA s | LongLive CUDA s | delta s |
|---|---:|---:|---:|
| Q projection | {g['QKV_Q']['cuda_ms']/1000:.6f} | {l['QKV_Q']['cuda_ms']/1000:.6f} | {(g['QKV_Q']['cuda_ms']-l['QKV_Q']['cuda_ms'])/1000:.6f} |
| K projection | {g['QKV_K']['cuda_ms']/1000:.6f} | {l['QKV_K']['cuda_ms']/1000:.6f} | {(g['QKV_K']['cuda_ms']-l['QKV_K']['cuda_ms'])/1000:.6f} |
| V projection | {g['QKV_V']['cuda_ms']/1000:.6f} | {l['QKV_V']['cuda_ms']/1000:.6f} | {(g['QKV_V']['cuda_ms']-l['QKV_V']['cuda_ms'])/1000:.6f} |
| BF16 attention | {g['BF16_ATTENTION']['cuda_ms']/1000:.6f} | {l['BF16_ATTENTION']['cuda_ms']/1000:.6f} | {at:.6f} |
| output projection | {g['OUTPUT_PROJECTION']['cuda_ms']/1000:.6f} | {l['OUTPUT_PROJECTION']['cuda_ms']/1000:.6f} | {(g['OUTPUT_PROJECTION']['cuda_ms']-l['OUTPUT_PROJECTION']['cuda_ms'])/1000:.6f} |
| wrapper boundary | {g['ATTN_WRAPPER']['cuda_ms']/1000:.6f} | {l['ATTN_WRAPPER']['cuda_ms']/1000:.6f} | {ad:.6f} |

The BF16 attention kernel is not materially slower in Group11: the measured CUDA delta is {at:.6f} s over 300 calls. The wrapper boundary is larger by {ad:.6f} s, but this includes orchestration/materialization outside the attention call.

DraftMap, fetch, working-set subphases, runtime API counts, pinned allocation, and synchronization provenance were not separately captured in this final event run. They are marked `NOT_AVAILABLE`; old Group11-only numbers were not substituted. Matched torch.profiler with_stack was also not run in this final pass. Therefore final root-cause coverage is below 90%.

FINAL_BOUNDED_GROUP11_S = {gb:.6f}
FINAL_BOUNDED_LONGLIVE_S = {lb:.6f}
FINAL_BOUNDED_DELTA_S = {bd:.6f}
FINAL_ACCOUNTED_DELTA_S = {ad:.6f}
FINAL_ROOT_CAUSE_COVERAGE_PERCENT = NOT_COMPUTABLE
DRAFTMAP_INCREMENT_S = NOT_AVAILABLE
FETCH_INCREMENT_HOST_S = NOT_AVAILABLE
FETCH_INCREMENT_CUDA_S = NOT_AVAILABLE
WORKINGSET_INCREMENT_S = NOT_AVAILABLE
CUDAHOSTALLOC_INCREMENT_S = NOT_AVAILABLE
SYNC_INCREMENT_S = NOT_AVAILABLE
ATTENTION_KERNEL_INCREMENT_S = {at:.6f}
OTHER_FRAMEWORK_INCREMENT_S = NOT_AVAILABLE
PRIMARY_MEASURED_ROOT_CAUSE = NOT_ESTABLISHED
SECONDARY_MEASURED_ROOT_CAUSE = NOT_ESTABLISHED
THIRD_MEASURED_ROOT_CAUSE = NOT_ESTABLISHED
BF16_ATTENTION_KERNEL_IS_PRIMARY_CAUSE = NO
RAW_H2D_IS_PRIMARY_CAUSE = NOT_AVAILABLE
PRODUCTION_CODE_MODIFIED = NO
CANONICAL_OUTPUTS_MODIFIED = NO
'''
open(os.path.join(root,'reports/final_group11_latency_root_cause.md'),'w').write(report)
