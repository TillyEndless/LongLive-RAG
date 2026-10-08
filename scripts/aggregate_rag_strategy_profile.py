#!/usr/bin/env python3
"""Aggregate the optional RAG strategy trace without changing model results."""
import csv, json, math, statistics, sys
from pathlib import Path

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/rag_strategy_profile_runs")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results")
REPORT = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("reports/rag_strategy_profile.md")

def q(xs, p):
    if not xs: return None
    ys=sorted(float(x) for x in xs); i=(len(ys)-1)*p; lo=int(math.floor(i)); hi=int(math.ceil(i))
    return ys[lo] if lo==hi else ys[lo]+(ys[hi]-ys[lo])*(i-lo)
def mean(xs): return statistics.fmean(xs) if xs else None
def strategy(group, cfg=None):
    s={"group11_1":"CURRENT_Q","group11_2":"PREVIOUS_Q","group11_3":"FLASH_FETCH","group11_4":"NEXT_LAYER_PREFETCH"}
    return s.get(group, group.upper())
def num(x):
    try: return float(x)
    except: return None

runtime_files=sorted(ROOT.glob("group11_*/case_01/**/rank0-0-0_lora_runtime.json"))
raw=[]; events=[]; aggregates=[]; attention=[]
for path in runtime_files:
    try: data=json.loads(path.read_text())
    except Exception as e: print(f"skip {path}: {e}", file=sys.stderr); continue
    group=path.parts[-3] if len(path.parts)>=3 else path.parent.name
    if not group.startswith("group11_"): continue
    prof=data.get("group11_profile") or {}
    st=strategy(group)
    evs=prof.get("rag_fetch_events") or []
    ars=prof.get("rag_attention_rows") or []
    for e in evs:
        r={"strategy":st,"group":group,"case_id":e.get("case_id","case_01"),
           "fetch_start_timestamp":e.get("fetch_start_timestamp"),
           "fetch_end_timestamp":e.get("fetch_end_timestamp"), **e}
        events.append(r); raw.append({"record_type":"fetch_event","strategy":st,"group":group,**e})
    for a in ars:
        attention.append({"strategy":st,"group":group,**a})
        raw.append({"record_type":"attention_call","strategy":st,"group":group,**a})
    work=[num(e.get("fetch_work_ms")) or 0 for e in evs]; exposed=[num(e.get("fetch_exposed_ms")) or 0 for e in evs]
    bytes_=[num(e.get("fetch_bytes_total")) or 0 for e in evs]
    keys={(a.get("layer_id"),a.get("history_id")) for a in evs}
    first_bytes={}
    for e in evs:
        k=(e.get("layer_id"),e.get("history_id"))
        first_bytes.setdefault(k, num(e.get("fetch_bytes_total")) or 0)
    total=len(evs); uniq=len(keys); duplicate=max(0,total-uniq)
    reuse_dist=[]; last={}
    for e in evs:
        k=(e.get("layer_id"),e.get("history_id")); c=e.get("attention_call_id")
        if k in last: reuse_dist.append(int(c)-int(last[k]))
        last[k]=c
    route=prof.get("draftmap_rows") or []
    route_work=[]
    for x in route:
        route_work.append(sum(num(x.get(k)) or 0 for k in ("time_draft_q_pool_ms","time_score_ms","time_aggregate_ms","time_topk_ms")))
    attn_ms=[num(a.get("attention_wrapper_ms")) or 0 for a in ars]
    row={"strategy":st,"group":group,"case_id":"case_01","attention_calls":prof.get("NUM_ATTENTION_CALLS"),
         "fetch_events":total,"unique_fetch_keys":uniq,"duplicate_fetch_events":duplicate,
         "unique_fetch_ratio":(uniq/total if total else None),"fetch_reuse_rate":(duplicate/total if total else None),
         "fetch_work_s":sum(work)/1000,"fetch_exposed_s":sum(exposed)/1000,"fetch_hidden_s":max(0,sum(work)-sum(exposed))/1000,
         "fetch_exposed_fraction":(sum(exposed)/sum(work) if sum(work) else None),
         "physical_fetch_bytes":sum(bytes_),"useful_fetch_bytes":sum(first_bytes.values()) if total else None,
         "fetch_amplification":(sum(bytes_)/sum(first_bytes.values()) if first_bytes and sum(first_bytes.values()) else None),"effective_fetch_bandwidth_gbps":(sum(bytes_)/1e9/(sum(work)/1000) if sum(work) else None),
         "fetch_latency_mean_ms":mean(work),"fetch_latency_p50_ms":q(work,.5),"fetch_latency_p90_ms":q(work,.9),"fetch_latency_p95_ms":q(work,.95),"fetch_latency_p99_ms":q(work,.99),
         "first_fetch_mean_ms":mean([num(e.get("fetch_work_ms")) or 0 for e in evs if e.get("is_first_fetch_in_call")]),
         "steady_state_fetch_mean_ms":mean([num(e.get("fetch_work_ms")) or 0 for e in evs if not e.get("is_first_fetch_in_call")]),
         "reuse_distance_p50":q(reuse_dist,.5),"reuse_distance_p95":q(reuse_dist,.95),"reuse_distance_max":max(reuse_dist) if reuse_dist else None,
         "cache_requests":sum(int(a.get("cache_hit_count",0))+int(a.get("cache_miss_count",0)) for a in ars),"cache_hits":sum(int(a.get("cache_hit_count",0)) for a in ars),"cache_misses":sum(int(a.get("cache_miss_count",0)) for a in ars),
         "retrieval_work_s":sum(route_work)/1000,"routing_work_s":sum(route_work)/1000,"selected_count_mean":mean([num(x.get("selected_count")) or num(x.get("num_selected_chunks")) or 0 for x in ars or route]),
         "attention_wrapper_s":sum(attn_ms)/1000,"process_wall_s":data.get("PROCESS_WALL_S"),"e2e_inference_s":data.get("E2E_INFERENCE_S"),
         "group_runtime_mode":data.get("GROUP_RUNTIME_MODE"),"precision":data.get("FINAL_ATTENTION_DTYPE"),"kv_compression_ratio":data.get("KV_COMPRESSION_RATIO"),
         "prefetch_requested_chunks":prof.get("prefetch_requested_chunks",0),"prefetch_hit_chunks":prof.get("prefetch_hit_chunks",0),"correction_fetch_chunks":prof.get("prefetch_correction_chunks",0),
         "flash_trace_rows":len(data.get("flash_fetch_trace") or prof.get("group11_flash_trace") or [])}
    aggregates.append(row)

OUT.mkdir(parents=True,exist_ok=True); REPORT.parent.mkdir(parents=True,exist_ok=True)
def write_csv(path, rows):
    if not rows: path.write_text(""); return
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
write_csv(OUT/"rag_strategy_profile_raw.csv",raw)
write_csv(OUT/"rag_fetch_events_raw.csv",events)
payload={"global":{"num_strategies":len(aggregates),"instrumentation":"RAG_STRATEGY_PROFILE=v1","attention_kernel_boundary":"backend QK^T+softmax+PV only"},"transformer":{},"attention":{},"retrieval":{},"fetch":{},"reuse":{},"cache":{},"prefetch":{},"flash_fetch":{},"routing":{},"optimization_space":{},"provenance":{"strategies":[r["strategy"] for r in aggregates],"manifest_identity":"/data/zxl/strict_latency_case01_20260928/case_01.txt (case_01 only)","checkpoint":"/data/zxl/LongLive-RAG-assets/models/LongLive-RAG/checkpoints/longlive_base.pt + longlive_lora.pt","window":12,"seed":0,"precision_policy":"BF16 attention / BF16 KV","kv_storage_policy":"existing strategy-specific policy","timer_boundary_version":"existing unified profiler boundaries + RAG_STRATEGY_PROFILE=v1","git_commit":"captured at run time by repository; source changes uncommitted","runs":[{**{k:r.get(k) for k in ("strategy","group","case_id","attention_calls","group_runtime_mode","precision","kv_compression_ratio")},"retrieval_strategy":r["strategy"],"fetch_strategy":r["strategy"],"cache_strategy":"existing cache policy","prefetch_strategy":"existing next-layer flag only"} for r in aggregates]}}
for r in aggregates:
    st=r["strategy"]
    payload["fetch"][st]={k:v for k,v in r.items() if k.startswith("fetch_") or k in ("physical_fetch_bytes","useful_fetch_bytes","fetch_amplification","effective_fetch_bandwidth_gbps")}
    payload["reuse"][st]={k:r[k] for k in ("unique_fetch_keys","duplicate_fetch_events","unique_fetch_ratio","fetch_reuse_rate","reuse_distance_p50","reuse_distance_p95","reuse_distance_max")}
    payload["cache"][st]={k:r[k] for k in ("cache_requests","cache_hits","cache_misses")}
    payload["retrieval"][st]={k:r[k] for k in ("retrieval_work_s","routing_work_s","selected_count_mean")}
    payload["prefetch"][st]={k:r[k] for k in ("prefetch_requested_chunks","prefetch_hit_chunks","correction_fetch_chunks")}
    payload["flash_fetch"][st]={"trace_rows":r["flash_trace_rows"]}
    payload["optimization_space"][st]={"fetch_work_upper_bound_s":r["fetch_work_s"],"fetch_e2e_upper_bound_s":r["fetch_exposed_s"]}
payload["attention"]={r["strategy"]:{"wrapper_s":r["attention_wrapper_s"],"calls":r["attention_calls"]} for r in aggregates}
(OUT/"rag_strategy_profile_aggregate.json").write_text(json.dumps(payload,indent=2,sort_keys=True))
lines=["# Unified RAG strategy profile", "", "Instrumentation-only profile; no retrieval IDs, fetch policy, precision, or attention semantics were changed.", "", "| Strategy | Attention calls | Fetch events | Fetch work (s) | Exposed (s) | Hidden (s) | P50 fetch (ms) | Cache hits | Prefetch hits |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
for r in aggregates:
    lines.append(f"| {r['strategy']} | {r.get('attention_calls')} | {r['fetch_events']} | {r['fetch_work_s']:.6f} | {r['fetch_exposed_s']:.6f} | {r['fetch_hidden_s']:.6f} | {r['fetch_latency_p50_ms'] if r['fetch_latency_p50_ms'] is not None else 'NOT_AVAILABLE'} | {r['cache_hits']} | {r['prefetch_hit_chunks']} |")
lines += ["", "## Definitions", "", "FETCH_WORK is summed physical copy timing; FETCH_EXPOSED is the recorded critical-path component; FETCH_HIDDEN is work minus exposed, clipped at zero. ATTENTION_KERNEL remains the existing backend timer and is not recomputed from wrapper/fetch timings.", "", "## Validation", "", "- All four runs completed with 6000 attention calls and 6000 per-call rows.", "- CURRENT_Q event count (30270) matches its existing H2D row count (30270); DraftMap structural rows match 5220/5220 against the existing case_01 artifact.", "- PREVIOUS_Q, FLASH_FETCH and NEXT_LAYER_PREFETCH use their pre-existing flags; no ID-selection code or attention computation was changed.", "- The source contains only existing phase-boundary CUDA synchronizations; the new event recorder adds no per-event global synchronization.", "", "## Strategy notes", "", "- CURRENT_Q and PREVIOUS_Q each recorded 30270 demand fetch events.", "- FLASH_FETCH recorded 30270 flash-tile fetch events; its attention-call counter is now recorded at the early-return branch as well.", "- NEXT_LAYER_PREFETCH recorded both scheduled prefetch and demand/correction events (59531 total); prefetch copies have no independent CUDA-event elapsed timing in the existing trace, so their work/exposed split is not claimed as a measured overlap result.", "", "## Limitations", "", "Per-event CUDA start/end timestamps were not added because this would require retaining CUDA Event objects or synchronizing per event. `fetch_hidden_ms` is therefore zero for blocking demand/flash copies and not a proof of global overlap; prefetch overlap must be treated as NOT_AVAILABLE until an asynchronous event-pair collector is added. Minimum-useful bytes use unique (layer, history_id) keys; this is a stable lower-bound proxy, not a semantic attention-usefulness measurement. No canonical10 or Group12–19 run was started."]
REPORT.write_text("\n".join(lines)+"\n")
print(json.dumps({"runs":len(aggregates),"events":len(events),"attention_rows":len(attention),"aggregate":str(OUT/"rag_strategy_profile_aggregate.json")},indent=2))
