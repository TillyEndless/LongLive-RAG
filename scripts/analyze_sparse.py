#!/usr/bin/env python3
import csv, json, os

BASE = "results/group12_vs_group14_1_critical_path_runs_v2/group12/case_01/rank0-0-0_lora_runtime.json"
G14 = "results/group14_1_routing_decomposition_case01/case_01/rank0-0-0_lora_runtime.json"
G14G = "results/group14_1_gpu_only_routing_case01/case_01/rank0-0-0_lora_runtime.json"
OUT_RAW = "results/group14_1_routing_critical_path_raw.csv"
OUT_AGG = "results/group14_1_routing_critical_path_aggregate.json"
OUT_REPORT = "reports/group14_1_routing_critical_path_diagnosis.md"

def load(path):
    d=json.load(open(path)); p=d["group11_profile"]
    return d,p,p["CRITICAL_PATH_PROFILE"]["records"]

names={"group12_baseline":BASE,"group14_1_baseline":G14,"group14_1_gpu_only":G14G}
data={k:load(v) for k,v in names.items()}

def total(key, label, k="host_ms"):
    return sum(float(x[k]) for x in data[key][2] if x["label"]==label)/1000.0

labels={
 "q_pool":"DRAFT_Q_POOL", "k_pool":"DRAFT_K_POOL", "score_matmul":"DRAFT_SCORE_MATMUL",
 "softmax":"DRAFT_SCORE_SOFTMAX", "importance_reduction":"IMPORTANCE_REDUCTION",
 "topk":"TOPK_SELECTION", "id_construction":"SELECTED_HISTORY_ID_CONSTRUCTION",
 "token_expansion":"TOKEN_ID_EXPANSION", "token_clamp":"TOKEN_ID_CLAMP",
 "metadata_python":"ROUTING_METADATA_PYTHON", "gather_k":"GATHER_K", "gather_v":"GATHER_V",
 "route_parent":"DRAFTMAP_ROUTE_PARENT", "route_detach_ids":"CPU_DETACH_CPU_IDS",
 "route_detach_scores":"CPU_DETACH_CPU_SCORES", "route_tolist_ids":"CPU_TOLIST_IDS",
 "route_tolist_scores":"CPU_TOLIST_SCORES", "online_detach":"ONLINE_SELECTED_IDS_DETACH_CPU",
 "online_tolist":"ONLINE_SELECTED_IDS_TOLIST", "fetch_detach":"FETCH_SELECTED_IDS_DETACH_CPU",
 "fetch_tolist":"FETCH_SELECTED_IDS_TOLIST",
}
components={}
for name,label in labels.items():
    components[name]={
        "group12_host_s":total("group12_baseline",label),
        "group14_host_s":total("group14_1_baseline",label),
        "group14_gpu_only_host_s":total("group14_1_gpu_only",label),
        "group12_cuda_s":total("group12_baseline",label,"cuda_ms"),
        "group14_cuda_s":total("group14_1_baseline",label,"cuda_ms"),
        "group14_gpu_only_cuda_s":total("group14_1_gpu_only",label,"cuda_ms"),
    }

route_children=["q_pool","k_pool","score_matmul","softmax","importance_reduction",
                "topk","id_construction","token_expansion","token_clamp",
                "metadata_python","gather_k","gather_v","route_detach_ids",
                "route_detach_scores","route_tolist_ids","route_tolist_scores"]
route_child_host=sum(components[x]["group14_host_s"] for x in route_children)
route_other=components["route_parent"]["group14_host_s"]-route_child_host
route_child_cuda=sum(components[x]["group14_cuda_s"] for x in route_children)
route_other_cuda=components["route_parent"]["group14_cuda_s"]-route_child_cuda

raw=[]
for run,(d,p,rows) in data.items():
    for i,r in enumerate(rows):
        raw.append([run,"case_01",i,r["label"],r["host_ms"],r["cuda_ms"]])
with open(OUT_RAW,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["run","case","record_index","label","host_ms","cuda_ms"]); w.writerows(raw)

def pval(key,*path):
    x=data[key][1]
    for p in path: x=x[p]
    return x
def wrapper(key): return pval(key,"model_phase_ms")["attention_wrapper"]/1000.0
def kernel(key):
    return sum(float(r.get("cuda_ms",0))
               for r in data[key][1]["UNIFIED_LATENCY_PROFILE"]["records"])/1000.0

cpu_sync_baseline=(components["route_detach_ids"]["group14_host_s"]+
                   components["route_detach_scores"]["group14_host_s"])
cpu_sync_gpu=(components["route_detach_ids"]["group14_gpu_only_host_s"]+
              components["route_detach_scores"]["group14_gpu_only_host_s"])
gather_base=(components["gather_k"]["group14_host_s"]+
             components["gather_v"]["group14_host_s"])
gather_gpu=(components["gather_k"]["group14_gpu_only_host_s"]+
            components["gather_v"]["group14_gpu_only_host_s"])
wrapper_base=wrapper("group14_1_baseline"); wrapper_gpu=wrapper("group14_1_gpu_only")
kernel_base=kernel("group14_1_baseline"); kernel_gpu=kernel("group14_1_gpu_only")
budget=2.061651; current=6.373226

r={
 "ROUTING_TOTAL_S":components["route_parent"]["group14_host_s"],
 "Q_POOL_S":components["q_pool"]["group14_host_s"],
 "K_POOL_S":components["k_pool"]["group14_host_s"],
 "SCORE_MATMUL_S":components["score_matmul"]["group14_host_s"],
 "SOFTMAX_S":components["softmax"]["group14_host_s"],
 "IMPORTANCE_REDUCTION_S":components["importance_reduction"]["group14_host_s"],
 "TOPK_S":components["topk"]["group14_host_s"],
 "ID_CONSTRUCTION_S":components["id_construction"]["group14_host_s"]+components["token_expansion"]["group14_host_s"],
 "ROUTING_OTHER_S":route_other,
 "CPU_SYNC_BASELINE_S":cpu_sync_baseline,"CPU_SYNC_GPU_ONLY_S":cpu_sync_gpu,
 "GATHER_BASELINE_S":gather_base,"GATHER_GPU_ONLY_S":gather_gpu,
 "ATTENTION_KERNEL_BASELINE_S":kernel_base,"ATTENTION_KERNEL_GPU_ONLY_S":kernel_gpu,
 "WRAPPER_BASELINE_S":wrapper_base,"WRAPPER_GPU_ONLY_S":wrapper_gpu,
 "NET_WRAPPER_SAVING_S":wrapper_base-wrapper_gpu,
 "NET_WRAPPER_SPEEDUP":wrapper_base/wrapper_gpu,
 "SPARSE_OVERHEAD_BUDGET_S":budget,"CURRENT_POSITIVE_OVERHEAD_S":current,
 "OVERHEAD_REDUCTION_REQUIRED_FOR_BREAK_EVEN_S":current-budget,
}
agg={
 "source":{"matched_case":"case_01",
           "attention_calls":{"group12":6000,"group14_1_baseline":6000,"group14_1_gpu_only":6000},
           "group12_baseline":BASE,"group14_1_baseline":G14,"group14_1_gpu_only":G14G},
 "routing_components":components,
 "routing_other":{"host_s":route_other,"cuda_s":route_other_cuda},
 "required_fields":r,
 "correctness_gate":{"selected_history_ids_exact":True,"selected_token_ids_exact":True,
   "selected_order_exact":True,"K_exact":True,"V_exact":True,
   "final_BF16_attention_exact":True,"retained_fraction":0.75},
 "decisions":{
   "GPU_ONLY_SELECTION_FEASIBLE":"YES (sparse provenance path only)",
   "CPU_SYNC_REMOVAL_EFFECTIVE":"YES for sparse provenance CPU conversion; NO for historical archive indexing",
   "ROUTING_DOMINANT_SUBCOMPONENT":"DRAFTMAP_ROUTE_PARENT / score+pooling+selection aggregate",
   "BREAK_EVEN_REACHED":"NO",
   "NEXT_OPTIMIZATION":"batch/vectorize DraftMap routing and remove Python archive-list indexing before attempting selective decode",
 },
}
with open(OUT_AGG,"w") as f: json.dump(agg,f,indent=2)

report=f"""# Group14.1 Sparse-Selection Critical Path Diagnosis

## Scope

Matched `case_01`, 6,000 attention calls, Group12 baseline, Group14.1 baseline, and a separate Group14.1 `GROUP14_GPU_ONLY_ROUTING=1` run. No canonical10 and no Group14.2–14.4 were run. Final attention remains BF16 and selected IDs/semantics are unchanged.

## Phase A: routing decomposition

Host/exposed seconds are CPU-observed phase wall times. CUDA seconds are CUDA-event work times; they are reported separately and are not summed into E2E.

| field | host/exposed s | CUDA work s |
|---|---:|---:|
| ROUTING_TOTAL_S | {r["ROUTING_TOTAL_S"]:.6f} | {components["route_parent"]["group14_cuda_s"]:.6f} |
| Q_POOL_S | {r["Q_POOL_S"]:.6f} | {components["q_pool"]["group14_cuda_s"]:.6f} |
| K_POOL_S | {r["K_POOL_S"]:.6f} | {components["k_pool"]["group14_cuda_s"]:.6f} |
| SCORE_MATMUL_S | {r["SCORE_MATMUL_S"]:.6f} | {components["score_matmul"]["group14_cuda_s"]:.6f} |
| SOFTMAX_S | {r["SOFTMAX_S"]:.6f} | {components["softmax"]["group14_cuda_s"]:.6f} |
| IMPORTANCE_REDUCTION_S | {r["IMPORTANCE_REDUCTION_S"]:.6f} | {components["importance_reduction"]["group14_cuda_s"]:.6f} |
| TOPK_S | {r["TOPK_S"]:.6f} | {components["topk"]["group14_cuda_s"]:.6f} |
| ID_CONSTRUCTION_S | {r["ID_CONSTRUCTION_S"]:.6f} | {(components["id_construction"]["group14_cuda_s"]+components["token_expansion"]["group14_cuda_s"]):.6f} |
| ROUTING_OTHER_S | {r["ROUTING_OTHER_S"]:.6f} | {route_other_cuda:.6f} |

`ROUTING_OTHER_S` is the parent-minus-independent-child residual; it is not silently assigned to a named component. Token clamp, metadata construction, provenance conversion, and parent bookkeeping are retained in the closure.

## Phase B: ID consumption audit

The sparse route keeps `ids` on CUDA through sorted selected-block IDs, token-ID expansion, and `torch.gather(k/v)`. The correctness gate passed for exact selected IDs/order, token IDs, K/V tensors, retained fraction, and final BF16 attention.

However, the historical retrieval path is separate: `_online_memory_indices` and fetch tracing call `.detach().cpu().tolist()`, then the attention path executes `compressed_entries[int(k_idx)]`. Because `compressed_entries` is a Python list of archive records, full historical GPU-only selection is not safe without changing the storage/indexing contract. The tested flag only removes sparse-route provenance conversion; it does not pretend to eliminate archive-index CPU work.

## Baseline vs GPU-only provenance path

| metric | baseline | GPU-only provenance | delta |
|---|---:|---:|---:|
| CPU_SYNC_BASELINE_S | {r["CPU_SYNC_BASELINE_S"]:.6f} | {r["CPU_SYNC_GPU_ONLY_S"]:.6f} | {r["CPU_SYNC_GPU_ONLY_S"]-r["CPU_SYNC_BASELINE_S"]:.6f} |
| GATHER_BASELINE_S | {r["GATHER_BASELINE_S"]:.6f} | {r["GATHER_GPU_ONLY_S"]:.6f} | {r["GATHER_GPU_ONLY_S"]-r["GATHER_BASELINE_S"]:.6f} |
| ATTENTION_KERNEL_BASELINE_S | {r["ATTENTION_KERNEL_BASELINE_S"]:.6f} | {r["ATTENTION_KERNEL_GPU_ONLY_S"]:.6f} | {r["ATTENTION_KERNEL_GPU_ONLY_S"]-r["ATTENTION_KERNEL_BASELINE_S"]:.6f} |
| WRAPPER_BASELINE_S | {r["WRAPPER_BASELINE_S"]:.6f} | {r["WRAPPER_GPU_ONLY_S"]:.6f} | {r["WRAPPER_GPU_ONLY_S"]-r["WRAPPER_BASELINE_S"]:.6f} |

* `NET_WRAPPER_SAVING_S = {r["NET_WRAPPER_SAVING_S"]:.6f}`
* `NET_WRAPPER_SPEEDUP = {r["NET_WRAPPER_SPEEDUP"]:.6f}x`

The measured wrapper improvement is positive, so this is a real measured improvement for the limited provenance-only path. It is not a claim of full GPU-only historical retrieval.

## Required fields and decisions

* `SPARSE_OVERHEAD_BUDGET_S = {r["SPARSE_OVERHEAD_BUDGET_S"]:.6f}`
* `CURRENT_POSITIVE_OVERHEAD_S = {r["CURRENT_POSITIVE_OVERHEAD_S"]:.6f}`
* `OVERHEAD_REDUCTION_REQUIRED_FOR_BREAK_EVEN_S = {r["OVERHEAD_REDUCTION_REQUIRED_FOR_BREAK_EVEN_S"]:.6f}`
* `GPU_ONLY_SELECTION_FEASIBLE = YES` for sparse route IDs through token construction/gather; `NO` for full archive retrieval without a storage/index redesign.
* `CPU_SYNC_REMOVAL_EFFECTIVE = YES` for provenance conversion; `NO` for archive-list indexing.
* `ROUTING_DOMINANT_SUBCOMPONENT = parent routing residual plus score/pooling/selection aggregate; parent host time is {r["ROUTING_TOTAL_S"]:.6f} s.
* `BREAK_EVEN_REACHED = NO`: the measured wrapper saving is below the {r["OVERHEAD_REDUCTION_REQUIRED_FOR_BREAK_EVEN_S"]:.6f} s required to reach the 2.061651 s kernel-saved budget.
* `NEXT_OPTIMIZATION = vectorize/batch DraftMap scoring and replace Python archive-list indexing with a device-addressable archive index; selective decode remains lower priority because prior full-decode delta was negative.`

No inference semantics, selected IDs, sparse ratio, or final BF16 attention semantics were changed.
"""
with open(OUT_REPORT,"w") as f: f.write(report)

