import json
from pathlib import Path

root = Path("/data/zxl/LongLive-RAG-group11_15_h200")
reports = root / "reports"
results = root / "results"
reports.mkdir(parents=True, exist_ok=True)
results.mkdir(parents=True, exist_ok=True)

head = "410948eb7e328513e7751ac2d44193115363cee4"
src = "wan/modules/causal_model_latentmem.py"

common = """# Group11 static implementation audit

Audited worktree: `/data/zxl/LongLive-RAG-group11_15_h200`
Audited HEAD: `{head}` (the worktree also contains unrelated uncommitted experimental files; this audit is source-level only).

No inference, benchmark, or code modification was performed for this audit.

## Common execution points

- `_online_memory_indices`: `{src}:248-343`; scores DraftMap candidates and returns selected history IDs.
- `forward`: `{src}:550-607`; computes Q/K/V, invokes retrieval, and passes `memory_indices` into the history assembly path.
- History assembly/final working set: `{src}:930-1114`; selected historical K/V, sink, and local/resident K/V are assembled before unchanged BF16 attention.
- Final attention: `{src}:1122-1154`; `prepare_attention_kv` is followed by `attention(roped_query, k_cat, v_cat)`.

The implementation still records selected IDs with `selected.detach().cpu().tolist()` for provenance. That is a host-side control/observability synchronization; it is not used to change the returned GPU selection tensor for the normal fetch path.
""".format(head=head, src=src)

current_json = {
    "strategy": "Group11.1 CURRENT_Q",
    "source": {"worktree": str(root), "head": head, "file": src},
    "static_only": True,
    "CURRENT_Q_SOURCE_EXACT": "YES",
    "SELECTED_IDS_USED_DIRECTLY": "YES",
    "NO_QPREV_STATE": "YES (config retrieval_query_mode defaults to current_q)",
    "NO_NEXT_LAYER_PREDICTION": "YES",
    "NO_SPECULATIVE_FETCH": "YES",
    "NO_HOT_CACHE_REUSE": "YES by Group11.1 config; optional GROUP11_REUSE_CACHE code is disabled",
    "FINAL_WORKING_SET_EQUALS_SELECTED_PLUS_LOCAL": "YES",
    "call_graph": [
        "current Q -> _online_memory_indices (248-343)",
        "DraftChunkRecord / DraftMap score_history (278-293)",
        "select_topk (294-296)",
        "selected GPU IDs returned (343)",
        "history lookup and H2D in forward (930-1057)",
        "RoPE and stack/cat (1060-1114)",
        "BF16 attention (1122-1154)",
    ],
    "evidence": {
        "config": "configs/g11_1.yaml:15-17; no retrieval_query_mode, group11_fetch_mode, or reuse flag",
        "current_query": f"{src}:259-260, 263-273",
        "selection_return": f"{src}:294-343",
        "working_set": f"{src}:1093-1114",
    },
    "limitations": ["The CPU ID conversion used for trace/provenance remains a possible control overhead, but does not alter the selected GPU tensor."],
}

qprev_json = {
    "strategy": "Group11.2 PREVIOUS_Q",
    "source": {"worktree": str(root), "head": head, "file": src},
    "static_only": True,
    "QPREV_STATE_LAYER_SPECIFIC": "YES: per-layer kv_cache and explicit layer check",
    "QPREV_STATE_TIME_ALIGNED": "NO / NOT_PROVABLY_EXACT: layer is checked, but predecessor chunk/denoising-step sequence is not validated",
    "QPREV_RETRIEVAL_DIRECT": "YES after bootstrap",
    "CURRENT_Q_CORRECTION_PRESENT": "NO",
    "EXTRA_FETCH_BEYOND_SELECTED_IDS": "NO in the qprev path; group11_fetch_mode remains serial_full",
    "FINAL_WORKING_SET_EQUALS_QPREV_SELECTION": "YES after bootstrap",
    "QPREV_EXTRA_CONTROL_OVERHEAD": [
        f"{src}:442-448 query.detach().clone() into draft_q_pending on every invocation",
        f"{src}:450-457 pending->history commit and metadata dictionary update",
        f"{src}:270-275 only checks layer and optionally records denoising-step age; it does not enforce exact t-1",
        f"{src}:300-306 selected ID detach/cpu/tolist for provenance, also present in CURRENT_Q",
    ],
    "call_graph": [
        "current Q computed -> _queue_draft_q_history (600)",
        "previous history read or bootstrap current Q in _online_memory_indices (262-277)",
        "DraftMap selection and direct selected tensor return (278-343)",
        "selected qprev IDs fetched and assembled into attention working set (930-1114)",
        "current Q is committed only after attention (1216-1217)",
    ],
    "evidence": {
        "config": "configs/g11_2.yaml:15-18 retrieval_query_mode: previous_q",
        "state_storage": f"{src}:442-457",
        "selection_source": f"{src}:263-277",
        "final_working_set": f"{src}:1093-1114",
    },
    "conclusion": "It is not a predictive-prefetch path and does not add a second current-Q correction retrieval. It differs from CURRENT_Q by retrieval-query state plus clone/metadata overhead, with a first-use bootstrap to current Q. Exact temporal t-1 alignment is not statically guaranteed.",
}

prefetch_json = {
    "strategy": "Group11.4 NEXT_LAYER_PREFETCH",
    "source": {"worktree": str(root), "head": head, "file": src},
    "static_only": True,
    "CURRENT_Q_RETRIEVAL_PRESERVED": "YES",
    "PREDICTOR_EQUALS_CURRENT_SELECTED_IDS": "YES",
    "NEXT_LAYER_KEY_CORRECT": "YES: (layer_index + 1, history_id)",
    "SEPARATE_PREFETCH_AND_COMPUTE_STREAMS": "YES",
    "PREFETCH_SOURCE_PINNED": "YES, pins source if needed",
    "PREFETCH_NON_BLOCKING": "YES",
    "PREFETCH_DONE_EVENT_PRESENT": "YES",
    "PREFETCH_BLOCKS_CURRENT_LAYER": "NO explicit wait before current attention; YES for pre-launch host control overhead because selected.detach().cpu().tolist() occurs first",
    "OVERLAP_WINDOW_STRUCTURALLY_EXISTS": "YES: prefetch is scheduled at 604-607 before current-layer history assembly and attention",
    "PREFETCH_GPU_DATA_ACTUALLY_REUSED": "YES on exact hit",
    "HIT_REUSES_PREFETCH_TENSOR_DIRECTLY": "YES at 440",
    "HIT_REQUIRES_NEW_H2D": "NO",
    "HIT_REQUIRES_NEW_GPU_COPY": "NO explicit copy; later stack/cat materializes the attention operand",
    "CORRECTION_FETCH_EXACTLY_ONCE": "YES per selected missing (layer, history_id) in the visible loop; IDs are top-k unique",
    "WASTE_NEVER_ENTERS_ATTENTION": "YES; _reconcile_prefetch removes stale entries before assembly",
    "WASTE_POSTPROCESSING_PRESENT": "NO RoPE/repack before discard; waste still pays pinning/allocation/H2D",
    "WASTED_EXTRA_WORK": "H2D, source pinning when needed, destination allocation, event creation, and Python bookkeeping",
    "DUPLICATE_H2D_POSSIBLE": "YES across separate invocations after an entry is popped; no durable cross-call hit cache",
    "DUPLICATE_H2D_CODE_PATH": f"{src}:420-423 pops the entry; {src}:965-1001 falls back to demand .to(device)",
    "GPU_TO_CPU_ID_SYNC_PRESENT": "YES at 378 and 605",
    "PYTHON_ARCHIVE_INDEXING_PRESENT": "YES at 376-394",
    "PER_PREFETCH_HOST_LOOP_PRESENT": "YES at 381-412",
    "PER_PREFETCH_GPU_ALLOCATION": "YES: src.to(device) at 394",
    "PER_HIT_GPU_ALLOCATION": "NO new H2D/allocation on hit",
    "PER_CORRECTION_GPU_ALLOCATION": "YES: src.to(device) at 995 and 998",
    "CROSS_LAYER_PHYSICAL_KV_REUSE": "NO: only IDs are predicted; next_cache[layer+1] supplies that layer's physical K/V",
    "FINAL_WORKING_SET_EQUALS_TRUE_CURRENT_Q_SELECTION": "YES",
    "PREFETCH_DESTINATION_STRUCTURE": "owner['next_layer_prefetch'] Python dict keyed by (target_layer, history_id), holding destination K/V tensors and a CUDA event",
    "PREFETCH_STREAM": f"created/stored at {src}:371-375",
    "CURRENT_LAYER_STREAM": "torch.cuda.current_stream(device), used by attention and wait_event",
    "NEXT_LAYER_STREAM": "same current compute stream; no separate next-layer compute stream",
    "execution_class": "TRUE_ASYNC_PREFETCH structurally, with avoidable host-control/allocation overhead",
    "root_cause_classification": "ASYNC path is semantically correct, but GPU→CPU ID conversion, Python per-block indexing, per-block allocation/pinning, waste traffic, and lack of durable reuse can erase the overlap benefit.",
    "evidence": {
        "predictor": f"{src}:366-416",
        "hit": f"{src}:418-440",
        "reconcile": f"{src}:354-364",
        "call_site": f"{src}:600-607",
        "correction": f"{src}:956-1057",
        "final_assembly": f"{src}:1093-1114",
    },
}

def write_json(name, data):
    (results / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")

write_json("group11_1_current_q_static_audit.json", current_json)
write_json("group11_2_qprev_static_audit.json", qprev_json)
write_json("group11_4_next_layer_prefetch_static_audit.json", prefetch_json)

(reports / "group11_1_current_q_static_audit.md").write_text(common + """
## Result

| Field | Result |
|---|---|
| CURRENT_Q_SOURCE_EXACT | YES |
| SELECTED_IDS_USED_DIRECTLY | YES |
| NO_QPREV_STATE | YES for this config |
| NO_NEXT_LAYER_PREDICTION | YES |
| NO_SPECULATIVE_FETCH | YES |
| NO_HOT_CACHE_REUSE | YES for this config |
| FINAL_WORKING_SET | sink + selected current-Q history + local/resident KV |

The selected GPU tensor returned by `_online_memory_indices` is passed through `memory_indices` into history assembly. The final attention operand is built from sink, selected history, and local/resident KV; no second selection is performed. The only host conversion in this path is for provenance and trace metadata.

## Static caveat

The codebase contains optional reuse and qprev/prefetch branches, but `configs/g11_1.yaml` does not enable them. Therefore they are not part of the Group11.1 execution graph.
""")

(reports / "group11_2_qprev_static_audit.md").write_text(common + """
## Result

| Field | Result |
|---|---|
| QPREV_STATE_LAYER_SPECIFIC | YES |
| QPREV_STATE_TIME_ALIGNED | NO / NOT_PROVABLY_EXACT |
| QPREV_RETRIEVAL_DIRECT | YES after bootstrap |
| CURRENT_Q_CORRECTION_PRESENT | NO |
| EXTRA_FETCH_BEYOND_SELECTED_IDS | NO |
| FINAL_WORKING_SET_EQUALS_QPREV_SELECTION | YES after bootstrap |

`configs/g11_2.yaml:18` enables `retrieval_query_mode: previous_q`. The previous query is stored per `kv_cache` with layer/chunk/denoising metadata. `_online_memory_indices` rejects a layer mismatch, then uses the stored query directly. The first retrieval for a cache has `BOOTSTRAP_CURRENT_Q`, so it is not literally q(t-1) on that first invocation.

The state is committed after attention, which prevents the current query from replacing the previous query before the current selection. However, the implementation checks the layer but does not prove that the stored state is exactly the immediately preceding temporal/timestep invocation. It also clones and stores the full query tensor, adding control and memory traffic. No second current-Q correction retrieval is present in this mode.
""")

(reports / "group11_4_next_layer_prefetch_static_audit.md").write_text(common + """
## Required fields

| Field | Result |
|---|---|
| CURRENT_Q_RETRIEVAL_PRESERVED | YES |
| PREDICTOR_EQUALS_CURRENT_SELECTED_IDS | YES |
| NEXT_LAYER_KEY_CORRECT | YES |
| SEPARATE_PREFETCH_AND_COMPUTE_STREAMS | YES |
| PREFETCH_SOURCE_PINNED | YES, conditionally enforced |
| PREFETCH_BLOCKS_CURRENT_LAYER | NO explicit CUDA wait; pre-launch CPU ID conversion is blocking host control |
| OVERLAP_WINDOW_STRUCTURALLY_EXISTS | YES |
| PREFETCH_GPU_DATA_ACTUALLY_REUSED | YES on exact hit |
| HIT_REUSES_PREFETCH_TENSOR_DIRECTLY | YES |
| HIT_REQUIRES_NEW_H2D | NO |
| CORRECTION_FETCH_EXACTLY_ONCE | YES per missing selected key |
| WASTE_NEVER_ENTERS_ATTENTION | YES |
| WASTE_POSTPROCESSING_PRESENT | NO, but waste H2D/allocation is paid |
| DUPLICATE_H2D_POSSIBLE | YES across later invocations after pop; no durable cache |
| GPU_TO_CPU_ID_SYNC_PRESENT | YES |
| PYTHON_ARCHIVE_INDEXING_PRESENT | YES |
| PER_PREFETCH_GPU_ALLOCATION | YES |
| CROSS_LAYER_PHYSICAL_KV_REUSE | NO |
| FINAL_WORKING_SET_EQUALS_TRUE_CURRENT_Q_SELECTION | YES |
| NEXT_LAYER_PREFETCH_EXECUTION_CLASS | TRUE_ASYNC_PREFETCH, with avoidable overhead |
| ROOT_CAUSE_CLASSIFICATION | host control + allocation/pinning + waste/limited reuse |

## Actual execution graph

```text
Q_l
 -> _online_memory_indices(Q_l) -> I_l
 -> _reconcile_prefetch(l, I_l)
 -> _schedule_next_layer_prefetch(l+1, I_l)
      -> selected.detach().cpu().tolist()
      -> Python next_cache[history_id]
      -> optional pin_memory
      -> prefetch_stream: src.to(device, non_blocking=True)
      -> event and state[(l+1, history_id)] = {dst K,V,event}
 -> current layer selected-K/V assembly and BF16 attention
 -> next layer computes Q_(l+1), retrieves I_(l+1)
 -> reconcile predicted state
 -> each true ID: exact hit waits on event and returns prefetched tensor;
    missing ID performs one demand correction H2D
 -> stack/cat with sink/local KV
 -> BF16 attention over I_(l+1), never over stale prediction
```

## Timing expectation without benchmarking

| Cost | Static class | Reason |
|---|---|---|
| Current retrieval | UNAVOIDABLE | true current-Q selection still runs every layer |
| Prefetch H2D | HIDEABLE | separate stream and event exist |
| Prefetch host indexing | AVOIDABLE_OVERHEAD | CPU list conversion and Python loop precede copies |
| Prefetch allocation | AVOIDABLE_OVERHEAD | each destination uses `.to(device)` |
| Hit lookup/wait | UNAVOIDABLE / HIDEABLE | exact key lookup plus stream event wait |
| Correction H2D | UNAVOIDABLE for misses | demand path at 995/998 |
| Waste H2D | AVOIDABLE_OVERHEAD | stale prediction is copied before `_reconcile` at next layer can discard it |
| Current-layer synchronization | NOT_PRESENT as explicit CUDA wait | no `wait_event` before current attention |
| Next-layer synchronization | UNAVOIDABLE on an unready hit | `current.wait_event` at 426-429 |

The prefetch can beat CURRENT_Q only if exposed demand-fetch time saved exceeds correction-fetch time, prefetch host/allocation overhead, and resource contention. Static structure alone cannot establish a speedup.
""")

comparison = """# Group11 non-FlashFetch algorithm consistency

Static-only audit; no inference or benchmark was run. Source snapshot: `{head}`.

| Strategy | Retrieval query | Prediction | Final attention selection | Main extra work | Static semantic status |
|---|---|---|---|---|---|
| 11.1 CURRENT_Q | current Q_l | none | I_l directly + local/resident | provenance CPU ID conversion | REFERENCE: YES |
| 11.2 PREVIOUS_Q | stored Q from same layer's prior cache state; first use bootstraps current Q | none | I'_l directly + local/resident | detach/clone/persist query state; exact t-1 not proven | DIRECT QPREV after bootstrap, temporal alignment caveat |
| 11.4 NEXT_LAYER_PREFETCH | current Q_l for current layer and again current Q_(l+1) next layer | P_(l+1)=I_l | true I_(l+1), not prediction | CPU ID list/Python indexing, per-block pin/alloc/H2D, event wait, correction/waste handling | TRUE_ASYNC_PREFETCH structurally |

## Answers to final questions

1. **Does Group11.1 exactly implement current-Q DraftMap retrieval?** Yes for its enabled configuration; selected IDs are passed directly into the final working-set assembly.
2. **Is Group11.2 only replacing Q_t with Q_(t-1)?** Algorithmically it adds the intended previous-query state, but also adds clone/persistence/metadata overhead and a current-Q bootstrap; exact temporal predecessor alignment is not enforced.
3. **Does Group11.4 preserve current-Q retrieval for current and next layer?** Yes. It predicts from I_l, but recomputes I_(l+1) using Q_(l+1).
4. **Is prefetch launched early enough?** Structurally yes: after current retrieval and before current-layer history assembly/attention. The launch itself is preceded by host ID conversion.
5. **Separate CUDA stream?** Yes, a dedicated `torch.cuda.Stream`; compute uses the current stream.
6. **Direct hit consumption?** Yes, the exact prefetched K/V tensors are returned from the state entry.
7. **Can a hit trigger another H2D?** Not within the same retained state entry; after the entry is popped, later independent invocations have no durable cache and may fetch again.
8. **Correction fetch exactly once?** The visible selected-ID loop issues one correction for each missing `(layer, history_id)` key; top-k selection is unique.
9. **Waste postprocessing?** No RoPE/repack before discard, but stale predictions already paid pinning, allocation, and H2D.
10. **GPU→CPU sync/Python indexing?** Yes: selected IDs are converted with `.detach().cpu().tolist()` and used to index Python CPU lists.
11. **Physical K/V reused across layers?** No; only IDs cross layers, while target-layer archive supplies target-layer K/V.
12. **Structurally capable of beating CURRENT_Q?** Yes, conditionally: a separate prefetch stream and event wait create a plausible hideable interval.
13. **What may prevent a win?** CPU ID synchronization, Python per-block control, per-prefetch allocation/pinning, stale/wasted H2D, correction traffic, and no durable cache after hit consumption. These are structural cost sources, not proof of a measured slowdown.

## Important non-conclusions

This audit does not claim a timing win, does not claim exact qprev temporal alignment, and does not claim that all H2D is hidden. Those require a matched runtime benchmark with explicit exposed/hidden event accounting.
""".format(head=head)
(reports / "group11_non_flashfetch_algorithm_consistency.md").write_text(comparison)

print("wrote static Group11 audit reports and JSON")
