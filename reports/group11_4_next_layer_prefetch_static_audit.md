# Group11 static implementation audit

Audited worktree: `/data/zxl/LongLive-RAG-group11_15_h200`
Audited HEAD: `410948eb7e328513e7751ac2d44193115363cee4` (the worktree also contains unrelated uncommitted experimental files; this audit is source-level only).

No inference, benchmark, or code modification was performed for this audit.

## Common execution points

- `_online_memory_indices`: `wan/modules/causal_model_latentmem.py:248-343`; scores DraftMap candidates and returns selected history IDs.
- `forward`: `wan/modules/causal_model_latentmem.py:550-607`; computes Q/K/V, invokes retrieval, and passes `memory_indices` into the history assembly path.
- History assembly/final working set: `wan/modules/causal_model_latentmem.py:930-1114`; selected historical K/V, sink, and local/resident K/V are assembled before unchanged BF16 attention.
- Final attention: `wan/modules/causal_model_latentmem.py:1122-1154`; `prepare_attention_kv` is followed by `attention(roped_query, k_cat, v_cat)`.

The implementation still records selected IDs with `selected.detach().cpu().tolist()` for provenance. That is a host-side control/observability synchronization; it is not used to change the returned GPU selection tensor for the normal fetch path.

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
