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
