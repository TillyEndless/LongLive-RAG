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
