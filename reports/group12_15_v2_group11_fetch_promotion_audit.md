# Group12–15 v2 Group11 Fetch/Promotion Audit (before alignment)

Repository: `/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2`
Base commit: `ae1c2faa8139a065139fbc0d4164565c64706e69`
Canonical10: not run.

## Frozen ownership contract

The v2 local low-bit owner is present in `utils/local_lowbit_kv.py`. The CPU
history archive is `utils/compressed_history_archive.py`. The intended contract
is GPU persistent local low-bit KV, CPU BF16 authoritative history, transient
BF16 materialization, and `GPU_HISTORICAL_PERSISTENT_KV_BYTES=0`.

## Current call path

| Stage | Current source location | Finding |
|---|---|---|
| Q/K/V creation | `wan/modules/causal_model_latentmem.py:486-500` | `qkv_fn()` creates Q/K/V, then `_queue_draft_q_history()` and `_online_memory_indices()` run. |
| Historical DraftMap scoring | `wan/modules/causal_model_latentmem.py:192-240` | `DraftMapChunkIndex.score_history()` scores persistent GPU Draft-K against current/previous Q; `select_topk()` produces history IDs. |
| HISTORY_SELECTED_IDS | `wan/modules/causal_model_latentmem.py:256-270` | IDs are copied to CPU for trace serialization; selection tensor itself remains available. |
| Local low-bit materialization | `wan/modules/causal_model_latentmem.py:178-190`, `709-745` | `_materialize_local_cache()` decodes the whole resident local store before attention and RoPE. |
| Local promotion selection | `wan/modules/causal_model_latentmem.py:1603-1618` | Existing code uses local Draft-K magnitude and calls `store.promote_top()`. Before alignment this converts selected low-bit slots to persistent BF16 slots. |
| CPU BF16 history gather | `wan/modules/causal_model_latentmem.py:801-823` | Existing path loops through selected IDs and calls `archive.fetch()` per ID. |
| History H2D | `utils/compressed_history_archive.py:84-96` | Each `fetch()` issues separate K and V `.to(device, non_blocking=True)` calls, but caller consumes them immediately. |
| CPU BF16 promotion gather | not implemented | No CPU-local BF16 archive/list is maintained for current local slots. |
| Synchronization/wait | `wan/modules/causal_model_latentmem.py:816-924` | Fetches are consumed in the same loop; there is no single unified fetch handle/barrier for history and promotion. |
| Low-bit non-promoted dequant | `utils/local_lowbit_kv.py:60-69`, `109-121` | Dequantizes local slots into call-local BF16 tensors; however prior promotion altered ownership. |
| Final K/V assembly | `wan/modules/causal_model_latentmem.py:931-967` | Historical and local tensors are concatenated, then runtime metadata is applied. |
| BF16 attention | `wan/modules/causal_model_latentmem.py:972-980` | Stock `attention(roped_query, k_cat, v_cat)` remains the final attention call. |
| Transient release/reuse | Python reference lifetime after attention | No explicit unified-handle release/reuse contract exists. |

## Before-alignment classification

- History retrieval: **serialized per selected chunk**.
- Local promotion fetch: **not implemented from CPU BF16**; instead it is a persistent BF16 conversion inside the GPU local store.
- History and promotion: **not planned together**.
- H2D copies: technically marked non-blocking, but immediate Python consumption and per-record calls expose serialized waits.
- Group12/13: history-only behavior exists but uses the same serial per-record fetch loop.
- Group14/15: Q-sparse and local promotion paths are separate; promotion does not use the same current-Q unified fetch plan.

## Required changes

1. Add a shared fetch-plan abstraction separating selection, submission, wait, and assembly.
2. Keep history IDs and local promotion IDs logically separate but submit them before a single consumption wait.
3. Maintain CPU BF16 local authoritative copies for transient promotion overlays.
4. Make promotion non-persistent: local low-bit owner must remain unchanged after attention.
5. Preserve the v2 ownership invariant: no persistent GPU historical KV owner.
6. Add separate history/promotion bytes, calls, physical-copy count, CUDA work, and exposed wait accounting.
7. Add static semantic tests and one minimal Group14/15 serial-vs-unified smoke only.

This report is the pre-change audit; the post-change report is
`reports/group12_15_v2_fetch_alignment.md`.
