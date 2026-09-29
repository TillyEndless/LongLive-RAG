# Group11–15 final semantic alignment audit

## Conclusion

The implementation is aligned with the requested ownership and numerical
semantics for Groups 11–15. A fresh v6 real multi-chunk smoke completed
Groups 12, 13, 14 and 15, each with 10 generation blocks and no exception.
No canonical10 inference or quality evaluation was run.

The only remaining non-equivalence is an implementation-performance detail:
Group14/15 currently perform the historical fetch wait, then Q-side sparse
selection, then a separate local-promotion fetch wait. This is semantically
correct but is not a single unified wait or proven overlap implementation.

## Group semantics

| Group | Verified behavior |
|---|---|
| 11.1 | Current Q uses GPU-resident Draft-K/latent history descriptors; selected historical IDs fetch authoritative CPU BF16 K/V; retrieved K/V are appended to the local candidate set; final attention is BF16. |
| 11.2 | `previous_q` selects history and schedules same-layer CPU BF16 K/V prefetch before the consumer; the consumer waits on the event if needed. |
| 11.3 | Current-Q online chunk fetch uses a chunk pipeline: next chunk H2D is submitted before the current chunk is consumed; online softmax accumulation preserves attention semantics. |
| 11.4 | `next_layer_prefetch` predicts the next layer's IDs and schedules its CPU BF16 K/V transfer; the next layer consumes matching IDs and corrects misses. |
| 12 | Same retrieval/promotion contract as 11.1; local window persistent owner is K INT8/V FP8 E4M3; new projected K/V enter the low-bit store; attention receives call-local BF16 materialization. |
| 13 | Same as 12 with persistent local K/V NVFP4; attention remains call-local BF16 after dequantization. |
| 14 | Group12 plus Q-side sparse routing over retrieved-history + local-window candidates; only retained candidates enter attention; local promotion is applied only within retained local IDs. |
| 15 | Group13 plus the same Q-side sparse routing and retained-local promotion rules. |

## Static evidence

- Unified history fetch: `wan/modules/causal_model_latentmem.py::_v2_unified_fetch`.
- Local promotion fetch: `wan/modules/causal_model_latentmem.py::_v2_promotion_fetch`.
- DraftMap retrieval: `wan/modules/causal_model_latentmem.py::_online_memory_indices`.
- Q sparse routing: `utils/persistent_draftmap.py::route_candidate_chunks`.
- Local low-bit owner: `utils/local_lowbit_kv.py::LocalLowbitKVStore`.
- Local low-bit insertion: causal cache `roll_and_insert`/`direct_insert` update paths.
- Local attention materialization: `LocalLowbitKVStore.materialize_frames`.
- Stock BF16 attention is still called after candidate construction; no native
  low-bit attention kernel is invoked.

The local store does not allocate the legacy dense GPU `k`/`v` tensors for
Groups 12–15. Persistent local ownership is packed low-bit plus scales and
metadata. BF16 tensors are call-local. On the first call, uncommitted new
frames use the same quantize/dequantize rule as a bootstrap fallback and are
not retained as a second owner.

## Configuration checks

- Group12/13: `q_sparse_ratio=0.0`; `local_kv_promotion_ratio` is explicit.
- Group14/15: `q_sparse_ratio=0.5` in the smoke config and an independent
  `local_kv_promotion_ratio=0.2`.
- Group14/15 sparse scope is `local_plus_retrieved_history`.
- Group14/15 promotion denominator is retained local candidates, not all
  history. The current chunk remains mandatory in sparse routing.
- Query remains BF16; final attention dtype is BF16.
- Native low-bit attention kernel is disabled.
- Historical persistent GPU KV owner is `NONE`; historical full BF16 K/V live
  in the CPU archive and are fetched when selected.

## Validation

| Check | Result |
|---|---|
| Group12 v2 smoke | PASS |
| Group13 v2 smoke | PASS |
| Group14 v2 smoke | PASS |
| Group15 v2 smoke | PASS |
| Static tests | PASS |
| Python compilation | PASS |
| Canonical10/evaluation | NOT RUN |
| Group14/15 single-wait overlap | NOT CLAIMED |

Fresh smoke log:
`/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/results/group11_15_final_semantic_alignment_smoke_v6/runner_v6.log`

Fresh smoke output root:
`/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/results/group11_15_final_semantic_alignment_smoke_v6/`

The fresh runtime metadata reports, for Groups 12–15:

- `FINAL_ATTENTION_DTYPE=bfloat16`
- `GPU_PERSISTENT_LOWBIT_OWNER=YES`
- `GPU_HISTORICAL_PERSISTENT_RESIDENT_KV_BYTES=0`
- Group12/14 owner `LOCAL_INT8/FP8_E4M3`
- Group13/15 owner `LOCAL_NVFP4/NVFP4`
- Group14/15 `INTERACTION_SPARSE_ROUTING=YES`

## Files changed in this pass

- `utils/local_lowbit_kv.py`: call-local frame materialization from the
  persistent low-bit owner; promoted frames bypass low-bit dequantization;
  bootstrap frames use transient quantize/dequantize.
- `wan/modules/causal_model_latentmem.py`: v2 local candidate construction
  now reads the low-bit store rather than the BF16 temporary cache; promotion
  is overlaid only for the selected/promoted IDs.

No checkpoint, dataset, model architecture, or old canonical result was
modified.
