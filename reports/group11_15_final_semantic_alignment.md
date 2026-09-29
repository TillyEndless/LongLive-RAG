# Group11–15 final semantic alignment audit

## Scope

This pass performed source/static alignment and a real multi-chunk 10-block smoke for Groups 12–15. It did **not** run canonical10, quality evaluation, or modify old result directories. The smoke used one existing case prompt and produced new smoke artifacts only.

## Gate fields

| Field | Result | Evidence |
|---|---|---|
| CURRENT_CHUNK_FAKE_QUANT_ALIGNED | YES | `wan/modules/causal_model_latentmem.py:727-735,808-817`; smoke runtime traces report fake K/V quantization and BF16 final attention. |
| GROUP14_FULL_CANDIDATE_SPARSE | YES | `wan/modules/causal_model_latentmem.py:1040-1062` builds retrieved-history + local candidate set and routes it with `route_candidate_chunks`; rerun metadata reports `SPARSE_SCOPE=local_plus_retrieved_history`, `Q_SPARSE_RATIO=0.5`. |
| GROUP15_FULL_CANDIDATE_SPARSE | YES | Same path as Group14, with NVFP4/NVFP4 local owner; rerun metadata confirms v2 sparse active. |
| G12_G14_PROMOTION_IDENTICAL | YES* | Same local promotion owner/ratio and BF16 overlay mechanism; Group14 applies it after sparse retention by design. *The denominator is retained local candidates for sparse groups. |
| G13_G15_PROMOTION_IDENTICAL | YES* | Same qualification as above, with NVFP4 local owner. |
| SPARSE_THEN_PROMOTION | YES | Group14/15 path routes the full candidate set first, then promotes only retained local IDs (`causal_model_latentmem.py:1055-1089`). |
| GPU_HISTORICAL_PERSISTENT_KV_BYTES | 0 | Smoke memory JSON: `GPU_HISTORICAL_PERSISTENT_RESIDENT_KV_BYTES=0`; only local low-bit KV is persistent. |
| GROUP11_2_TRUE_PREFETCH_OVERLAP | NOT_REVALIDATED_THIS_TURN | No new Group11.2 timeline benchmark was run. |
| GROUP11_3_UNCHANGED | YES | No Group11.3 source changes in this pass. |
| GROUP11_4_REAL_SMOKE_PASS | NOT_REVALIDATED_THIS_TURN | No new Group11.4 smoke was run. |
| GROUP11_4_TRUE_NEXT_LAYER_OVERLAP | NOT_REVALIDATED_THIS_TURN | No new overlap benchmark was run. |
| MEMORY_ACCOUNTING_ALIGNED | YES | Smoke JSON separates persistent local owner, draft bytes, historical resident bytes, and transient dequant peak. |
| STATIC_TESTS_PASS | YES | `v2_static_test.py` and `v2_fetch_plan_static_test.py` both passed; Python compilation passed. |
| REAL_MULTI_CHUNK_SMOKE_PASS | YES | Groups 12, 13, 14, 15 each completed 10 generated blocks without exception; Group14/15 were rerun after metadata correction. |
| CANONICAL10_RUN | NO | Explicitly not run. |

## Source-level dataflow

- Unified history fetch entry: `wan/modules/causal_model_latentmem.py:209-244` (`_v2_unified_fetch`).
- Local promotion fetch: `wan/modules/causal_model_latentmem.py:246-260` (`_v2_promotion_fetch`).
- Current-window low-bit insertion occurs before attention in the roll/direct cache paths around `:727-735` and `:808-817`.
- Candidate construction concatenates retrieved history and resident local window at `:1036-1050`.
- Group14/15 chunk routing is `utils/persistent_draftmap.py:68-128`, called at `:1055-1062`; the current-Q score is now also reused for promotion ranking at `:1066-1075`.
- Promotion overlays CPU BF16 K/V only on retained local IDs at `:1079-1089`; non-promoted retained local IDs remain GPU low-bit and are materialized to BF16 by the existing attention preparation path.
- Final attention passes BF16 tensors through `prepare_attention_kv` and the unchanged stock attention call at `:1122-1140`.
- Runtime metadata writer: `inference.py:391-429`; v2 sparse fields were corrected so they no longer appear as legacy `NO/0.0`.

## Smoke evidence

The machine-readable details are in `results/group11_15_final_semantic_alignment_smoke.json`. The rerun metadata for Group14 and Group15 records:

- `INTERACTION_SPARSE_ROUTING=YES`
- `Q_SPARSE_RATIO=0.5`
- `LOCAL_KV_PROMOTION_RATIO=0.2`
- `SPARSE_SCOPE=local_plus_retrieved_history`
- `FINAL_ATTENTION_DTYPE=bfloat16`
- `NATIVE_LOWBIT_KERNEL_USED=NO`
- `GPU_HISTORICAL_PERSISTENT_OWNER=NONE`

The prior smoke metadata had stale legacy field names for v2; those fields were fixed in code and Group14/15 smoke was rerun.

## Important remaining limitation

The current implementation is semantically ordered as sparse selection then local promotion, but Group14/15 currently perform the history fetch wait and the promotion fetch wait as two runtime waits. This is correct for ownership/ID semantics, but it is **not** evidence of a single unified one-wait overlap implementation for the sparse path. The report therefore does not claim `UNIFIED_SINGLE_WAIT=YES` or any speedup. A separate plan-level fusion is still needed if the requirement is strict “one FetchPlan/one wait” for sparse history plus local promotion.

No quality metrics or canonical10 results should be interpreted from this smoke.

## Reproducibility

- Branch: `group12-15-local-lowbit-v2`
- Base/source commit before this uncommitted pass: `ddfe1b6e39a63d61d16f2858231ba43abcdf73df`
- Output: `/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/results/group11_15_final_semantic_alignment_smoke.json`
- Canonical10: **not run**
