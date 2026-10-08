# Group11.2 previous-Q temporal prefetch implementation

## Exact runtime graph

```text
same-layer invocation t-1:
  Q_(t-1,l)
    -> DraftMap selection P_(t,l)
    -> CPU archive ID materialization
    -> dedicated prefetch stream
    -> reusable GPU staging slots + completion event
    -> remaining output/FFN work

same-layer invocation t:
  lookup exact temporal key
    -> wait on copy event only if needed
    -> consume prefetched K/V
    -> dense stock FlashAttention
    -> persist Q_(t,l)
    -> DraftMap selection P_(t+1,l)
    -> async prefetch for t+1
```

The initial bootstrap is explicit and is not counted as a qprev-prefetch hit.
Non-bootstrap missing or stale state raises in validation mode; it does not
silently run qprev direct retrieval or current-Q correction retrieval.

## Required implementation fields

| Field | Value |
|---|---|
| `OLD_MODE_PRESERVED` | YES |
| `NEW_MODE_NAME` | `previous_q_prefetch` |
| `EXACT_Q_T_MINUS_1_ALIGNMENT` | PASS in v8 smoke |
| `PREFETCH_TARGET_SAME_LAYER` | YES |
| `PREFETCH_TARGET_NEXT_TEMPORAL_INVOCATION` | YES, next valid temporal invocation |
| `PREFETCH_LAUNCHED_DURING_PREVIOUS_INVOCATION` | YES; after source attention, before output projection/FFN |
| `DEDICATED_PREFETCH_STREAM` | YES |
| `CPU_SOURCE_PREPINNED` | YES, validated by source contract |
| `REUSABLE_GPU_STAGING` | YES |
| `CURRENT_INVOCATION_QPREV_RETRIEVAL` | NO in steady state |
| `CURRENT_INVOCATION_H2D_ON_VALID_HIT` | NO |
| `STALE_PREFETCH_REJECTED` | YES |
| `FINAL_WORKING_SET_EQUALS_QPREV_SELECTION` | YES in smoke |
| `CURRENT_Q_CORRECTION_PRESENT` | NO |
| `STOCK_DENSE_FLASHATTN` | YES |
| `BUFFER_REUSE_SAFE` | YES; consumer CUDA event guards reuse |

## Answers

1. Yes. The old implementation used previous-Q for selection but fetched on the current invocation.
2. Yes. The new mode performs the prediction from `Q_(t-1,l)` during the preceding valid invocation.
3. Yes. H2D is launched before the target invocation begins.
4. Yes. The steady-state prediction is generated from the immediately preceding same-layer Q.
5. Yes. The target is the next valid temporal invocation at the same layer.
6. Yes. A valid target consumes the stored prediction without rerunning qprev retrieval.
7. Yes. The smoke recorded zero demand H2D on valid prefetch hits.
8. Yes. Bootstrap is explicit and separate.
9. Yes. Target/layer metadata is validated and stale entries are rejected.
10. Yes. The final working set is the qprev prediction plus the unchanged sink/local components.
11. Yes. The normal dense FlashAttention backend is unchanged.
12. Yes. The prefetch is launched after current attention and before output projection/FFN work.
13. The corrected mechanism matches the intended Group11.2 temporal-prefetch principle in the minimal smoke.

No speedup claim is made; this was a correctness smoke only.
