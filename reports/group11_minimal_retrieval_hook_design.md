# Minimal retrieval-policy hook design

## Proposed interface

```python
selected_ids = retrieval_policy.select(
    current_query=current_q,
    history_index=canonical_history_index,
    recent_exclude=recent_exclude,
    budget=memory_size,
)  # torch.int64, [B, k], device may be GPU
```

The callback is invoked where the original pipeline currently constructs
`memory_indices` (`pipeline/causal_inference.py:267-304`). The existing LongLive
consumer remains responsible for CPU/archive lookup, H2D, stack/view, RoPE,
concatenation, BF16 attention, and cache bookkeeping.

## Required sidecar

Maintain `draft_history_id -> canonical_history_id` and append both records at
the same cache commit. The selector may score Draft-K records, but it must return
canonical IDs or an explicit adapter must map them. Assert identical append order,
eviction order, sink exclusion, recent exclusion, batch semantics, and candidate
count before fetch.

## DraftMap placement

- Draft-Q: transient current-Q pooling, no change to attention Q.
- Draft-K: persistent GPU BF16 sidecar only.
- DraftMap: mean-pool 64-token blocks, float score, softmax, aggregate by
  chunk, top-k.
- Full K/V: original LongLive ownership and fetch path.

## Non-goals

No new attention wrapper, no new `torch.cat`, no new RoPE, no new attention
backend, no CPU Draft-K, no low-bit kernel.

## Estimated impact

The current wrapper span is roughly 550 source lines; roughly 250-330 lines are
reimplemented LongLive dataflow and could be bypassed by delegation. Draft-Q,
Draft-K, scoring, provenance, and cache-sidecar bookkeeping remain. Runtime
benefit is expected primarily in fetch/working-set/allocation/synchronization
orchestration; DraftMap cost remains.

## Validation and rollback

Run gates in the trace report: selected-ID equality, Q/K/V equality, attention
output equality, case01 output equality, then matched block 6-7 profiling. Keep
the current wrapper as a comparison path until all gates pass.
