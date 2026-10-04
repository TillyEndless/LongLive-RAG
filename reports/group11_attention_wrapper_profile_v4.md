# Group11 attention-wrapper profile v4

Diagnostic-only case01. No production algorithm, retrieval decision, canonical10 output, quality result, or Group12–15 path was modified.

## Boundary

- `WRAPPER_ENTRY`: `/data/zxl/LongLive-RAG-group11_15_h200/wan/modules/causal_model_latentmem.py`, `CausalWanSelfAttention.forward`, wrapper region beginning at line 515.
- `WRAPPER_EXIT`: same function, after attention output projection around line 689. The historical `ATTENTION_WRAPPER` label was not a literal source symbol; it is the model phase timer around this region.
- `INNER_BF16_ATTN`: `/data/zxl/LongLive-RAG-group11_15_h200/wan/modules/attention.py:139`, called at `causal_model_latentmem.py:663-668`.

## Decomposition

- ATTENTION_WRAPPER_HOST_TIME: **77.948 s**
- fetch end-to-end host: **13.461 s**
- host K/V `.to(cuda)`: **0.842 s**
- selected working-set stack/gather: **0.188 s**
- K/V cat: **0.115 s**
- BF16 attention host wrapper: **1.926 s**
- exclusive accounted by these timers: **15.387 s**
- exclusive unattributed: **62.561 s**

ATTENTION_WRAPPER_COVERAGE < 90%: **19.74%** of the wrapper is covered by the currently separated timers. The remaining `62.561 s` is not assigned to cat, layout, mask, RoPE, sync, allocator, or other phases without a valid CUDA/host timeline.

## Call statistics

See `group11_wrapper_calls_v4.csv`. The trace contains 5,220 historical wrapper fetch rows and 6,000 BF16 attention calls. Source pinning was recorded by the underlying trace; explicit CUDA event timings for each copy were not captured in this pass.

## Control and Nsight

The prior LongLive-RAG wrapper control forced synchronization after every `.to()` and is explicitly not reused for causal comparison. The published 61.2 s value is retained only as context. Nsight was available but could not write its report because `/` had zero free space; no unknown files were deleted.

## Root-cause ranking

1. **Unattributed wrapper/framework time**: measured residual above; not safe to classify further.
2. **Historical working-set materialization/fetch path**: measured by 13.461 s end-to-end, including 0.115 s cat and 0.188 s stack/gather.
3. **BF16 attention host wrapper**: 1.926 s; the inner kernel time is not separately available in this pass.

No production optimization is recommended from this incomplete decomposition.
