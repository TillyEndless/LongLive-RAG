# Group11 case01 latency root-cause profile v3

## Scope and run identity

- Group11 W12 case01 semantics were not modified; this is a profiling rerun in the v3 primary directory.
- The reuse run is diagnostic only and is not a final Group11 result.
- LongLive-RAG control used the same W12 case01 prompt/config family, but its legacy profiling wrapper forces CUDA synchronize after every tensor to; its measured wall is therefore reported but not used as a causal wall comparison.

## Complete host-wall retrieval measurement (Group11)

- fetch_lookup_host_s = 0.003618 s
- fetch_k_to_device_host_s = 0.487234 s
- fetch_v_to_device_host_s = 0.354605 s
- working_set_gather_host_s = 0.188140 s
- working_set_cat_host_s = 0.114604 s
- fetch_e2e_host_s = 13.460508 s
- cuda_h2d_s = 0.841838 s
- fetch rows = 5220; full-KV H2D bytes = 290,127,052,800

## Coarse model phases (measured host timers)

- qkv_projection_s = 7.253791 s
- norm_rope_s = 0.324657 s
- attention_wrapper_s = 77.948116 s
- attention_output_projection_s = 2.541961 s
- ffn_mlp_s = 11.681696 s
- cache_update_s = 0.000000 s
- scheduler_bookkeeping_s = 0.000000 s

Phase timers are nested in the attention wrapper where noted; they must not be blindly summed as an exclusive partition.

## Strict delta table

| Metric | Group11 | LongLive-RAG control | Delta |
|---|---:|---:|---:|
| total wall | 203.761124 s | 471.852961 s | -268.091836 s |
| DraftMap | 6.500905 s | NOT_AVAILABLE | NOT_AVAILABLE |
| fetch end-to-end host | 13.460508 s | 0.061027 s | 13.399482 s |
| CUDA H2D | 0.841838 s | 32.537414 s | -31.695576 s |
| working-set build | 0.302744 s | 9.743746 s | -9.441001 s |
| BF16 attention | 1.926196 s | 61.295982 s | -59.369785 s |
| QKV projection | 7.253791 s | NOT_AVAILABLE | NOT_AVAILABLE |
| FFN/MLP | 11.681696 s | NOT_AVAILABLE | NOT_AVAILABLE |
| norm/RoPE | 0.324657 s | NOT_AVAILABLE | NOT_AVAILABLE |
| cache update | 0.000000 s | NOT_AVAILABLE | NOT_AVAILABLE |
| scheduler/bookkeeping | 0.000000 s | NOT_AVAILABLE | NOT_AVAILABLE |

Control caveat: the legacy LongLive-RAG profiler synchronizes each transfer, so its 471.853 s wall is instrumentation-distorted. The prior published non-instrumented W12 reference of 61.2 s is context only, not substituted into the strict table.

## Retrieved-KV reuse diagnostic

- CACHE_HITS = 29,351, CACHE_MISSES = 919, CACHE_HIT_RATE = 0.969640
- AVOIDED_H2D_CALLS = 58,702, AVOIDED_H2D_BYTES = 281,318,768,640, GPU_CACHE_PEAK_BYTES = 507,985,920
- untouched Group11 wall = 203.761 s; reuse wall = 188.220 s; delta = -15.541 s
- selected-ID digest unchanged = True

## Coverage against the prior non-instrumented reference

- Group11 measured wall = 203.761 s (config-to-runtime artifact interval).
- Prior published LongLive-RAG W12 reference = 61.200 s; wall delta = 142.561 s.
- Sum of measured Group11 sub-timers is 119.712 s, or 84.0% of that contextual delta. This is an upper-bound coverage because several timers are nested, not an exclusive attribution.
- The strict aligned control cannot provide a valid positive wall delta because its legacy wrapper adds per-transfer synchronization. Remaining causal uncertainty is reported rather than attributed to repeated fetch or model compute without an uninstrumented phase-aligned control.

## Conclusion

The reuse run tests whether repeated exact historical KV transfers are avoidable while keeping selected IDs unchanged. It is diagnostic evidence only. Host-wall fetch and model-phase timings are separated from CUDA transfer timing; no claim is made that either repeated fetch or model compute alone is the root cause until control-wrapper overhead is removed from the comparison.
