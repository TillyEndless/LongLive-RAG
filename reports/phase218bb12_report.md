# Phase BB-1.2 — Minimal Fix and Correctness Gate Report

Worktree: `/data/zxl/LongLive-RAG-group11_1-w1-phase218bb`  
Source base: `410948eb7e328513e7751ac2d44193115363cee4`  
Generated: `2026-10-04T14:09:43.258805+00:00`

## Changes applied

1. `pipeline/causal_inference.py`: removed the duplicated
   `physical_kv_cache_frames` override. The optional Physical4 override is now
   applied exactly once after the original local/global policy selection.
2. `wan/modules/causal_model_latentmem.py`: the actual runtime
   `frame_seqlen` is recorded in the cache and in both `cache_update_info`
   variants. `_apply_cache_updates()` consumes that metadata and uses it for
   rolling/draft accounting and logical eligibility. It no longer relies on a
   hard-coded 1560 or an undefined local variable.

The AO W1 helper, retrieval policy, CPU archive design, Native FA2 path and
post-attention join were not replaced.

## Verification

```text
PY_COMPILE = PASS
BB12_CACHE_UPDATE_RUNTIME_FRAME_TEST = PASS
D2H_EVENT_LIFETIME_SMALL_CUDA_TEST = PASS
```

The cache-update test called the real `_apply_cache_updates()` method with a
2-token frame geometry, exercised direct insertion, then exercised the
`update_info=None` path using cached runtime frame metadata. This specifically
guards against the former undefined `frame_seqlen` failure.

The bounded CUDA D2H test used a pinned CPU destination, recorded a CUDA event
after the copy, observed the event incomplete before synchronization, and
verified the destination contents after the event completed. This validates
the copy/event behavior in isolation; it is not a substitute for a full model
eviction lifecycle capture.

## Correctness gate status

```text
MATCHED_ARCHIVE_PARITY = NOT_MEASURED
REAL_RETRIEVAL_PARITY = NOT_MEASURED
REAL_ROPE_PARITY = NOT_MEASURED
FINAL_KV_PARITY = NOT_MEASURED
NATIVE_FA2_OUTPUT_PARITY = NOT_MEASURED
F3_SMOKE = NOT_RUN
F120 = NOT_RUN
UNIFIED_EVALUATION = NOT_RUN
READY_FOR_F3 = NO
```

The BB worktree contains no authoritative AO/BB real committed-state capture
with global frame IDs, matching DraftMap state, or matching Q/K/V operands.
The earlier BB-1.1 report explicitly recorded these gates as unmeasured, and
the available result tree contains only static checks and provenance files.
Therefore no Archive, RoPE, final-K/V, or stock-FA2 parity result is claimed.
F3 and F120 are intentionally not launched.

## D2H interpretation

The inherited expression is:

```python
f.to("cpu", non_blocking=True).contiguous().pin_memory()
```

The isolated test observed a pinned destination and correct event-ordered
contents while the source remained alive. This establishes bounded copy
readiness behavior, but does not prove the complete model path has no exposed
host synchronization or premature source reuse. That full lifecycle remains
unmeasured and is a required prerequisite for performance testing.

## Source hashes

See `results/phase218bb12_status.json` for the post-fix SHA256 values and
`patches/phase218bb12_minimal_fix.patch` for the exact source transformation.
