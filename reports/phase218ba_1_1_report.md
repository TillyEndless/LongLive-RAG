# Phase BA-1.1 — Targeted Fixes and Strict Attention Parity

## Scope and isolation

This is an incremental continuation of BA-1 at `/data/zxl/LongLive-RAG-group11_1-w1-phase218ba`. AO, AY, AZ-W and production FA2 were not modified. The BA config now writes only to:

`/data/zxl/LongLive-RAG-group11_1-w1-phase218ba/results/w1_4frame/case_01`

The config remains baseline/uncompressed W20/R16/S1 with `physical_kv_cache_frames: 4`.

## Fixes applied

1. Added the dedicated BA output folder; the reviewed BA YAML no longer points at AO.
2. Added an explicit rejection for `physical_kv_cache_frames < local_attn_size` when compressed-history mode is enabled. Compressed logical promotion is intentionally out of scope.
3. Replaced promotion's hard-coded `1560` with the cache's existing `frame_seq_length` field.
4. Reworked the actual cache replay to decode frame identity from sentinel values stored in the actual K/V tensors after every committed block. It separately records archive K/V, pending K/V, local/sink K/V, Draft-K, visible candidates and recompute stability.
5. Added a CUDA Event after each eviction batch's nonblocking D2H copies. The event is synchronized before that batch is promoted to the retrieval-visible archive. This is an explicit batch-level host-read readiness point, not a per-frame blocking `.to("cpu")` operation.

## Actual replay result

The test used 16 three-frame commits and real BF16 K/V tensors whose first value encoded the global frame ID. It exercised first physical eviction, first logical W20 eviction, multiple mature rolls and no-update recompute calls.

The authoritative actual replay result is `results/phase218ba_gate_11.json`, under `actual_cache_update_replay`.

For the final BA physical-4 state:

```text
local/sink K/V IDs = [1, 46, 47, 48]
archive K/V IDs    = [2, 3, ..., 29]
pending K/V IDs    = [30, 31, ..., 45]
Draft-K IDs        = [2, 3, ..., 29]
recompute unchanged = true
```

The corresponding AO W20 archive is the same `[2, ..., 29]`; the difference is that BA holds the later physically evicted frames in pending CPU storage until the original logical W20 boundary. Per-block archive K/V, Draft-K and retrieval-visible candidate lists matched between AO W20 and BA physical4. Pending K/V and local/sink K/V were validated against their decoded sentinel identities and K/V equality, rather than synthesized from list length.

The separate simplified selection replay remains labeled as a simplified model. It is not used as evidence of actual source IDs.

## D2H lifetime conclusion

The inherited path creates CPU tensors with `to("cpu", non_blocking=True)`. A Python return alone is not treated as proof of host readiness. BA now records a CUDA Event after the eviction batch's copies on the source stream and synchronizes those events before logical promotion. The destination tensors remain strongly referenced in the pending lists until promotion.

```text
ASYNC_D2H_LIFETIME = EXPLICIT_BATCH_EVENT_GUARDED
EXPOSED_D2H_PROMOTION_WAIT = POSSIBLE / NOT_TIMED
PER_FRAME_BLOCKING_D2H = NO
SOURCE_OVERWRITE_BEFORE_EVENT = PREVENTED_BY_EVENT_GUARD
```

The promotion wait has not been timed in a real model run. It must be reported separately in any later latency experiment.

## Strict all-ready Attention parity

No matching all-ready AO-vs-BA Q/K/V capture, complete final K/V artifact or Native FA2 output artifact exists in the AO/BA worktrees. The available historical AO W1 path is explicitly unsynchronized and is not a correctness reference. No mismatched video, synthetic attention output or isolated RoPE-only test was substituted.

Therefore:

```text
RAW_QKV_PARITY = NOT_MEASURED
FINAL_KV_PARITY = NOT_MEASURED
NATIVE_FA2_OUTPUT_PARITY = NOT_MEASURED
FIRST_DIVERGENT_TENSOR = NOT_AVAILABLE
```

The implementation-level RoPE gate remains passing: actual `causal_online_rope` on Wan frequencies gives max/mean absolute error 0 for positions `[0,17,18,19]` versus the W20 logical slice. This does not replace full attention parity.

## Status

```text
OUTPUT_ISOLATION = PASS
COMPRESSED_MODE_REJECTION = PASS
FRAME_SEQ_LENGTH_SOURCE = PASS
ACTUAL_PER_BLOCK_SENTINEL_REPLAY = PASS
RETRIEVAL_EQUIVALENCE = PASS (tested replay)
ROPE_PARITY = PASS (implementation-level gate)
ASYNC_D2H_LIFETIME = EXPLICIT_BATCH_EVENT_GUARDED
ATTENTION_PARITY = NOT_MEASURED
SHORT_RUN_LATENCY = NOT_MEASURED
READY_FOR_F120 = NO
```

The remaining blocker is a matched all-ready real-state capture. No inference campaign was started.
