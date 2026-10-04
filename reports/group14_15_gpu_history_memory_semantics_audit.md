# Group14/15 GPU Historical KV Memory Semantics Audit

静态审计；未运行 inference、benchmark 或 GPU。源码版本：`/data/zxl/LongLive-RAG-group11_15_h200`。

## Conclusion

当前实现的真实结构是：

```text
CPU BF16 archive
  -> Q-sparse retained IDs
  -> promoted IDs: CPU BF16 H2D
  -> other retained IDs: lazy persistent GPU low-bit owner -> BF16 dequant
  -> BF16 attention
```

因此：

- `GPU_KV_MEASURED_BYTES` 主要是 `cache[k] + cache[v]`，即 local dense BF16 KV；不是历史 packed KV。
- 历史 GPU packed owner 是每个 archive record 的 `gpu_k_payload/gpu_v_payload`，由首次选中且未 promotion 的 fetch 懒创建，并保留在 record 上。
- Q-sparsity 一定减少当前 attention 的 active working set，但不等于按比例减少 persistent GPU packed cache。
- 当前实现没有 ratio-based eviction；只有历史 rolling eviction 会移除 archive record。

## Ownership and lifetime

| Object | Allocation / creation | Owner | Lifetime | Reused across calls | Eviction |
|---|---|---|---|---|---|
| Local K/V | `pipeline/causal_inference.py:595-614` | `kv_cache["k"]`, `kv_cache["v"]` | cache lifetime | YES | rolling cache overwrite |
| CPU historical BF16 K/V | `utils/compressed_history_archive.py:100-113` | `CompressedHistoryRecord.k_payload/v_payload` | archive-record lifetime | YES | rolling history archive eviction |
| GPU low-bit K/V | `utils/compressed_history_archive.py:116-168`, `170-196` | `rec.gpu_k_payload/gpu_v_payload` + metadata | lazy cache record lifetime | YES after first creation | only with record eviction; no Q-ratio eviction |
| Promoted BF16 K/V | `materialize_selected`, lines 208-285 | local `k_out/v_out` working lists | current attention materialization | NO | released/reused after attention |
| Dequantized BF16 K/V | `fetch` / `materialize_selected`, lines 116-168 and 248-280 | local output tensors | current attention materialization | NO as owner | released/reused after attention |

## Sparse ordering

当前是 **Q-sparse selection -> materialize selected**。在 `wan/modules/causal_model_latentmem.py:1264-1298`，Group14/15 先确定 retained IDs；随后 archive materialization 对 retained IDs 做 promotion/low-bit source split。不是“先把所有历史块 packed，再按 ratio 截取”。但 low-bit owner 对已经访问过的 selected record 会跨调用保留。

分类：

```text
GROUP14_STORAGE_MODEL = LAZY_RESIDENT_CACHE
GROUP15_STORAGE_MODEL = LAZY_RESIDENT_CACHE
```

## Current accounting formulas

`utils/runtime_memory_measurement.py:170-246`：

```text
GPU_KV_MEASURED_BYTES
  = deduplicated bytes(cache[k]) + bytes(cache[v])

GPU_KV_ACTUAL_PERSISTENT_BYTES
  = GPU_KV_MEASURED_BYTES
  + GPU_PACKED_K_BYTES
  + GPU_PACKED_V_BYTES
  + GPU_PACKED_SCALE_BYTES
  + GPU_PACKED_METADATA_BYTES

GPU_KV_BF16_EQUIVALENT_BYTES
  = logical BF16 bytes for every archive record shape
  + local BF16 bytes
```

`GPU_KV_BF16_EQUIVALENT_BYTES` is a logical estimate. It is not a direct allocation measurement. `TRANSIENT_DEQUANT_GPU_PEAK_BYTES` and `TRANSIENT_PROMOTION_GPU_PEAK_BYTES` are tracked separately in cache counters, but there is no single explicit `GPU_HISTORICAL_ACTIVE_WORKING_SET_BYTES` field.

## Required conceptual classes

| Conceptual class | Current source representation | Current status |
|---|---|---|
| `GPU_HISTORICAL_PERSISTENT_RESIDENT_KV_BYTES` | `GPU_PACKED_K_BYTES + GPU_PACKED_V_BYTES + scale/meta` for records whose low-bit owner exists | Partially available, not separately named |
| `GPU_HISTORICAL_ACTIVE_WORKING_SET_BYTES` | selected `k_out/v_out`, gathered/dequantized/promoted working tensors | Not directly aggregated as one field |
| `GPU_HISTORICAL_TRANSIENT_BYTES` | `TRANSIENT_DEQUANT_GPU_PEAK_BYTES`, `TRANSIENT_PROMOTION_GPU_PEAK_BYTES` | Available as separate counters |
| `PEAK_GPU_ALLOCATED_BYTES` | whole-process allocator peak | Not emitted by this function; requires explicit peak allocator instrumentation |

## Supplied retained-ratio numbers

The current repository does not contain a memory artifact or source formula that can trace these as GiB:

- Group14: `3.7889, 3.2270, 2.6303, 1.9970`
- Group15: `2.1532, 1.7853, 1.5181, 1.0684`

Several matching decimals occur in existing profiling CSVs as timing values, e.g. `time_score_ms` or `EXPOSED_CRITICAL_PATH_TIME`. Therefore the numbers are **not validated as GPU memory** by this audit. They should be marked `UNVERIFIED_FIELD_MAPPING`, not used as persistent KV GiB.

## Answers

1. Current Group14/15 “GPU history KV” is not a single well-defined field. `GPU_KV_MEASURED_BYTES` is local dense BF16 KV; historical packed storage is in `GPU_PACKED_*`.
2. Q-sparsity reduces active working-set materialization: **YES**.
3. Q-sparsity does not automatically delete a previously created low-bit owner: **YES, previously selected chunks can remain packed**.
4. Corrected path selects first, then materializes; low-bit creation is lazy on first selected non-promoted fetch.
5. A lazy persistent GPU cache exists.
6. The four Group14 values are not traceable as memory GiB from current artifacts; matching values appear in timing traces.
7. The four Group15 values have the same issue.
8. `GPU_KV_ACTUAL_PERSISTENT_BYTES` is not sufficiently precise as a final-table name because it hides the historical persistent component and does not expose active working-set bytes.
9. Final tables should report separately: local persistent KV, historical persistent resident KV, active working-set KV, transient peak KV, and whole-process peak allocated memory.

## Recommended formulas

Compute-only sparse:

```text
M_persistent = M_local + M_hist_resident
M_active = selected retained materialization
```

Storage-sparse only if the implementation explicitly evicts/unpacks unselected owners:

```text
M_persistent = M_local + r * M_hist_full + metadata
```

Current lazy cache:

```text
M_persistent(t) = M_local + occupancy(gpu_lowbit_owner_cache_t)
M_active(t) = current retained BF16 working set
```

The current source matches the lazy-cache formula, not `r * full_history`.

## Recommended field names

- `GPU_LOCAL_PERSISTENT_KV_BYTES`
- `GPU_HISTORICAL_PERSISTENT_RESIDENT_KV_BYTES`
- `GPU_HISTORICAL_ACTIVE_WORKING_SET_BYTES`
- `GPU_HISTORICAL_TRANSIENT_PEAK_BYTES`
- `GPU_KV_ACTUAL_PERSISTENT_BYTES`
- `KV_RELATED_GPU_PEAK_BYTES`
- `PEAK_GPU_ALLOCATED_BYTES`

No source or profiling code was modified by this audit.
