# Runtime memory measurement method

Date: 2026-09-27

## Authority

Final GPU KV and Draft-memory fields use runtime inspection of the persistent
cache owners, not tensor-shape formulas, nominal bit rates, or whole-process
`nvidia-smi` usage. Each tensor is measured with
`tensor.untyped_storage().nbytes()` and deduplicated by device, storage pointer,
and storage size. KV and Draft tensors are classified from their cache-owner
fields; model weights, activations, temporary attention workspaces, allocator
fragmentation, and unrelated processes are excluded.

`GPU_KV_LOGICAL_GiB` and `DRAFT_LOGICAL_GiB` are retained as sanity checks from
`numel * element_size`; they are not the authoritative final-table values.
`torch.cuda.memory_allocated/reserved` may be recorded separately as allocator
sanity checks and are never used as GPU KV.

## Persistent fields

- KV: K/V payloads, scales/amax, validity and block metadata owned by the cache.
- Draft: persistent Draft-K/Draft-Q and persistent Draft route/index metadata.
- Transient current Draft-Q, scores, masks, and workspace are excluded; their
  peak is `NOT_MEASURED` unless separately instrumented.
- CPU KV uses the same backing-storage measurement. CPU Draft is reported
  separately in the measurement artifact when present.

## Implementations and validation

- Group12 recorder: `LongLive-RAG-persistent-draftmap-native/utils/runtime_memory_measurement.py`.
- Group12 record fields include `MEMORY_MEASUREMENT_VALID=YES` and the required
  measured/logical GiB fields.
- Group11 remeasurement: isolated worktree
  `LongLive-RAG-group11-memory-measure`, one deterministic W12 case with a full
  120-frame cache fill.
- Low-bit custom objects are counted through their backing tensors, including
  payloads, scales, amax, and persistent metadata.

Blocked groups retain `NOT_MEASURED`; they are not marked complete.
