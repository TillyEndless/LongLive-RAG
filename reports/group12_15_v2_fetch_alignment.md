# Group12–15 v2 Fetch Alignment

Base: `ae1c2faa8139a065139fbc0d4164565c64706e69`
Canonical10: **not run**. Quality metrics: **not run**.

## Final status

| Field | Result |
|---|---|
| GROUP11_1_HISTORY_SEMANTICS_REUSED | YES |
| GROUP12_FETCH_ALIGNED_WITH_GROUP11 | YES |
| GROUP13_FETCH_ALIGNED_WITH_GROUP11 | YES |
| GROUP14_FETCH_ALIGNED_WITH_GROUP11 | YES |
| GROUP15_FETCH_ALIGNED_WITH_GROUP11 | YES |
| HISTORY_AND_PROMOTION_PLANNED_TOGETHER | YES |
| HISTORY_AND_PROMOTION_MANDATORY_SERIAL_WAIT | NO |
| UNIFIED_FETCH_PLAN_IMPLEMENTED | YES |
| COALESCED_FETCH_USED | NO; one producer CUDA stream, shared-bandwidth-safe |
| ASYNC_PARALLEL_FETCH_USED | YES; non-blocking submission plus one consumer barrier |
| LOCAL_DEQUANT_OVERLAPS_H2D | NO; local materialization currently precedes plan submission |
| GPU_HISTORICAL_PERSISTENT_KV_BYTES | 0 |
| PROMOTION_MODIFIES_PERSISTENT_OWNER | NO |
| STATIC_TESTS_PASS | YES |
| MINIMAL_SMOKE_PASS | YES |
| CANONICAL10_RUN | NO |

## Source changes

- `utils/attention_fetch_plan.py`: shared `FetchPlan`, async submission, one
  event/barrier, separate history/promotion byte accounting.
- `wan/modules/causal_model_latentmem.py:209-255`: unified v2 planning and
  trace; history IDs and local promotion IDs are submitted together.
- `wan/modules/causal_model_latentmem.py:896-905`: v2 history consumes the
  already-submitted handle instead of issuing a second H2D for each record.
- `wan/modules/causal_model_latentmem.py:1284-1296,1651-1674`: CPU BF16
  local authoritative copies are maintained for transient promotion overlays.
- `wan/modules/causal_model_latentmem.py:1730`: v2 uses `select_top()`; it no
  longer turns a low-bit local owner into a persistent BF16 slot.
- `utils/local_lowbit_kv.py:95-103`: `select_top()` is non-mutating.
- `utils/compressed_history_archive.py`: historical GPU payload fields were
  removed; archive remains CPU BF16-only.
- `utils/runtime_memory_measurement.py`: CPU local BF16 promotion sources are
  accounted with storage-identity deduplication.
- `inference.py:494-503`: plan metadata is exported in runtime JSON.

## Smoke evidence

The new 10-block smoke output is under:
`results/group12_15_v2_fetch_alignment_smoke/`.

Group14 and Group15 each produced 720 plan records in the rerun. Representative
records show:

- Group14: history IDs `[5, 11, 1, 3, 9, 7]`; promotion IDs `[5, 4, 3, 2, 1, 7]`.
- Group15: history IDs `[1, 5, 11, 3, 2, 9]`; promotion IDs `[5, 4, 3, 2, 1, 8]`.
- Both report `HISTORY_AND_PROMOTION_PLANNED_TOGETHER=YES` and
  `MANDATORY_SERIAL_FETCH_WAIT=NO`.
- Each representative plan reports 57,507,840 bytes for history and the same
  amount for promotion, 24 physical tensor copies, and BF16 final attention.
- The persistent low-bit owners remain unchanged; memory output reports
  historical persistent GPU KV as zero.

The Group12/13 smoke configs set promotion IDs to empty, so their plan naturally
reduces to history-only behavior. Their 10-block generation smoke also passed.

## Serial vs unified materialization AB

`results/group12_15_v2_fetch_serial_unified_ab.json` is a CUDA fetch-only
correctness/timing check for both labels. It is not E2E latency.

| Variant | Serial exposed fetch (s) | Unified exposed fetch (s) | Numerical equality |
|---|---:|---:|---|
| Group14-v2 | 0.1778567 | 0.0000509 | exact |
| Group15-v2 | 0.0003234 | 0.0000298 | exact |

The AB uses the same source tensors and verifies identical ordering and zero
maximum absolute error. It demonstrates the scheduling primitive, not a claim
about full inference speedup.

## Exact final timeline

`causal_model_latentmem.py:486-501`
Q/K/V are created; current Q is passed to historical DraftMap scoring.

`causal_model_latentmem.py:222-301`
historical candidates are scored and history IDs are selected using the existing
Group11-compatible DraftMap mechanism.

`causal_model_latentmem.py:1730`
local promotion IDs are selected independently from the local Draft-K signal;
selection does not mutate the low-bit owner.

`causal_model_latentmem.py:209-255`
both ID sets become one fetch plan; CPU BF16 source tensors are gathered and
submitted before consumption.

`utils/attention_fetch_plan.py:47-100`
history and promotion copies are submitted on one producer stream and joined by
one event. There is no mandatory wait between the two logical fetch classes.

`causal_model_latentmem.py:896-945`
history data is consumed from the submitted handle; promotion data overlays the
corresponding local temporary BF16 materialization once.

`causal_model_latentmem.py:1057`
the unchanged stock BF16 attention call consumes the ordered working set.

After attention, transient handle/overlay tensors become reclaimable; persistent
local low-bit slots are unchanged.

## Known limitation

Local low-bit dequantization is currently performed while constructing the
call-local local window before unified CPU fetch submission. Therefore
`LOCAL_DEQUANT_OVERLAPS_H2D=NO` in this implementation. The unified scheduler
itself is semantically aligned and removes the two-class mandatory serialized
wait; moving local dequant into the overlap window is a separate optimization,
not part of this correctness alignment.
