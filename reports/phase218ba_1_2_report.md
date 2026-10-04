# Phase BA-1.2 — Event Bookkeeping and Matched All-Ready Parity

## Event bookkeeping correction

The previous BA-1.1 implementation kept a global event list. BA-1.2 replaces it with `logical_history_pending_batches`. Each batch owns:

```text
K frame destinations
V frame destinations
Draft-K entries
the CUDA Event recorded after that batch's D2H copies
```

Promotion consumes batches in order. It synchronizes only the event attached to the batch containing the frame being promoted. If promotion splits a batch, the event remains attached to the remaining frames. The event reference is retired only when the batch is empty. No device-wide synchronization and no per-frame `.to("cpu")` blocking operation were added.

The BA all-ready config is prepared separately with `group11_fetch_mode: serial_full` and a BA-only output folder:

`/data/zxl/LongLive-RAG-group11_1-w1-phase218ba/results/w1_4frame_allready/case_01`

It has not been used for a full run in this phase.

## Real CUDA D2H test

The test was run with PyTorch `2.5.1+cu124`, CUDA `12.4`, on H200.

| Path | Destination pinned | Event query before stream sync | After sync | Event duration |
|---|---:|---:|---:|---:|
| Existing `to("cpu", non_blocking=True)` | true | false | true | 21.766 ms |
| Explicit pinned `copy_(..., non_blocking=True)` | true | false | true | 0.691 ms |

The actual environment's existing destination was observed as pinned in this test, and the completion Event was not ready before the copy stream synchronized. This proves asynchronous completion bookkeeping is needed; it does not prove useful overlap with model compute. The latter remains `NOT_PROVEN`.

```text
D2H_BATCH_LIFETIME = PASS (event-guarded test)
PINNED_DESTINATION = OBSERVED_TRUE
ACTUAL_USEFUL_OVERLAP = NOT_PROVEN
```

## Replay regression

The actual `_apply_cache_updates` replay was rerun after the event-queue change. It still passed physical allocation, per-block raw sentinel K/V, pending K/V, local/sink IDs, Draft-K, candidate visibility and recompute stability. Final BA physical4 state:

```text
local/sink = [1, 46, 47, 48]
archive   = [2, 3, ..., 29]
pending   = [30, 31, ..., 45]
recompute_unchanged = true
```

The simplified selection replay remains labeled separately and is not treated as real DraftMap evidence.

## Strict AO W20 vs BA physical4 parity

The verified all-ready BA configuration was prepared as `serial_full`, which performs ordinary CPU→GPU historical fetch, Historical RoPE and final K/V assembly before stock Native FA2 reads. However, the required matched real-state capture was not available in the AO/BA worktrees:

- no matching AO all-ready Q/K/V state artifact;
- no matching BA replay input/state artifact;
- no complete final K/V pair for the same layer/chunk/denoising step;
- no paired Native FA2 output artifact.

The historical AO `w1_zero_acquire` run was not used as a correctness reference. No divergent tensor is claimed and no tolerance was changed.

```text
RAW_QKV_PARITY = NOT_MEASURED
FINAL_KV_PARITY = NOT_MEASURED
NATIVE_FA2_OUTPUT_PARITY = NOT_MEASURED
FIRST_DIVERGENT_TENSOR = NOT_AVAILABLE
READY_FOR_SHORT_PERFORMANCE_TEST = NO
READY_FOR_F120 = NO
```

No full inference or Unified Evaluation was started.
