# BA-1.3 Final Gate — Historical Archive Content Diagnosis

## Scope and preservation

This is a read-only diagnosis of the existing BA-1.3 AO/BA captures. No BA
cache code was changed, and no new F30/F120 inference was launched. The
original capture files were verified against the hashes in the capture
manifest.

## Hash verification

`results/phase218ba13/sha256_verification.json` reports all 12 manifest-listed
source/config/capture artifacts present with matching SHA256. The original
capture paths are the `/tmp/ba13_captures/ao` and `/tmp/ba13_captures/ba`
paths recorded in the manifest.

## Identity finding

The existing `.pt` payloads do **not** contain authoritative per-entry global
frame IDs. Their archive-related fields are only:

```text
archive_k:  list[Tensor]
archive_v:  list[Tensor]
pending_k:  list[Tensor]
pending_v:  list[Tensor]
```

There is no archive ID list, promotion-batch ID, frame start/end metadata, or
per-tensor global frame identity. `memory_indices` and `selected_ids` identify
retrieval selections, not the identities of every archive slot.

Therefore the existing artifacts cannot prove whether block-9 slots 2–6 are:

```text
A. different global frames at corresponding positions; or
B. the same global frames with different K/V contents.
```

Assigning identity from list position would violate the requested gate.

## Existing evidence

At block 7:

```text
AO committed archive length = 1
BA committed archive length = 1
AO pending length = 0
BA pending length = 16
```

At block 9:

```text
AO committed archive length = 7
BA committed archive length = 7
AO pending length = 0
BA pending length = 16
memory_indices AO = [[0, 1]]
memory_indices BA = [[0, 1]]
selected IDs AO = [0, 1]
selected IDs BA = [0, 1]
```

Direct tensor comparisons show archive slots 0–1 exact at block 9. Slots 2–6
are not exact. The largest direct K differences are 4.328125, 3.6953125,
3.171875, 5.546875 and 5.3671875; the corresponding V differences are
11.609375, 9.1953125, 9.953125, 11.375 and 12.34375. Pairwise comparison
against all BA slots does not produce exact matches for these entries.

This is actual content divergence, but its global-frame interpretation is
unresolved because identity metadata was not captured.

The existing BA-1.3 attention result remains valid in its limited scope:
captured raw Q/K/V, final K/V, transformed query and Native FA2 output were
exact at blocks 0, 1, 7 and 9. Block 9 selects only IDs `[0, 1]`, so it does
not exercise the divergent archive slots 2–6.

## Earliest divergence

The earliest captured archive-content divergence is:

```text
block_009_layer_000
archive_k slot 2
archive_v slot 2
```

The preceding capture at block 7 contains only one committed entry and cannot
localize the first differing frame among the later entries. The captures also
do not occur at the exact physical-eviction, D2H completion, or promotion
events. Consequently the first lifecycle stage causing the content difference
cannot be established from the current evidence.

Source review confirms BA associates pending K/V frames with a batch-owned
CUDA event and synchronizes that event before promotion; however, that source
fact does not identify which global frame each captured tensor represents and
does not prove the historical content divergence occurs at D2H, recompute,
promotion, or insertion.

## Parity scope

```text
CAPTURED_ATTENTION_PARITY     = PASS (four layer-0 capture points)
FULL_ARCHIVE_CONTENT_PARITY   = NO / UNRESOLVED
FULL_30_LAYER_PARITY          = NOT_MEASURED
FULL_R16_RETRIEVAL_PARITY     = NOT_MEASURED
```

The four-point result must not be generalized to full archive parity, all 30
layers, or full R16 retrieval. No performance benchmarking or F120 is
authorized by this diagnosis.

## Required next evidence

The smallest decisive follow-up is an eviction/promotion-only capture that
adds a global frame ID and lifecycle metadata to each K/V and Draft-K entry:

```text
global_frame_id, source block, physical slot, pending batch ID,
commit/promotion sequence, D2H-ready event association
```

It should compare the first mismatching global ID through physical eviction,
pending CPU readiness, promotion and archive insertion. This is not run in the
current task.

## Final status

```text
FULL_ARCHIVE_PARITY = NO
GLOBAL_ID_NORMALIZATION = BLOCKED_BY_MISSING_CAPTURE_METADATA
FIRST_CONTENT_DIVERGENCE = block_009 layer_0 archive slot_2 (identity unresolved)
BA_CACHE_MODIFIED = NO
NEW_FULL_INFERENCE = NO
F120 = NO
PERFORMANCE_BENCHMARK = NO
```
