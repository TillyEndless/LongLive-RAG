# Group11 RAG strategy matched comparison

Matched case: `case_01`; window=12; 6000 attention calls; seed=0. This is a runtime-measurement comparison, not a canonical quality evaluation.

| Strategy | E2E s | Wrapper s | Retrieval work s | Fetch work s | Fetch exposed s | Hidden fraction | Fetch BW GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| CURRENT_Q | 147.386232 | 12.855735 | 10.044694 | 0.916738 | 0.916738 | 0.000000 | 316.478 |
| PREVIOUS_Q | 106.942340 | 12.686383 | 5.389340 | 0.859441 | 0.859441 | 0.000000 | 337.576 |
| FLASH_FETCH | 216.603399 | 12.906464 | 0.000000 | 1.461386 | 1.461386 | 0.000000 | 198.529 |
| NEXT_LAYER_PREFETCH | 119.783940 | 12.722295 | 5.790559 | 1.099396 | 1.099396 | 0.000000 | 518.997 |

## Decision facts

- FASTEST_MEASURED_E2E_STRATEGY = `PREVIOUS_Q`.
- LOWEST_SELF_ATTN_WRAPPER_STRATEGY = `PREVIOUS_Q`.
- LOWEST_FETCH_EXPOSED_STRATEGY = `PREVIOUS_Q`.
- HIGHEST_FETCH_HIDDEN_FRACTION_STRATEGY = `NEXT_LAYER_PREFETCH` only under the measured prefetch copy/wait accounting; demand/Flash copies are blocking and have zero hidden fraction.
- LOWEST_RETRIEVAL_EXPOSED_STRATEGY = `NOT_IDENTIFIABLE`: retrieval exposed was not independently separated from the existing wrapper boundary.
- BEST_CANDIDATE_FOR_COMPRESSION_BASE = `NOT_DECIDABLE_YET`: runtime is matched, but canonical quality provenance and Flash/prefetch semantic overlap evidence are not sufficient.

## Current-Q headroom

- CURRENT_Q_FETCH_WORK_UPPER_BOUND_S = 0.916738.
- CURRENT_Q_FETCH_E2E_UPPER_BOUND_S = 0.916738.
- CURRENT_Q_RETRIEVAL_UPPER_BOUND_S = NOT_IDENTIFIABLE because retrieval-exposed timing is not independently measured.
- CURRENT_Q bottleneck under these fields: the remaining self-attention wrapper/transformer work, not the measured H2D copy sum alone.

## Previous-Q

- Selected-set recall/precision/Jaccard/exact/order are `NOT_AVAILABLE`: the persisted profile contains aggregate DraftMap rows but not a stable per-call selected-ID trace for both strategies.
- Previous-Q has lower measured retrieval-work sum and lower E2E in this matched run, but selected-set or quality equivalence must not be inferred.

## Flash Fetch

- FLASH_THEORETICAL_SAVING_S = 1.1957811629399657.
- FLASH_FETCH_OVERLAP_PROVEN = NO: the existing Flash path is serial (`async_overlap=false`); no asynchronous overlap was measured.

## Next-layer prefetch

- PREFETCH_ACTUAL_OVERLAP_PROVEN = YES for CUDA-event work/wait observability only; this does not prove net E2E benefit.
- Prefetch counters: requested=29261, hits=15286, corrections=30270, waste=13975.
- PREFETCH_PRECISION = 0.522402; PREFETCH_RECALL = 0.335543; exact-set rate = NOT_AVAILABLE.
- PREFETCH CUDA work = 6.216781 s; exposed wait = 0.069704 s; hidden-by-difference = 6.147077 s.
- Prefetch precision/recall counters are recorded by the existing runtime counters; event-level ID-to-wait attribution is not used to redefine them.
- High prefetch hit/recall does not by itself establish a net critical-path gain.

## Integrity and limitations

- All four runs are case_01, seed=0, window=12, same checkpoint paths and 6000 attention calls.
- Group11.4 overlap rerun preserved the existing prefetch stream, wait_event, selected IDs, correction/waste counters, and BF16 attention path; no canonical10 or Group12–19 was run.
- ATTENTION_KERNEL is `NOT_AVAILABLE` in this run because unified profiler records were disabled; no wrapper/fetch time was substituted for it.
- Quality provenance is `NOT_AVAILABLE` for this runtime-only case; do not use this report as a quality ranking.
- Process wall was not used for ranking.
