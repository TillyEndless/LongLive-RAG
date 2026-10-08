# Final matched Group11 vs LongLive W12 latency

## Contract

- GPU: H200 NVL, GPU1
- case: case_01
- window: W12
- seed: 0
- checkpoint/LoRA: same H200 paths
- frame workload: same 10-prompt manifest entry and same bounded blocks 6-7
- Group11: DraftMap online retrieval, GPU Draft-K, BF16 attention
- LongLive: native W12 baseline, no DraftMap
- timing: CUDA event pairs with one final synchronization

The run was executed sequentially in tmux session final_matched_gpu1. No
existing output was overwritten. GPU0 was not used because an independent
reuse-equivalence validation occupied it.

## Measured matched phases

| phase | Group11 (s) | LongLive (s) | delta (s) |
|---|---:|---:|---:|
| wrapper CUDA boundary | 3.736702 | 1.047700 | +2.689002 |
| BF16 attention | 0.522427 | 0.557028 | -0.034601 |
| Q projection | 0.016159 | 0.014479 | +0.001681 |
| K projection | 0.018569 | 0.014351 | +0.004218 |
| V projection | 0.025637 | 0.014293 | +0.011344 |
| output projection | 0.013546 | 0.013389 | +0.000157 |

The wrapper-boundary ratio is 3.565x. The BF16 attention kernel is slightly
faster in Group11 in this capture; it is not the latency cause.

## Group11 counters

- DraftMap calls: 5,220
- attention calls: 6,000
- CPU historical fetch requests: 30,270
- unique fetch identities: 919
- duplicate fetch requests: 29,351
- H2D calls: 60,540
- H2D bytes: 290,127,052,800
- cache hits/misses in this run: 0 / 30,270

These values are from the new GPU1 runtime artifact, not copied from an older
profile.

## End-to-end timing caveat

LongLive runtime JSON reports e2e_latency_s = 42.717759 s. The Group11
runtime JSON does not contain a canonical E2E field; its progress output reports
an inference-loop elapsed wall proxy of approximately 103 s. Therefore the
full-E2E delta is not treated as a strict apples-to-apples number.

The reliable apples-to-apples result is the matched wrapper CUDA boundary:
Group11 3.736702 s versus LongLive 1.047700 s, delta 2.689002 s.

The current instrumentation does not separately measure DraftMap, CPU lookup,
working-set construction, or H2D subphase CUDA events, so those components
cannot be assigned individual seconds without over-attribution.

## Root-cause conclusion

- BF16 attention: NOT the cause.
- Additional Group11 wrapper/orchestration: measured cause of the 2.689 s
  matched wrapper delta.
- DraftMap/fetch/H2D/working-set individual shares: NOT_RESOLVED in this pass.
- NSYS: recommended only for subphase attribution; not launched automatically.

## Required return fields

TRACK_B_STATUS = COMPLETED
GROUP11_PROFILE_STATUS = COMPLETED_GPU1
LONGLIVE_W12_PROFILE_STATUS = COMPLETED_GPU1
GROUP11_LATENCY = 3.736702 s matched wrapper CUDA boundary
LONGLIVE_W12_LATENCY = 1.047700 s matched wrapper CUDA boundary
LATENCY_DELTA = +2.689002 s
LATENCY_RATIO = 3.565x
GROUP11_DRAFTMAP_TIME = NOT_SEPARATELY_MEASURED
GROUP11_H2D_TIME = NOT_SEPARATELY_MEASURED
GROUP11_H2D_BYTES = 290127052800
GROUP11_FETCH_REQUESTS = 30270
GROUP11_UNIQUE_FETCHES = 919
GROUP11_DUPLICATE_FETCHES = 29351
ACCOUNTED_DELTA = 2.689002 s at wrapper-boundary level
UNACCOUNTED_DELTA = NOT_COMPUTABLE for full E2E
ROOT_CAUSE = Group11 wrapper/orchestration overhead; subphase attribution unresolved
