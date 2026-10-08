# Group11.4 direct predicted working-set ablation — case01

## Scope

This is one matched `case_01` latency profile only. No canonical10 and no
quality evaluation were run. The accepted run used GPU1 after the competing
GPU1 process was stopped. The previous GPU0-contended and stale-dispatch runs
are archived and excluded.

## Semantics

| Field | Value |
|---|---|
| MODE_NAME | `next_layer_prefetch_direct` |
| PREDICTOR | `P_(l+1) = I_l` |
| NEXT_LAYER_CURRENT_Q_RETRIEVAL | OFF |
| CORRECTION_FETCH | OFF |
| RECONCILIATION | OFF |
| FINAL_HISTORY_SET | `P_(l+1)` |
| STOCK_DENSE_FLASHATTN | YES |
| CPU_SOURCE_PREPINNED | YES |
| REUSABLE_GPU_STAGING | YES |
| HOST_ID_SYNC_PRESENT | YES (kept for Python-indexed archive) |
| QUALITY_STATUS | NOT_EVALUATED |

## Correctness smoke

Smoke result: `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_4_direct_prefetch_smoke_v2/rank0-0-0_lora_runtime.json`.

| Counter | Value |
|---|---:|
| attention calls | 1500 |
| prefetch requests | 3161 |
| direct predicted sets consumed | 696 |
| prefetch state misses | 109 |
| next-layer history retrieval calls | 0 |
| correction fetches | 0 |
| reconciliations | 0 |

The smoke produced a valid video with no NaN/Inf and passed the required zero
checks. State misses used demand fetch of the same predicted IDs; they did not
trigger current-Q re-retrieval.

## Accepted case01 profile

| Metric | Direct prediction | Current-Q reference | Direct − Current-Q |
|---|---:|---:|---:|
| E2E inference (s) | 108.752754 | 118.088107 | -9.335353 |
| Transformer (s) | 96.011585 | 105.350580 | -9.338995 |
| Self-attention wrapper (s) | 83.211331 | 92.601804 | -9.390473 |
| Wrapper (s) | 12.741168 | 12.737527 | +0.003641 |

The matched current-Q reference is
`results/rag_strategy_profile_runs/group11_4/case_01/`.

## Prefetch timing

| Metric | Value |
|---|---:|
| DraftMap retrieval total (s) | 0.247278 |
| Host ID control total (s) | 0.089497 |
| Prefetch preparation total (s) | 1.483933 |
| H2D CUDA work total (s) | 0.090248 |
| H2D exposed wait total (s) | 0.002171 |
| H2D hidden total (s) | 0.088077 |
| Median available overlap window (ms) | 13.641341 |
| Median H2D work (ms) | 0.177120 |
| Median exposed H2D (ms) | 0.004096 |
| Median host control (ms) | 0.017556 |
| Fully-hidden useful H2D rate | 0.996032 |
| timed useful prefetch events | 504 |

`CURRENT_LAYER_COMPUTE_AFTER_LAUNCH_MS` was not emitted by the existing
runtime schema and is reported as `NOT_AVAILABLE`; no synthetic overlap value
is substituted. Attention-kernel-exclusive time was also not separately
emitted, so the wrapper boundary is retained as the authoritative attention
measurement.

## Hypothesis decision

The direct mode reduces E2E by `9.335353` s, transformer by
`9.338995` s, and wrapper by
`9.390473` s relative to the matched current-Q profile.
Thus the measured result supports
`HYPOTHESIS_CORRECTION_DOMINANT = SUPPORTED` for this case01 comparison:
removing next-layer current-Q selection and correction/reconciliation produces
a real end-to-end reduction. This is a latency-only conclusion; it does not
establish quality preservation.

## Required answers

1. Directly consumes `P_(l+1)=I_l`: **YES** for non-bootstrap calls.
2. Correction fetch and reconciliation removed: **YES** (`0` in smoke/profile).
3. Prediction + host control + prepare: see the totals above; per-layer medians are in the raw CSV.
4. H2D time: median work/exposed values above; totals are measured CUDA events.
5. Available overlap: median `13.641341` ms from launch-to-consumer timestamps.
6. Fully hidden useful H2D: `0.996032`.
7. Remaining next-layer wait: exposed H2D total `0.002171` s.
8. Savings: E2E `9.335353` s; Transformer `9.338995` s; wrapper `9.390473` s.
9. Expected latency reduction: **YES**, measured at E2E boundary.
10. Correction/reconciliation main slow path: **SUPPORTED for this matched case**, not generalized beyond it.
11. Beats CURRENT_Q on case01: **YES** on E2E, transformer, and wrapper.
12. Residual bottleneck: ordinary transformer/non-attention work plus remaining dense attention wrapper; the current schema does not isolate an exclusive attention kernel timer.

## Artifacts

- Raw metrics: `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_4_direct_prefetch_case01/group11_4_direct_prefetch_case01_raw.csv`
- Summary: `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_4_direct_prefetch_case01/group11_4_direct_prefetch_case01_summary.json`
- Runtime JSON: `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_4_direct_prefetch_case01/rank0-0-0_lora_runtime.json`
- Config: `/data/zxl/LongLive-RAG-group11_15_h200/configs/group11_4_direct_prefetch_case01.yaml`
