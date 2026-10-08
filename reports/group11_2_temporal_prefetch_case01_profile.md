# Group11.2 previous-Q temporal-prefetch case01 profile

Matched case01 latency profile only; no smoke, canonical10, or evaluator.

- result directory: `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_2_temporal_prefetch_case01_profile`
- process wall time: **229.880 s**
- E2E: **114.455 s**; Transformer: **101.739 s**; Wrapper: **12.716 s**

## Correctness invariants

- attention calls: `6000`; bootstrap: `30`; temporal predictions: `5220`
- requested/H2D IDs: `30270` / `30270`; valid prefetch hits: `30090`
- non-bootstrap miss: `0`; fallback: `0`; current-Q re-retrieval: `0`
- demand H2D on valid prefetch: `0`
- exact Q(t−1) alignment: **PASS**
- final working set equals Q(t−1) prediction: **YES**

## Component timing

| component | count | mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|---:|
| Q-predict work | 5220 | 2.786 | 2.693 | 3.249 |
| host ID control | 5220 | 0.021 | 0.020 | 0.023 |
| archive lookup | 5220 | 0.000 | 0.000 | 0.000 |
| prefetch prepare | 5220 | 0.191 | 0.159 | 0.472 |
| H2D CUDA work | 5220 | 1.062 | 1.086 | 1.115 |
| exposed/event wait | 5190 | 0.001 | 0.001 | 0.002 |
| KV assembly | 5190 | 0.041 | 0.040 | 0.047 |
| available overlap window | 5190 | 546.865 | 396.590 | 1192.612 |
| overlap margin | 5190 | 545.803 | 395.496 | 1191.525 |

H2D CUDA work and consumer-stream wait are reported separately. Positive overlap margin means measured H2D work fit within the available window.

## Limits

- Output projection, FFN, and norm/residual were not separately instrumented; they remain `NA` rather than being inferred from the parent wrapper timer.
- This is one matched case and is not a multi-case speed claim.

## Artifacts

- events CSV: `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_2_temporal_prefetch_case01_events.csv`
- summary JSON: `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_2_temporal_prefetch_case01_summary.json`
- report: `/data/zxl/LongLive-RAG-group11_15_h200/reports/group11_2_temporal_prefetch_case01_profile.md`
