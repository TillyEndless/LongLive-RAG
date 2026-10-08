# Unified Latency Composition Profile

## Scope

Matched `case_01`, seed 0, W12, 120 latent output frames (474 decoded frames),
same checkpoint and group semantics. Only Group12 and Group14.1 were run;
Group14.2/14.3/14.4 were not run.

The optional `unified_latency_profile=true` flag records CUDA events around the
actual `attention(...)` backend call. It performs one current-stream
synchronization at the end of the interval and writes records after inference.
No per-call synchronization was added.

## Validation

Both variants produced 6000 attention CUDA-event records. Group14.1 captures
`original_k_len` before sparse routing and `actual_k_len` after gather.
The H200 environment has `flash_attn` and no `flash_attn_interface`; source
dispatch therefore selects the FlashAttention 2 path (`FLASH_ATTN`) for both
variants.

| metric | Group12 | Group14.1 |
|---|---:|---:|
| Process wall | 217.834s | 232.129s |
| E2E inference | 101.467s | 106.785s |
| Transformer total | 88.716s | 93.846s |
| Self-attention wrapper | 75.487s | 80.519s |
| Attention kernel CUDA events | 10.717s | 8.655s |
| QK elements | 5766463872000 | 4060988006400 |
| H2D exposed/work | 0.199s | 0.191s |
| Gather + cat | 0.294s | 0.298s |

## Accounting

`SELF_ATTN_WRAPPER` is a parent boundary and contains projection, cache/routing,
fetch, gather, attention and output projection. It must not be summed with
children. `ATTENTION_KERNEL_S` is GPU work from CUDA events only. H2D values
come from the existing archive aggregate; separate copy-stream work/exposed
events were not available. D2H, quant-pack, dequant-decode, mask-layout and
host-sync wait remain `NOT_AVAILABLE` as independent timers.

## Group12 vs Group14.1

- QK element reduction: **29.58%**.
- Attention-kernel CUDA work saved: **2.062s**.
- Attention-kernel speedup: **1.238x**.
- Self-attention wrapper speedup: **0.938x**.
- Transformer speedup: **0.945x**.
- E2E speedup: **0.950x**.

The sparse kernel work is now measured, but sparse-only routing is not yet
isolated from the common DraftMap retrieval trace. Therefore
`SPARSE_TOTAL_OVERHEAD` and `NET_SPARSE_GAIN` remain `NOT_AVAILABLE` rather
than being inferred from overlapping parent/child timers.

## Outputs

- Raw per-call CSV: `/data/zxl/LongLive-RAG-group11_15_h200/results/unified_latency_profile_raw.csv`
- Aggregate JSON: `/data/zxl/LongLive-RAG-group11_15_h200/results/unified_latency_profile_aggregate.json`
- Report: `/data/zxl/LongLive-RAG-group11_15_h200/reports/unified_latency_profile.md`
