# H200 Real LongLive Fetch-Attention Overlap Validation

## Scope

Experimental tree only: /data/zxl/b01_longlive_grouped_streaming_20261007
Canonical workload: B01, 120 output latent frames, S1 + R14 + C3 = 18, BF16 attention, one prompt, H200 NVL GPU1.
Both clean performance runs were sequential on the same GPU with no active competing compute process.

Performance source snapshot before validation-only shadow instrumentation:
- causal_model_latentmem.py SHA256: 177fdc3ffe68d6e4ca76b49659dc1fc3f421775485dd3772b44d56d838b0a2a5
- pipeline/causal_inference.py SHA256: ebb80348d3492db701eaa4cb3286c63f6519c99a2579f91a91c46b9076a7bca2
- H200 grouped FA2 extension SHA256: cec3f4625bc551ec1af846789234ae2720fa0a89aa686017336046c524ccfdba

## Implementation

1. Historical K is RoPE-transformed at eviction with relative_frame_index=0 and stored in stable pinned CPU memory.
2. Historical V is stored in stable pinned CPU memory.
3. Retrieval producer is pure DMA: pinned CPU K/V -> final GPU K/V buffer on a dedicated fetch stream.
4. The fetch stream publishes the contiguous frontier with cuStreamWriteValue32; no runtime producer CUDA/RoPE kernel needs an SM.
5. Modified FA2 consumes ready historical groups with G=4; resident sink/current/local tiles bypass the gate.
6. Serial baseline uses the identical producer but waits for producer completion before stock FA2.

## Correctness

Historical K/V shadow at mature R14: K and V are bitwise equal to the original dynamic-RoPE retrieval path (max_abs=0, nonzero=0).
Progressive grouped-attention same-call shadow at Q=4680, KV=28080, selected_chunks=14:
- equal=True
- max_abs=0
- mean_abs=0
- nonzero=0

Independent full-video files are not byte-identical across separate runs (PSNR ~37.66 dB, SSIM ~0.9766).
Same-call K/V and attention-output shadows are bitwise exact, so end-to-end run-level bitwise determinism is not claimed.

## Canonical 120-frame latency

| Metric | Serial safe | Grouped streaming G=4 | Change |
|---|---:|---:|---:|
| E2E | 93.031 s | 88.019 s | -5.012 s (5.39%) |
| Transformer | 79.790 s | 75.240 s | -4.550 s (5.70%) |
| Wrapper | 13.241 s | 12.779 s | -0.462 s |

Mature R14 has 4050 CUDA-event layer samples in each run.

| Mature R14 median | Serial safe | Grouped G=4 |
|---|---:|---:|
| Producer GPU | 2.5011 ms | 2.5056 ms |
| Producer before attention | 2.5011 ms | 0.5607 ms |
| Attention interval | 2.7491 ms | 3.5821 ms |
| Producer/attention overlap | 0.0000 ms | 1.9438 ms |
| Producer tail after attention | 0.0000 ms | 0.0000 ms |
| Critical path | 5.2519 ms | 4.1445 ms |

Mature per-layer critical-path reduction: 5.2519 -> 4.1445 ms = -1.1074 ms (21.09%).
Producer duration itself is unchanged (~2.50 ms), while 77.58% of producer time is directly overlapped with the consumer.
Grouped consumer interval is 0.8330 ms longer than stock FA2 because it includes readiness waits/gating/contention.

Across all 4800 W1 calls:
- Serial W1 critical sum: 23.532 s
- Grouped W1 critical sum: 19.126 s
- Saved: 4.406 s (18.72%)
- attention_wrapper phase: 65.784 -> 61.386 s, saving 4.398 s.

## Remaining H200 gap

Ideal full-overlap lower bound using current medians is max(producer, stock_FA2) = 2.7491 ms.
Actual grouped critical path is 4.1445 ms, leaving 1.3954 ms above this bound.
Main terms: pre-attention producer head-start 0.5607 ms and grouped consumer penalty 0.8330 ms.
Current implementation captures about 44.2% of the theoretical serial-to-perfect-overlap saving.

## Raw artifacts

- Serial runtime: /data/zxl/b01_longlive_grouped_streaming_20261007/outputs_clean/h200_serial_f120/rank0-0-0_lora_runtime.json
- Grouped runtime: /data/zxl/b01_longlive_grouped_streaming_20261007/outputs_clean/h200_grouped_f120/rank0-0-0_lora_runtime.json
- Serial log: /data/zxl/b01_longlive_grouped_streaming_20261007/results_clean/h200_serial_f120_gpu1.log
- Grouped log: /data/zxl/b01_longlive_grouped_streaming_20261007/results_clean/h200_grouped_f120_gpu1.log
- Mature grouped-attention shadow: /data/zxl/b01_longlive_grouped_streaming_20261007/results_clean/h200_grouped_attn_shadow_mature_f45_gpu1.log
- CSV summary: /data/zxl/b01_longlive_grouped_streaming_20261007/results_clean/h200_real_overlap_profile.csv

## Superseded result

The earlier ~13.4% H200 E2E figure came from non-clean, non-matched runs and is superseded.
Canonical clean same-GPU 120-frame result: 5.39% E2E and 5.70% Transformer acceleration.
