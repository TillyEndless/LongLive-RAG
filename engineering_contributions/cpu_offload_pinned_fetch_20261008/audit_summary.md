# H200 CPU Offload LongLive-RAG alignment A/B (2026-10-08)

Decision: **NOT_ADOPTED — preliminary 1-pair performance gain, insufficient robust validation**.

## Scope
Control: Group11.3 original CPU archive path `to(cpu, non_blocking=True).contiguous().pin_memory()`.
Aligned: same code except K and V archive D2H lists are `to(cpu, non_blocking=True)`.
Original Ours and SolarWM untouched.

Matched Wan2.1-T2V-1.3B, GPU1 H200, seed0, 30 latent frames / 10 blocks, W12, R6, draftmap_online, flashfetch_async/async_stock_flashattn, single prompt. Both use the exact same source checkpoint paths.
Both torchrun runs EXIT0 and produced MP4 + per-run runtime/memory JSON.
NFS startup delays ~minutes are excluded from model E2E latency readings.

## Results
| Metric | Original control | LongLive-style aligned |
|---|---:|---:|
| Generation E2E | 22.736 s | 21.149 s |
| Transformer | 19.244 s | 17.726 s |
| GPU KV | 3.213501 GiB | 3.213501 GiB |
| CPU KV | 4.820251 GiB | 4.820251 GiB |

Single-pair E2E change = -1.587 s (-6.98%). Not proof of reliable speedup.

Cross-run output comparison (117 decoded frames, FFmpeg YUV420P): SSIM 0.974347, PSNR 38.549226 dB.
Output SHA differs. SSIM/PSNR are *between Control and Aligned*; NOT ground-truth quality metrics and do not prove mathematical parity.
No DINO/LPIPS/VBench re-evaluation or exact historical-K/V shadow comparison in this run.
`PINNED_SOURCE=YES` and `cpu_pinned=True` appear in both runtime configs: these flags do not prove archive evicted tensors remain pinned or that async transfer lifetime is safe.

## Adoption gate
Do not merge. Next: 3+ repeats under same matched conditions (ideally interleaved order), historical-K/V exactness, pinned memory ownership/lifetime, H2D fetch synchronization, full 120-frame quality/latency and GPU/CPU allocations. A 30-frame smoke cannot certify full archival correctness, overlapping fetch or CPU memory safety.

Artifacts: outputs/control and outputs/longlive_archive; logs/control_v2.log and logs/longlive_archive_v2.log; ab_results_20261008.json.
