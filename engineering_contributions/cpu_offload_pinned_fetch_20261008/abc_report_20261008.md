# H200 GPU1 Offload Alignment Experiments A/B/C — 2026-10-08

Status: MICROBENCH_COMPLETED, END_TO_END_ADOPTION_PENDING.

Scope: standalone 2x4MiB INT16 CUDA tensors, 5 iterations per policy, not the full WAN/LLRAG model and not a 120-frame A/B.

| Mode | Archive median ms | First fetch median ms | Repeated fetch median ms | Exact fetched tensor |
|---|---:|---:|---:|---|
| archive_pin | 0.1682 | 0.1779 | 0.1732 | True |
| fetch_pin | 0.1677 | 0.1788 | 0.1736 | True |
| reusable | 0.1689 | 0.1767 | 0.1730 | True |

B: event-synchronized D2H exactness = True. It does NOT prove unsafe immediate reads or original application synchronization.

C: VmRSS before/after and torch.cuda peak allocated are recorded in JSON, but they are NOT true peak pinned-memory ownership accounting. C is preliminary only.

WARNING: benchmark source uses non_blocking GPU->pageable CPU followed by pin_memory before CUDA completion in archive_pin path; an equality check at end does not establish race freedom. Application-level exact KV lifetime, pinning cache amortization, and direct pageable H2D fallback must still be verified.

Previous single matched 30-frame Control vs Aligned: 22.736s vs 21.149s E2E; measured video cross-output SSIM=0.974347, PSNR=38.549dB. Single pair insufficient, no merge.

Original Ours and SolarWM source unmodified. Test script /tmp/offload_abc_gpu1_20261008.py.
