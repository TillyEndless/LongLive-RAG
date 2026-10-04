# Phase 2.18ap quality/latency tradeoff

Latency:
- Baseline E2E: 119.113 s
- W1 exploratory E2E: 113.397 s
- Single-run difference: -5.716 s (-4.80%)

This is one matched exploratory pair, not a statistically established acceleration.

Quality protocol:
Both passed the same decode/protocol checks and evaluator. Official frame0-vs237 rows are very close overall but are not ground-truth quality scores. Corresponding-frame similarity is high through frames 1-200, then degrades substantially from 201 onward, with mean DINO 0.6148 / SSIM 0.5810 / LPIPS 0.4121 in frames 201-300.

Conclusion: W1 shows a possible latency/quality tradeoff in this exploratory run. It is not synchronization-safe, and the available evidence cannot attribute the later divergence specifically to premature KV reads.
