# Engineering contribution: CPU Historical KV Archive and Pinned Streaming Fetch

**Selected implementation:** original Ours, retained after controlled H200 GPU1 three-run comparison. This is a supplementary *engineering implementation*, not a novel offload algorithm or a claimed statistically significant acceleration over LongLive-RAG.

## Implementation

- On historical KV eviction, copy frames GPU to CPU and materialize contiguous, pinned CPU archive tensors using `f.to("cpu", non_blocking=True).contiguous().pin_memory()` for K and V.
- On retrieval, use existing DraftMap to select history and fetch the pinned CPU sources to reusable GPU double buffers over a dedicated CUDA stream, coordinating consumption via CUDA events.
- Keep the Ours DraftMap retrieval, attention math and streaming attention unchanged.

Archived implementation file: `causal_model_latentmem.py`; SHA256: `3fa16767c3d482202177ecf1e6eb28d70579f77c3f0e041452a8ff7904a8965f`.
The file is a snapshot, not a drop-in standalone distribution; its original repo dependencies and checkpoints are required.

## H200 single-prompt matched A/B (30 latent frames, 10 blocks, Wan2.1-T2V-1.3B, W12/R6, GPU1)

| CPU archive + fetch policy | E2E mean (3 runs) | Median (3 runs) |
|---|---:|---:|
| **Ours: archive-time pin + pinned streaming fetch (retained)** | **21.364 s** | **20.843 s** |
| Pageable archive + lazy fetch-time pin and cache | 22.068 s | 21.365 s |
| Pageable archive + direct H2D on existing fetch stream | 21.603 s | 21.430 s |

All six additional runs exited 0 (nine successful runs total); raw values in `repeat_comparison_summary.json`. Generated outputs are not bitwise identical. GPU KV (3.2135 GiB) and CPU KV (4.8203 GiB) were equal across the initial single-run configurations, but these do not measure total host pinned allocations.

**Decision:** keep original Ours implementation; do not merge experimental alignment variants. The current limited sample shows no reliable advantage for either alignment method. Exact historical-KV shadow correctness, host-pinning peak lifetime accounting and larger 120-frame multi-prompt validation are pending.

**Scope:** comparison is within Ours retrieval and streaming fetch framework. The direct-pageable variant is not full LongLive-RAG's end-to-end runtime. Original sources and SolarWM were not modified during these tests.
