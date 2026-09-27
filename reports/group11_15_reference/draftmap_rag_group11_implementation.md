# Group 11 DraftMap-RAG implementation status

Date: 2026-09-26

## Status

`GROUP_11_STATUS = COMPLETE` for the quality-first W12 BF16 canonical run.
The former checkpoint blockers were resolved by moving retrieval into
`CausalWanSelfAttention.forward` and defining one global key-axis
normalization plus block-mass aggregation. The implementation remains a
synchronous-fetch reference, not a qualified latency benchmark.

## Implemented behavior

- Current per-layer BF16 Draft-Q is transiently scored against historical
  same-layer BF16 Draft-K, then discarded.
- Historical Draft-Q is not persisted and is not read by the current score;
  Q(t-1) predictive retrieval is a separate future experiment.
- All eligible historical blocks share one row-softmax normalization domain;
  block mass is summed per canonical LongLive frame ID and averaged over heads
  and current query blocks.
- Existing CPU K/V archive, IDs, five-frame recent exclusion, six-frame W12
  retrieval budget, RoPE handling, context assembly, and dense BF16 attention
  are reused.
- No quantization, low-bit Anemoi execution, sparse routing, or persistent
  low-bit cache is active.

## Validation

- Focused tests: `4 passed`.
- Case01 trace-validation smoke: MP4 completed, 474 frames, 16 FPS, 832x480;
  trace contains 30 layers and real warped denoising steps.
- Canonical run: 10/10 MP4s completed with the same video protocol.
- Frozen evaluator: 10 quality rows; DINO mean `0.834218168259`, SSIM mean
  `0.387697658402`, PSNR mean `11.354638432428`, LPIPS `NOT_AVAILABLE`.

## Artifacts

- Config: `configs/group11_draftmap_online_w12_10case.yaml`
- Results: `results/draftmap_rag_online_w12_10case/`
- Summary: `results/draftmap_rag_online_w12_10case/summary.csv`
- Detailed report: `/home/zju/work/zxl/reports/draftmap_rag_online_current_q_w12.md`
- Flash Fetch design: `/home/zju/work/zxl/reports/draftmap_rag_flash_fetch_followup.md`

The prior blocked audit remains available at
`reports/draftmap_rag_group11_insertion_point.md`; it records the original
pipeline-boundary design and is superseded for implementation status by this
report.
