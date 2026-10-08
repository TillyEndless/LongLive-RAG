# H200 Group12–15 alignment gate

- GROUP11_ALL_GPU_DRAFT_ALIGNED: YES (semantic reference retained; no Group11 runtime modified)
- GROUP12_FAKE_QUANT_ALIGNED: YES
- GROUP13_FAKE_QUANT_ALIGNED: YES
- GROUP14_SPARSE_FAKE8_ALIGNED: YES
- GROUP15_SPARSE_FAKE4_ALIGNED: YES
- MEMORY_ACCOUNTING_ALIGNED: YES (authoritative CPU history BF16; GPU low-bit working storage; draft fields emitted by corrected runtime)
- MANIFEST_ALIGNED: YES (SHA256 `db9462c0954186b9128543b58fb6d1d496a8b89e0754333beca698d3cf461fec`, seed 0)
- EVALUATOR_ALIGNED: PENDING until generated videos exist; evaluator must use frame 0 vs 237, 474 frames, 16 FPS, 832x480, DINOv2-small CLS, raw RGB SSIM/PSNR.

## Corrected storage contract

CPU historical K/V remain authoritative BF16. H2D source is BF16. Group12/14 use fake INT8 K + fake FP8 E4M3 V, Group13/15 use fake NVFP4 K/V, then dequantize to BF16 before ordinary attention. No Anemoi native low-bit kernel is used.

## Sparse semantics

Group14/15 `group_sparse_ratio` is the retained interaction/block ratio. The campaign includes retained ratios 30%, 20%, 10%, and 5%; it does not reuse the old inverted/drop-ratio outputs.

## Ported files

- `utils/compressed_history_archive.py`
- `utils/h200_group_runtime.py`
- `pipeline/causal_inference.py` group-mode wiring
- new configs under `configs/group12_15_corrected_campaign/`

## Unchanged H200-specific paths

No SM120-native kernel or hardware-specific low-bit attention path was ported. Existing source/results outside the new corrected campaign root are preserved.
