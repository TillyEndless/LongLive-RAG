# Group14 Non-monotonic DINO Audit

## Audit conclusion

**The current Group14 result is not a valid 10-case comparison with Group12.**
The canonical evaluator produced 10 rows for Group12 but only 4 rows for Group14: all are `case_01`, one for each retained target.
Therefore the observed non-monotonic DINO curve is a one-case diagnostic, not a reliable group-level quality curve.

## Evaluator audit

- Evaluator: `scripts/evaluate_all_canonical_aligned.py`.
- Manifest SHA256: `db9462c0954186b9128543b58fb6d1d496a8b89e0754333beca698d3cf461fec`.
- Frame protocol: frame 0 vs frame 237.
- Video protocol: 474 frames, 16 FPS, 832x480.
- DINO: DINOv2-small CLS cosine.
- Group12 coverage: 10 cases.
- Group14 coverage: 1 case × 4 retained targets.

## Group14 actual routing and DINO

| labelled retain | actual SPARSE_RATIO | retained blocks / total | actual retained fraction | DINO | SSIM | PSNR |
|---:|---:|---:|---:|---:|---:|---:|
| 30% | 0.3 | 88 / 293 | 0.300855 | 0.479986 | 0.115846 | 9.218417 |
| 20% | 0.2 | 59 / 293 | 0.201709 | 0.269011 | 0.177127 | 11.474186 |
| 10% | 0.1 | 30 / 293 | 0.102564 | 0.213184 | 0.178838 | 12.358354 |
| 5% | 0.055 | 17 / 293 | 0.058120 | 0.523238 | 0.283350 | 15.032237 |

## Findings

1. The 5% row is not exactly 5% retained: `SPARSE_RATIO=0.055`, retaining 17/293 blocks (about 5.81%).
2. The four Group14 videos all pass the same frame/fps/resolution protocol and point to distinct output paths.
3. Runtime metadata reports `REAL_SPARSE_EXECUTION` and the same routing implementation ID for all four rows.
4. The evaluator did not accidentally use the same video for all four rows; the paths and video hashes are distinct in the audit source artifacts.
5. The high 5% DINO is therefore not evidence of a general sparsity benefit: it is one case and can arise from case-specific top-k block selection.
6. The 10%/20% dip cannot be distinguished between genuine non-monotonic routing behavior and case-level variance until all four settings are evaluated on the same canonical 10 cases.

## Required correction

Run the same evaluator on Group14 retained 30/20/10/5 for case_01–case_10, then compare case-by-case and by mean. Do not use the current Group14 summary for a Group12-vs-Group14 claim.

Original videos and evaluation outputs were not modified.
