# Group 11 all-GPU Draft-K quality sanity

Date: 2026-09-27. This was quality evaluation only. No inference was rerun and
no MP4 was regenerated or modified.

## Protocol and validation

- Input: `/home/zju/work/zxl/anemoi_native_draftattention_5090/results/group11_all_gpu_draft/w12/`
- Videos: 10/10; each is 474 frames, 16 FPS, 832×480
- Frame pair: 0 and 237
- Manifest SHA256: `db9462c0954186b9128543b58fb6d1d496a8b89e0754333beca698d3cf461fec`
- Frozen evaluator SHA256: `0e02f19e2fd226df919ca45cb0c2ea2c0232818dd6bbc78d1132f46e2a0e8b95`
- DINO model SHA256: `ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1`

The frozen source was not edited. The local path-adapted wrapper received one
path-only fix: it discovers MP4s both directly under the input directory and
under one nested method directory. DINO, SSIM, PSNR, frame selection, and
aggregation formulas are unchanged. Missing `lpips` is represented as
`NOT_AVAILABLE` and does not suppress the other metrics.

## All-GPU result

| Statistic | DINO | SSIM | PSNR | LPIPS |
|---|---:|---:|---:|---|
| Mean | 0.845216065645 | 0.330167290172 | 10.161790850428 | NOT_AVAILABLE |
| Median | 0.879358202219 | 0.266275630747 | 9.273079952955 | NOT_AVAILABLE |
| Min | 0.647960543633 | 0.157356215930 | 7.181735094158 | NOT_AVAILABLE |
| Max | 0.949302017689 | 0.644306401800 | 14.485387525155 | NOT_AVAILABLE |

Per-case values are in
`results/group11_all_gpu_draft/w12/quality.csv`.

## Comparison with old Group 11

| Metric | All-GPU Draft-K | Old Group 11 | Difference |
|---|---:|---:|---:|
| DINO | 0.845216065645 | 0.834218168259 | +0.010997897386 |
| SSIM | 0.330167290172 | 0.387697658402 | -0.057530368230 |
| PSNR | 10.161790850428 | 11.354638432428 | -1.192847581999 |

The difference is **not explained** by the available evidence. Candidate
history IDs match in all 5,220 trace rows per case, but selected historical
IDs differ substantially:

| Case | Selected-set differences | Order-only differences |
|---|---:|---:|
| 01 | 3676/5220 | 707/5220 |
| 02 | 4670/5220 | 347/5220 |
| 03 | 4738/5220 | 257/5220 |
| 04 | 4466/5220 | 442/5220 |
| 05 | 4577/5220 | 393/5220 |
| 06 | 4688/5220 | 308/5220 |
| 07 | 4744/5220 | 273/5220 |
| 08 | 4707/5220 | 293/5220 |
| 09 | 4622/5220 | 356/5220 |
| 10 | 4586/5220 | 380/5220 |

Thus the quality change cannot be attributed solely to Draft-K CPU versus GPU
placement. The exact cause requires a separate audit of the retrieval-score
and generation differences; no equivalence claim is made.

## Final classification

```text
GROUP11_DINO = 0.845216065645
GROUP11_SSIM = 0.330167290172
GROUP11_PSNR = 10.161790850428
GROUP11_LPIPS = NOT_AVAILABLE

OLD_GROUP11_DINO = 0.834218168259
OLD_GROUP11_SSIM = 0.387697658402
OLD_GROUP11_PSNR = 11.354638432428

QUALITY_DIFFERENCE_EXPLAINED = NO
QUALITY_RESULT_VALID = YES
```

`QUALITY_RESULT_VALID=YES` means the ten existing videos were scored under
the frozen frame/metric protocol. It does not mean the all-GPU run is
numerically equivalent to the old Group 11 run.
