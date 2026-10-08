# Group12 corrected evaluation

## Evaluation status

The ten existing Group12 videos were evaluated; no inference was rerun.

## Frozen protocol

- 474 frames, 16 FPS, 832x480.
- Compare frame 0 with frame 237 (`num_frames // 2`).
- DINOv2-small CLS cosine; raw RGB SSIM and PSNR.
- LPIPS: NOT_AVAILABLE because the frozen environment does not have the LPIPS package.

## Aggregate

| cases | DINO | SSIM | PSNR | LPIPS | latency |
|---:|---:|---:|---:|---|---|
| 10 | 0.846198 | 0.364931 | 10.749284 | NOT_AVAILABLE | NOT_AVAILABLE |

## Measured memory

| GPU KV | CPU historical KV | persistent GPU Draft | compression |
|---:|---:|---:|---:|
| 3.2135009765625 GiB | 14.668309092521667 GiB | 0.23174285888671875 GiB | 1.9717002557443741x |

All ten `memory_measurement.json` files report the same measured values. The persistent storage contract is `LOWBIT_STORAGE_BF16_COMPUTE`; selected archived K/V is dequantized before BF16 attention. Native low-bit kernels were not used.

## Metadata caveat

Three runtime JSON files retain a stale top-level `PERSISTENT_STORAGE_MODE=BF16_FAKE_QUANT`/`KV_COMPRESSION_RATIO=1.0` marker, while their `runtime_trace` and all ten measured memory files report the corrected low-bit archive contract. The table above uses measured memory and corrected trace evidence, not the stale top-level marker.

Inference latency was not emitted by the Group12 runner; it is marked NOT_AVAILABLE rather than inferred from file timestamps.

## Per-case quality

| case | DINO | SSIM | PSNR |
|---|---:|---:|---:|
| case_01 | 0.838121 | 0.186999 | 8.517508 |
| case_02 | 0.754033 | 0.287235 | 9.402948 |
| case_03 | 0.824159 | 0.566911 | 10.396128 |
| case_04 | 0.839082 | 0.261738 | 10.654852 |
| case_05 | 0.937930 | 0.502051 | 13.092041 |
| case_06 | 0.871390 | 0.236385 | 9.281135 |
| case_07 | 0.851047 | 0.442123 | 12.548257 |
| case_08 | 0.849390 | 0.395258 | 12.762318 |
| case_09 | 0.878512 | 0.598888 | 12.213192 |
| case_10 | 0.818312 | 0.171722 | 8.624464 |
