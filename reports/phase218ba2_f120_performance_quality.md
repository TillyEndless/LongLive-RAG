# Phase BA-2 — Physical4 F120 Performance and Quality

Date: 2026-10-04

## Scope and validity

This report compares the same case_01 prompt, seed 0, checkpoint/LoRA and
474-frame evaluation protocol. The original Group 11.1 result is W12/R6/S1;
AO and BA are W20/R16/S1. The logical configurations are intentionally kept
separate.

BA used the existing implementation unchanged. The only setup correction was
adding the missing `wan_models -> /data/zxl/model_assets` symlink in the BA
worktree; the BA smoke config and output directory were dedicated to this run.

Archive full-state parity remains unresolved from BA-1.3b. Therefore the BA
video is a valid observed F120 artifact and the metrics below are reported, but
BA is **not correctness-certified** or production-ready.

## Results

| method | logical / physical | E2E s | Transformer s | Wrapper s | persistent GPU KV GiB | peak allocated | peak reserved | black frames | video |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Original Group 11.1 | W12/R6/S1 | 102.915148 | 89.850166 | 13.064982 | 3.213501 | 23.232698 | 24.359375 | 0 | original case_01 |
| Archived AO W1 | W20/R16/S1, Physical20 | 113.397373 | 100.506244 | 12.891129 | 5.355835 | NOT_AVAILABLE | NOT_AVAILABLE | 36 | `w1_corrected` |
| Current BA W1 | W20/R16/S1, Physical4 | 123.300954 | 110.416271 | 12.884683 | 1.071167 | NOT_AVAILABLE | NOT_AVAILABLE | 0 | `ba2_w1_f120` |

The Original Group 11.1 peak values are not re-derived here from the current
BA run; the source artifact's recorded memory file contains 24,945,919,488
bytes allocated and 26,155,679,744 bytes reserved (23.232698 and 24.359375
GiB using 1024^3). AO W1 and BA W1 did not record process allocator peaks in
their runtime memory files, so those cells remain `NOT_AVAILABLE` in the
machine-readable result rather than mixing allocator definitions.

BA versus archived AO W1:

- E2E: **+9.903581 s / +8.735% slower**.
- Transformer: **+9.910027 s / +9.860% slower**.
- Persistent GPU KV: **−4.284668 GiB / −80.0%**.

BA versus the original Group 11.1 result:

- E2E: **+20.385806 s / +19.809% slower**.
- This is a compatible same-protocol case comparison, but the source
  revisions and measurement dates differ; it is not a causal speedup claim.

## Safety gate and video integrity

The 1-block BA Physical4 smoke passed: 9 decoded frames, finite output,
Physical4 KV allocation and `w1_zero_acquire` were confirmed. The F120 output
contains exactly 474 frames at 16 FPS and 832×480.

Full scan:

- exact-black frames: 0
- frames with mean intensity < 5: 0
- frames 201–236: all non-black; minimum mean intensity 101.5689
- contiguous abnormal black spans: none

The archived AO W1 `w1_corrected` artifact has 36 exact-black/mean<5 frames,
including the 201–236 region; that historical result remains a corruption
artifact and is not a correctness oracle. BA does not reproduce that black
screen in this run, but one clean video does not mathematically eliminate the
known readiness-race hypothesis.

## Unified Evaluation

The original per-case evaluator was used for both available videos: 474
decoded frames, 16 FPS, 832×480, frame 0 versus frame 237, DINOv2-small,
RGB SSIM/PSNR and LPIPS AlexNet.

| method | DINO | SSIM | PSNR | LPIPS | protocol status |
|---|---:|---:|---:|---:|---|
| Original Group 11.1 | 0.871500 | 0.159932 | 7.994665 | 0.646038 | OK |
| Archived AO W1 | 0.870767 | 0.173224 | 8.987080 | 0.563489 | OK, but black-frame scan fails |
| Current BA W1 | 0.870062 | 0.271431 | 10.746253 | 0.486482 | OK, full scan clean |

The frame-pair evaluator reports valid numeric values, but the AO row must be
flagged `QUALITY_INVALID_FOR_FULL_VIDEO` because its full-frame scan contains
black frames. BA's quality row is numerically valid for the captured video,
while overall BA correctness remains gated by unresolved Archive parity.

## Provenance

- Original Group 11.1 config:
  `/data/zxl/tmp/one_case_eval_20260930/configs/g11_1.yaml`, SHA256
  `8cdad0fad1c99b18215f4444549b8d895d8886330e0c400246e18754e643119d`.
- Original Group 11.1 result head: `a7445fae3535abffd7fde2bff5a7ecc21bcb8274`.
- Archived AO W1 source head: `410948eb7e328513e7751ac2d44193115363cee4`.
- BA source head: `410948eb7e328513e7751ac2d44193115363cee4`, dirty worktree
  preserved.
- BA F120 config SHA256:
  `6c97bf6ac15770123a49d6db5381565a28a5005070bcfdc8645959160437e288`.
- Checkpoint and prompt manifest were the previously frozen verified artifacts;
  no weights were copied into this report bundle.

## Final status

```text
ACCELERATED_AO_REVISION = verified w1_corrected / 113.39737264066935 s
MATCHED_ARCHIVE_PARITY = NOT_MEASURED
FINAL_KV_PARITY = NOT_MEASURED
NATIVE_FA2_PARITY = NOT_MEASURED
PHYSICAL4_MEMORY = PASS (1.071167 GiB persistent GPU KV)
F30_ALL_READY = NOT_RUN
BA_W1_F120 = COMPLETED_OBSERVED_VIDEO
BLACK_FRAME_COUNT = 0
UNIFIED_EVALUATION = PASS_FOR_BA_VIDEO / OVERALL_CERTIFICATION_BLOCKED
LATENCY_VS_ACCELERATED_AO = +9.903581 s (+8.735%)
```
