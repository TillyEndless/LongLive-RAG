# Corrected Group 12–19 campaign

Date: 2026-09-27. This report supersedes the earlier all-blocked status only
for the validated Group 12 milestone. The implementation uses the isolated
branch `research/persistent-draftmap-native`; no active worktree or prior
experiment was overwritten.

## Group 12 — COMPLETE

Group 12 uses persistent K INT8/V FP8 E4M3, persistent BF16 Draft-K, transient
current Draft-Q, 64-token pooling, DraftMap scoring over historical and current
blocks, and the native SM120 per-chunk-V-scale operator. Its quota is
`HIGH/EIGHT/FOUR/ZERO = 0/100/0/0`.

The canonical W12 run completed 10/10 cases. A fresh one-case steady-state
remeasurement found resident KV storage of 1,807,505,860 bytes
(1.683371011168 GiB), versus 3,538,944,000 BF16-equivalent bytes, giving
1.957916x resident compression. Persistent Draft storage is 27,684,000 bytes
(27,648,000-byte Draft-K plus 36,000-byte metadata). The old 1,835,226,240
value included Draft bytes and is not used as GPU KV. Mean
latencies are 87.212496 s total and 8.433507 s attention/kernel. Mean quality
is DINO 0.894127339125, SSIM 0.066587321129, PSNR 8.141206619864; LPIPS is
NOT_AVAILABLE.

## Remaining groups

Groups 13–19 were not launched. The current isolated implementation has no
validated persistent NVFP4 consumer, no K4/V4 persistent owner, no direct
heterogeneous 8/4 execution ABI, and no selected-block high-precision or
zero-route executor. They remain blocked for those source/ABI reasons rather
than being mislabeled as completed experiments.

See `results/group12_19_corrected_campaign.csv` and
`reports/persistent_draftmap_native_implementation.md` for the detailed
status and evidence.
