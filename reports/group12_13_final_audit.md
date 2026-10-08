# Group12/13 final audit

Audit date: 2026-09-27. This audit used the existing corrected case01 artifacts
and did not regenerate videos or change source/checkpoints/dataset.

## Common protocol

- commit: 25d7e7a52a50a28e1c7fbc7d4eaf1493a7fe0098
- case: case_01
- seed: 0
- window: W12
- output: 474 frames, 832x480, 16 FPS
- final attention: BF16
- native low-bit attention kernel: NO
- persistent Draft-K: GPU BF16
- CPU Draft-K: 0
- Draft-K H2D: 0

Both videos were probed successfully as 474-frame, 29.625-second MP4s.

## Results

| group | status | CPU KV bytes | GPU local KV bytes | GPU Draft bytes | compression | latency | DINO | SSIM | PSNR | LPIPS |
|---:|---|---:|---:|---:|---:|---|---|---|---|---|
| 12 | valid existing artifact | 15,749,976,960 | 3,450,470,400 | 248,832,000 | 1.971700x (derived) | MISSING | MISSING | MISSING | MISSING | NOT_AVAILABLE |
| 13 | valid existing artifact | 8,778,818,880 | 3,450,470,400 | 248,832,000 | 3.537405x (derived) | MISSING | MISSING | MISSING | MISSING | NOT_AVAILABLE |

CPU_KV_BYTES is the measured persistent compressed archive total in the
memory JSON. GPU_KV_BYTES is the measured local BF16 KV allocation.
GPU_DRAFT_BYTES is the measured persistent Draft-K allocation. No full
history BF16 shadow and no CPU Draft-K were reported.

### Group12

Artifact:
results/group12_corrected_case01_rerun/

Runtime semantics identify corrected Group12 mode with historical K=INT8,
V=FP8 E4M3, BF16 final attention and no native low-bit kernel. The audited
memory file reports 3,240 evicted compressed entries and 31,770 retrieved
archived entries. It reports 15,749,976,960 CPU archive bytes and a derived
1.971700x ratio.

The runtime trace contains stale per-step fields saying
PERSISTENT_STORAGE_MODE=BF16_FAKE_QUANT and ratio 1.0, which conflict with
the top-level corrected runtime contract and the measured compressed archive
bytes. Those stale trace fields are not used for the final memory conclusion;
this inconsistency should be fixed in a future bookkeeping-only patch.

### Group13

Artifact:
results/group13_corrected_case01_rerun4/

Runtime semantics identify corrected Group13 mode with persistent NVFP4 K/V,
BF16 final attention, GPU Draft-K, and no native low-bit kernel. The audited
memory file reports 3,240 evicted compressed entries and 31,770 retrieved
archived entries. It reports 8,778,818,880 CPU archive bytes and a derived
3.537405x ratio.

The runtime file top-level contract is consistent with persistent low-bit
storage, although its per-step trace also contains stale fake-quant/1.0 fields.

## Quality/e2e latency availability

No DINO, SSIM, PSNR, LPIPS, or canonical end-to-end latency artifact was found
under the two corrected case01 output directories. They are therefore marked
MISSING/NOT_AVAILABLE in the machine-readable audit. No values were inferred
from unrelated experiments and no videos were rerun.

## Rerun decision

- GROUP12_RERUN_REQUIRED = NO: video and measured memory/runtime contract are
  present; only quality/e2e fields are missing.
- GROUP13_RERUN_REQUIRED = NO: video and measured memory/runtime contract are
  present; only quality/e2e fields are missing.
- Quality evaluation remains pending if the frozen evaluator is required.

## Final fields

GROUP12_STATUS = VALID_EXISTING_ARTIFACT_AUDITED
GROUP13_STATUS = VALID_EXISTING_ARTIFACT_AUDITED
GROUP12_RERUN_REQUIRED = NO
GROUP13_RERUN_REQUIRED = NO
