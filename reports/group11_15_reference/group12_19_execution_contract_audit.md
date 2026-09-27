# Group 12–19 execution-contract audit

Date: 2026-09-27. Read-only audit; no Group 12–19 inference was launched.

## Evidence used

- Native operator trace: `/home/zju/work/zxl/reports/anemoi_lowbit_attention_operator_trace.md`.
- Persistent K-INT8/V-FP8 ABI trace:
  `/home/zju/work/zxl/reports/anemoi_kint8_vfp8_persistent_compatibility_trace.md`.
- Unified capability audit:
  `/home/zju/work/zxl/reports/draft_attention_v2_unified_code_audit.md`.
- Existing ratio screen:
  `results/draft_v2_campaign/02_ratio_sweep_screen/sweep_results.csv`.
  Its candidate selection is explicitly `NOT_RUN`.

The current native Anemoi contracts are not interchangeable with the table
labels: the native 8-bit phase consumes Q=INT8, K=INT8, V=FP8 E4M3, while the
native 4-bit phase consumes Q=NVFP4, K=NVFP4, V=NVFP4. The LongLive adapter
accepts BF16/FP16 at its boundary but native preparation quantizes Q for those
phases. Evidence: `repos/LongLive-RAG/utils/anemoi_mpa.py:106-115,185-201`,
`repos/anemoi/anemoi/layers/attention/mpa/executor.py:371-413,480-548`, and
`repos/anemoi/csrc/attention/cuda/sm120/api.h:120-132,221-236`.

## Contract matrix

| Group | Requested persistent K/V | Requested execution | DraftMap / sparse | Exact support | Status | Blocker |
|---:|---|---|---|---|---|---|
| 12 | 8-bit | BF16 Q, same low-bit K/V, no sparse | none required | No | BLOCKED | Available native 8-bit ABI requires Q INT8, not BF16 execution Q; do not relabel Q8 as BF16-Q. |
| 13 | 4-bit | BF16 Q, same low-bit K/V, no sparse | none required | No | BLOCKED | Available native 4-bit ABI is homogeneous Q/K/V NVFP4; exact BF16-Q/K4/V4 path is not available. |
| 14 | 8-bit | BF16 plus sparse | ratio unresolved | No | BLOCKED | No authoritative sparsity ratio/config found. Current LongLive Anemoi adapter hardcodes sparse ratio 0.0. |
| 15 | 4-bit | BF16 plus sparse | ratio unresolved | No | BLOCKED | No authoritative sparsity ratio/config found; exact BF16-Q/K4/V4 path is also absent. |
| 16 | 8-bit | dynamic 16/8/0 | later guidance says no 16-bit phase | No launch | DEFERRED | Later teacher guidance supersedes this ablation: only 0/4/8 remains; no newer instruction retaining 16/8/0 was found. |
| 17 | 4-bit | dynamic 16/8/0 | later guidance says no 16-bit phase | No launch | DEFERRED | Same superseding 0/4/8 guidance. |
| 18 | 8-bit | dynamic 8/0 | ratio unresolved | No launch | BLOCKED | Existing sweep has 10/90 and 20/80 exploratory rows, but `candidate_selection.csv` says `NOT_RUN`; no authoritative best policy or selection criterion exists. |
| 19 | 4/8-bit | dynamic 8/4/0 | ratio unresolved | No launch | BLOCKED | Existing sweep has no selected canonical 8/4/0 policy; arbitrary candidate selection is forbidden. |

## Storage and execution details

The persistent K-INT8/V-FP8 implementation stores CPU historical K as signed
INT8 and V as FP8 E4M3, then dequantizes both to BF16 before dense attention;
it is therefore not a Group 12 native same-low-bit execution path. The
persistent FP4 E2M1/NVFP4 implementation similarly materializes dense BF16
K/V before dense attention; it is not Group 13 native 4-bit execution.
Evidence: `utils/persistent_kv_storage.py`,
`utils/persistent_nvfp4_cache.py`, `utils/quant.py`, and
`wan/modules/causal_model_latentmem.py` in the persistent worktree.

No DraftMap is used by Groups 12–13. Groups 14–19 would require an explicit
route policy and exact operator contract before any run. The existing native
route machinery supports query-dependent phase assignment, but the current
LongLive adapter uses `sparsity_ratio=0.0`; see
`repos/LongLive-RAG/utils/anemoi_mpa.py:132-201` and
`repos/anemoi/anemoi/layers/attention/mpa/routing.py:23-235`.

## Decision

No incompatible substitute was run. The only completed result relevant to
this queue is the preceding storage-only W12 K-INT8/V-FP8 campaign. Groups
12–19 require either an authoritative experiment definition or a new kernel /
adapter contract before inference.
