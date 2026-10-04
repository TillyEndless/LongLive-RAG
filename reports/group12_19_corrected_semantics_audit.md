# Group12–Group19 corrected semantics audit

Audit target: `/data/zxl/LongLive-RAG-group11_15_h200` on H200.

## Global invariant

The previous H200 implementation stored evicted history through
`utils/compressed_history_archive.py` as INT8/FP8 or NVFP4 on CPU. That is
`WRONG_CPU_COMPRESSION` for the corrected experiment definition. The shared
archive has now been changed to authoritative BF16 K/V CPU storage. GPU-side
fake quantization is applied only after the selected BF16 payload reaches the
GPU, and the final attention operands remain BF16.

| Group | CPU history | H2D | GPU KV | Compute/routing | Smoke | Old artifacts |
|---|---|---|---|---|---|---|
| 12 | BF16 K/V | BF16 | INT8 K / FP8 E4M3 V working set | BF16 attention | synthetic PASS; real pending | invalid: compressed CPU archive |
| 13 | BF16 K/V | BF16 | NVFP4 K/V working set | BF16 attention | synthetic PASS; real pending | invalid: compressed CPU archive |
| 14 | BF16 K/V | BF16 | Group12 GPU working set | BF16 retained routing | synthetic path covered; real pending | invalid: compressed CPU archive |
| 15 | BF16 K/V | BF16 | Group13 GPU working set | BF16 retained routing | synthetic path covered; real pending | invalid: compressed CPU archive |
| 16 | NOT_IMPLEMENTED in H200 checkout | — | RTX5090 native path only | blocked | NOT_AVAILABLE | not reusable |
| 17 | NOT_IMPLEMENTED in H200 checkout | — | RTX5090 native path only | blocked | NOT_AVAILABLE | not reusable |
| 18 | NOT_IMPLEMENTED in H200 checkout | — | RTX5090 native path only | blocked | NOT_AVAILABLE | not reusable |
| 19 | NOT_IMPLEMENTED in H200 checkout | — | RTX5090 native path only | blocked | NOT_AVAILABLE | not reusable |

## Source evidence

- `/data/zxl/LongLive-RAG-group11_15_h200/utils/compressed_history_archive.py`:
  eviction archive and fetch lifecycle. Corrected to retain CPU BF16 tensors
  and assert BF16 on fetch.
- `/data/zxl/LongLive-RAG-group11_15_h200/wan/modules/causal_model_latentmem.py`:
  `_apply_cache_updates` archives evicted K/V and the attention path fetches
  selected records before `prepare_attention_kv`.
- `/data/zxl/LongLive-RAG-group11_15_h200/utils/h200_group_runtime.py`:
  corrected Group12/14 fake INT8/FP8 and Group13/15 fake NVFP4 transforms now
  operate on GPU BF16 working tensors and return BF16 for attention.
- `/data/zxl/LongLive-RAG-group11_15_h200/utils/persistent_draftmap.py`:
  retained-block DraftMap routing for Groups14/15.

## Smoke result

Synthetic BF16 archive → BF16 fetch → GPU-side fake-quant/dequant passed for
the INT8/FP8 and NVFP4 modes. It did not launch model inference. Canonical
runs were not launched.

## Required status flags

```text
GROUP11_MODIFIED = NO
QPREV_MODIFIED = NO
LONGLIVE_REUSE_MODIFIED = NO
CANONICAL_RUNS_LAUNCHED = NO
ALL_GROUPS_CPU_HISTORY_BF16 = YES for implemented H200 Groups12–15
ANY_COMPRESSED_CPU_HISTORY_REMAINING = NO in corrected source path
OLD_GROUP12_CPU_MEMORY_VALUE_VALID = NO
OLD_GROUP13_CPU_MEMORY_VALUE_VALID = NO
OLD_CPU_COMPRESSION_RATIOS_VALID = NO
```

Old outputs remain untouched and must be labeled
`LEGACY_WRONG_SEMANTICS` / `INVALID_FOR_CORRECTED_GROUP12_19` until rerun.
