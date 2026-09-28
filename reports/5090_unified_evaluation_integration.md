# 5090 unified evaluation integration

Status: isolated integration candidate validated without inference.

Active worktree: `/home/zju/work/zxl/LongLive-RAG-common-group16_19-v2`
Active branch: `integration/group16-19-common-v2`
Active HEAD: `0cdf3ea3bf3292730a3867e9c6e3c169f1d8a44e`
Active worktree dirty: yes; left untouched.
Isolated worktree: `/home/zju/work/zxl/LongLive-RAG-5090-eval-integration`
Integration branch: `eval-pipeline-5090-integration`
H200 source commit: `410948eb7e328513e7751ac2d44193115363cee4`

The H200 commit cherry-picked cleanly in the isolated worktree as `df1b403`.
No path under `results/`, `checkpoints`, `repos_anemoi`, or `wan_models` was
imported. The active 5090 worktree had uncommitted Group16/Tier-A changes and
was not modified. No relevant inference or GPU process was found initially.

The 5090 adapter maps existing measured fields without changing inference:

```text
e2e_latency_s -> E2E_INFERENCE_S
transformer_latency_s -> TRANSFORMER_S
wrapper_latency_s -> SELF_ATTN_WRAPPER_S
native_kernel_ms -> ATTENTION_KERNEL_S (seconds)
h2d_ms/exposed_h2d_s -> fetch fields
persistent_*_bytes -> corresponding GiB memory fields
```

Missing quality, peak allocator, and compression measurements remain
`NOT_AVAILABLE`.

Static validation passed:

```text
python -m evaluation.unified_pipeline --help
python -m evaluation.aggregate_table --help
python -m py_compile evaluation/*.py unified_evaluation_schema.py
git diff --check
```

The no-inference reuse smoke consumed the existing 5090 Group16 profile at
`/home/zju/work/zxl/LongLive-RAG-common-group16_19-v2/results/group16_tierA_D_profile.json`.
It wrote only to `/tmp/longlive-rag-5090-eval-reuse-20260928/` and generated
JSON, CSV, Markdown, and aggregate CSV outputs.

Observed validity:

```json
{"QUALITY_VALID": false, "MEMORY_VALID": true, "LATENCY_VALID": true,
 "PROVENANCE_VALID": true, "INFERENCE_VALID": true,
 "FINAL_ROW_VALID": false}
```

`FINAL_ROW_VALID=false` is expected because the reused profile has no quality
artifact. No H200 measurement was copied or relabeled as 5090.

Existing Group11/11.1–11.4, Group12–15, Group16–18, and disabled Group19
configs were discovered. The unified layer is generic and does not redefine
algorithm or sparse/KV semantics. Existing result formats are not uniform
enough to claim complete per-group reusable validity without adding
measurements; unavailable fields are recorded as missing.

```text
INFERENCE_STARTED = NO
GPU_PROCESS_INTERRUPTED = NO
ACTIVE_5090_WORKTREE_MODIFIED = NO
H200_RESULTS_IMPORTED_AS_5090 = NO
EXPERIMENT_RESULTS_OVERWRITTEN = NO
GROUP19 = DISABLED
```

Do not merge this branch into the active 5090 worktree yet.
