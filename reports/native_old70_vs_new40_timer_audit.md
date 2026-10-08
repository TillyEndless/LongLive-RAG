# Native W6/W12 old ~70 s vs new ~40–43 s timer audit

## Executive conclusion

The historical approximately-70-second values were found in the old
`/data/zxl/LongLive-RAG/results/longlive_temporal_precision/` BF16 pilot.
They are per-prompt `e2e_latency_s` values from a run with `profile: true`,
not process-wall measurements. The old profile path synchronizes at every
generated block. The new matched case_01 runs use the H200 worktree and its
low-perturbation unified profiler. No actual model/code speedup is proven.

## Required fields

| Field | Finding |
|---|---|
| OLD_70S_SOURCE | `.../longlive_temporal_precision/window{6,12}/bf16/videos/*_lora_runtime.json` and `inference.log` |
| OLD_70S_METRIC_NAME | `e2e_latency_s` / per-prompt tqdm elapsed |
| OLD_70S_METRIC_CLASS | E2E inference interval with profiling perturbation, not process wall |
| OLD_TIMER_START | old `inference.py:241`, before `pipeline.inference()` |
| OLD_TIMER_END | old `inference.py:251`, after `torch.cuda.synchronize(device)` |
| NEW_TIMER_START | H200 `pipeline/causal_inference.py:207` |
| NEW_TIMER_END | H200 `pipeline/causal_inference.py:470-485` |
| TIMER_BOUNDARY_COMPATIBLE | Mostly yes for the main inference interval; instrumentation differs |
| WORKLOAD_COMPATIBLE | No: old 128-prompt pilot vs new matched case_01 |
| HARDWARE_COMPATIBLE | Unknown |
| OLD_PROFILER_PERTURBATION_RISK | High |
| NEW_PROFILER_PERTURBATION_RISK | Low |
| ACTUAL_MODEL_SPEEDUP_PROVEN | No |
| DIRECT_LATENCY_COMPARISON_VALID | No |

## Historical artifacts

| Artifact | Window | Prompt index | e2e_latency_s |
|---|---:|---:|---:|
| `LongLive-RAG/results/longlive_temporal_precision/window6/bf16/videos/rank0-0-0_lora_runtime.json` | 6 | 0 | 62.701 |
| `.../window6/bf16/videos/rank0-5-0_lora_runtime.json` | 6 | 5 | 69.609 |
| `.../window12/bf16/videos/rank0-0-0_lora_runtime.json` | 12 | 0 | 58.723 |
| `.../window12/bf16/videos/rank0-5-0_lora_runtime.json` | 12 | 5 | 68.078 |

The old logs/configs show `Number of prompts: 128`, seed 0, 120 latent
frames, four denoising steps, and the same base/LoRA checkpoint paths. The
old prompt source is `moviegenbench_128_refined.txt`; the new run uses one
explicit `native_case01.txt`. Thus prompt index 0 is textually aligned, but
the artifacts are not a paired benchmark. The old logs do not preserve enough
GPU/CUDA/PyTorch metadata to prove hardware equivalence.

## Exact timer semantics

Old `inference.py:241-251` starts the host `perf_counter` after noise/prompt
preparation and immediately before `pipeline.inference()`. It ends after the
pipeline returns and one explicit CUDA synchronization. It includes text
conditioning, latent preparation, denoising, VAE decode, and that final sync;
it excludes model/checkpoint load, process startup, later video writing,
result serialization, and evaluation.

With `profile: true`, old
`pipeline/causal_inference.py:385-388` synchronizes once per generated block;
additional profile synchronizations are at `394-399` and `421-425`. These
fall inside the old E2E interval. This is the primary observed perturbation
mechanism.

New `causal_inference.py:207` starts before the text encoder. The end at
`470-485` follows the VAE path and unified profiler finalization, before
`write_video` in `inference.py`. It includes the same broad inference stages.
The unified profiler at
`utils/unified_latency_profiler.py:24-57` records CUDA events without
per-call synchronization and synchronizes once at finalization. The matched
native run used `profile=False`, so it did not use the old per-block profile
sync path.

## Stage comparison

| Stage | Old | New | Notes |
|---|---|---|---|
| Process startup/model load | No | No | Outside timers |
| Text encoding/latent preparation | Yes | Yes | Inside pipeline interval |
| Generation/Transformer | Yes | Yes | Shared broad boundary |
| Per-block profiler sync | Yes when `profile=true` | No | Known instrumentation difference |
| VAE decode | Yes | Yes | Before pipeline return |
| Video write/evaluator | No | No | Outside both |
| Profiler finalization | Old profile syncs | One unified finalize sync | Different overhead |

`TIMER_BOUNDARY_COMPATIBLE` is therefore mostly yes at the coarse stage level,
but not compatible at the synchronization/instrumentation level.

## New matched-case decomposition

| Window | E2E s | Transformer s | Self-attention wrapper s | Attention kernel s | NON_TRANSFORMER_E2E_S |
|---:|---:|---:|---:|---:|---:|
| 6 | 40.435522 | 27.766429 | 14.990465 | 6.185424 | 12.669093 |
| 12 | 43.004939 | 30.181653 | 17.479825 | 10.659426 | 12.823285 |

Text encoder, VAE, video encode, and I/O sub-times are not separately
available in the matched artifacts. No values are imputed.

## Gap accounting

Using representative old prompt-index-5 values:

- W6: `69.608826 - 40.435522 = 29.173304 s`
- W12: `68.077900 - 43.004939 = 25.072961 s`

Using prompt index 0 instead gives 22.265153 s and 15.718072 s. This spread
demonstrates case/workload dependence. No paired counterfactual run isolates
the cost of removing old per-block synchronization, so timer-boundary,
workload, sync/profiler, hardware/contention, code-change, and unexplained
numeric attribution fields are `NOT_IDENTIFIABLE`.

## Final answers

1. The old ~70 s metric is old per-prompt `e2e_latency_s` from the BF16
   temporal-precision pilot with `profile:true`.
2. Its timer is old `inference.py:241-251` around `pipeline.inference()` plus
   a final CUDA sync.
3. The new timer is H200 pipeline `causal_inference.py:207` to `470-485`.
4. They share the broad inference interval but not the same workload or
   instrumentation conditions.
5. Both exclude setup, video write, serialization, and evaluator; both include
   text conditioning, generation, VAE, and final synchronization. Old profile
   additionally includes repeated per-block synchronization.
6. Old synchronization can explain part of the gap, but its exact magnitude is
   not measured by this read-only audit.
7. There is no evidence proving actual model/code speedup.
8. Keep old values only as a separately labeled profiled historical metric;
   do not mix them into the new table.
9. Use the new unified E2E definition for final comparisons, with matched-case
   versus canonical10 provenance stated explicitly.

## Field-mapping correction

The former `Wrapper s` column was mislabeled. Values 12.669093 s (W6) and 12.823286 s (W12) are derived `NON_TRANSFORMER_E2E_S = E2E_INFERENCE_S - TRANSFORMER_S`. The authoritative self-attention wrapper measurements remain 14.990465 s and 17.479825 s, from `native_case01_profile.json:self_attention_wrapper_s`, generated by `native_case01_profiler_v3.py:44`. No raw profile values were changed.
