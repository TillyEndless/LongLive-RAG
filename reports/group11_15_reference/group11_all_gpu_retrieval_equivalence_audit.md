# Group 11 all-GPU Draft-K retrieval-equivalence audit

Date: 2026-09-27, reopened 2026-09-27. Read-only audit. No inference was
rerun, no MP4 was modified, and no algorithmic fix was applied.

## Scope and evidence

Compared case 01 traces:

- Old: `results/draftmap_rag_online_w12_10case/rank0-0-0_lora_retrieval_trace.csv`
- All-GPU: `results/group11_all_gpu_draft/w12/rank0-0-0_lora_retrieval_trace.csv`

Each trace has 5,220 retrieval calls. The trace stores candidate IDs,
post-softmax aggregated selected scores, and selected IDs, but not raw Q/K
tensors, raw logits, pooled tensors, or per-history checksums. Those fields
therefore remain explicitly unavailable rather than inferred.

## First divergence

The first selected-ID divergence is trace row/call 168 (1-based):

| Field | Value |
|---|---|
| Case | case_01 |
| Generation unit | 21 |
| Denoising step | 937 |
| Transformer layer | 17 |
| Attention/retrieval call | 168 in trace order |
| Candidate IDs | identical; `[0, 1, 2, 3]` |
| Old selected IDs | `[1, 2, 3, 0]` |
| All-GPU selected IDs | `[1, 3, 2, 0]` |

At this call all four candidates are retained, so the saved selected-score
lists provide the complete candidate score table:

| history_id | old score | new score | difference | old rank | new rank |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.234619140625 | 0.229858398438 | -0.004760742187 | 4 | 4 |
| 1 | 0.277343750000 | 0.279052734375 | +0.001708984375 | 1 | 1 |
| 2 | 0.243652343750 | 0.243164062500 | -0.000488281250 | 3 | 3 |
| 3 | 0.244262695312 | 0.247924804688 | +0.003662109375 | 2 | 2 |

The selected-ID difference is the ordering of IDs 2 and 3. The score margin
between those two is 0.0006103515625 old versus 0.004760742187 new, so this
particular first divergence is not a pure exact-tie artifact.

## Source-level comparison

The relevant code difference is in
`wan/modules/causal_model_latentmem.py`, symbol
`CausalWanSelfAttention._online_memory_indices`:

- Old lines 149–163: read `cpu_draft_k_frames`, construct records, then call
  `r.draft_k.to(query.device, non_blocking=True)` before scoring.
- All-GPU lines 156–170: read `gpu_draft_k_frames`, require the tensors already
  match `query.device`, and score them without the copy.
- Both variants call `utils/draftmap_retrieval.py:107-109`,
  `DraftMapChunkIndex.score_history`, which calls `score_blocks`.
- `score_blocks` at `utils/draftmap_retrieval.py:87-90` converts Q/K to FP32,
  performs `torch.matmul`, scales by `1/sqrt(d)`, then row-softmaxes and casts
  to FP16.

The previous audit incorrectly inferred that the old path scored on CPU. The
old `.to(query.device)` result is CUDA when `query.device` is CUDA, and the
common `score_blocks` function receives that CUDA tensor. The all-GPU path also
passes CUDA tensors. Thus both paths are source-proven to score on CUDA; the
CPU-vs-CUDA explanation is not established and is retracted below.

The saved traces do not record runtime devices, Q/K byte checksums, raw logits,
pooled tensors, or matmul/TF32 state. They cannot prove whether the remaining
difference is a Draft-Q difference, Draft-K lifecycle difference, CUDA
numerical-policy difference, or earlier generation-state difference.

The cache-list lifecycle itself is unchanged in order:

- `pipeline/causal_inference.py:431-432` initializes the Draft-K list.
- `causal_model_latentmem.py:347-416` selects evicted and new Draft-K chunks.
- Old line 409 copies evicted Draft-K to CPU; all-GPU line 415 retains the same
  tensor on GPU.
- Old `:1049` and all-GPU `:1072` append evicted chunks to the historical list.
- The all-GPU traces have identical candidate IDs in every case and every
  call, which verifies that the history eligibility/order boundary is not the
  observed divergence.

## Case-01 trace findings

| Check | Finding | Evidence |
|---|---|---|
| Candidate IDs | identical | 5,220/5,220 calls in case 01; same for all 10 cases |
| Draft-Q tensor identity | NOT_AVAILABLE | no tensor capture/checksum in either trace |
| Draft-K tensor identity | NOT_AVAILABLE | no tensor capture/checksum in either trace |
| Draft-K/history alignment | valid at trace level | identical candidate sequences; unchanged append/eviction order |
| Raw logits | NOT_AVAILABLE | evaluator trace stores no logits |
| Aggregated scores | differ | first divergence table; selected-score records differ widely |
| Top-k behavior | deterministic for each device result | differences follow score ranking, not a source index offset |

Across case 01, 4,383 of 5,220 selected-ID rows differ; 707 additional rows
have order-only differences. Candidate IDs remain identical. Across cases 01–10,
candidate IDs are identical in all 52,200 compared rows, while selected sets
differ in 3,676–4,744 rows per case.

## Classification

```text
FIRST_DIVERGENCE_CASE = case_01
FIRST_DIVERGENCE_STEP = 937
FIRST_DIVERGENCE_LAYER = 17
FIRST_DIVERGENCE_CALL = 168 (historical trace only; not reproduced in instrumented pair)

CANDIDATE_IDS_IDENTICAL = YES
DRAFT_Q_IDENTICAL = YES (instrumented calls 150-180)
DRAFT_K_IDENTICAL = YES at score boundary (instrumented calls 150-180)
DRAFT_K_HISTORY_ALIGNMENT_VALID = YES
RAW_SCORE_IDENTICAL = NO
AGGREGATED_SCORE_IDENTICAL = NO

OLD_SCORE_DEVICE = CUDA (runtime-confirmed: cuda:0)
NEW_SCORE_DEVICE = CUDA (runtime-confirmed: cuda:0)
OLD_NEW_DRAFT_Q_EQUAL = YES (instrumented calls 150-180)
OLD_NEW_DRAFT_K_EQUAL = YES at score boundary (instrumented calls 150-180)
OLD_CPU_ROUNDTRIP_DRAFT_K_EQUAL = YES (tested GPU->CPU->GPU BF16 round-trip)
CONCATENATED_K_EQUAL = YES (instrumented calls 150-180)
FP32_INPUTS_EQUAL = YES (instrumented calls 150-180)
RAW_LOGITS_EQUAL = YES (instrumented calls 150-180)
SOFTMAX_EQUAL = YES (instrumented calls 150-180)
AGGREGATED_SCORES_EQUAL = YES (instrumented calls 150-180)
FIRST_TRUE_NUMERICAL_DIVERGENCE_STAGE = NONE OBSERVED in instrumented pair
ROOT_CAUSE = OTHER: historical canonical runtime/provenance difference unresolved
PREVIOUS_CPU_VS_GPU_SCORING_CLAIM = RETRACTED
FIX_APPLIED = NO
CASE01_ALL_SELECTED_IDS_IDENTICAL_AFTER_FIX = YES (paired gate passed without fix)
CANONICAL10_RERUN_REQUIRED = YES (not launched)
```

The only established fact about the historical artifacts is that their final
saved aggregated scores differ. The instrumented paired implementation gate
passes without a code fix, so the old all-GPU placement change is not the
demonstrated cause.

## Reopened-device conclusion

The source path proves both score inputs are CUDA when the model query is CUDA:

- Old `wan/modules/causal_model_latentmem.py:160-163` performs
  `r.draft_k.to(query.device, non_blocking=True)` and passes the result to
  `score_history`.
- All-GPU `wan/modules/causal_model_latentmem.py:169-172` verifies the resident
  tensor is already on `query.device` and passes it to `score_history`.
- `utils/draftmap_retrieval.py:87-90` performs the FP32 matmul on the device
  of those tensors.

This is source evidence for the historical runs. A new isolated paired
instrumented case-01 run has now supplied the requested runtime evidence.

## Minimal instrumented paired case-01 result

The harness ran one canonical prompt against the old and all-GPU worktrees,
with identical seed/configuration and temporary outputs outside canonical
result directories:

`/home/zju/work/zxl/tmp_group11_case01_instrument/{old3,new}/`

Runtime state at call 168 was identical: both score devices were `cuda:0`, Q
was BF16 with shape `[1,4680,12,128]` and stride
`[7188480,1536,128,1]`, matmul precision was `highest`, CUDA TF32 was false,
autocast was false, and deterministic algorithms were enabled. Both runs used
PyTorch `2.7.0+cu128` and CUDA `12.8`.

For every target call 150–180, SHA256 matched for Q, pooled Q, concatenated K,
FP32 Q/K, raw logits, scaled logits, softmax, and aggregated scores. The full
paired traces also matched exactly:

```text
retrieval_calls = 5220
candidate_history_ids_equal = 5220/5220
selected_history_ids_equal = 5220/5220
selected_scores_equal = 5220/5220
```

The old score-boundary Draft-K tensors and all-GPU tensors matched in shape,
dtype, stride, contiguity, and SHA256 for calls 150–180. The explicit
GPU→CPU→GPU BF16 round-trip was exact for every captured candidate:
`torch.equal=True`, max absolute difference `0`, mean absolute difference `0`,
and differing-element count `0`.

Creation-time snapshots were not captured; equality is established at the
first common scoring boundary. The historical canonical divergence is not
reproduced by the old-versus-all-GPU implementation change. Its runtime or
provenance cause remains unresolved.

## Required next step

The case-01 equivalence gate passes for the tested old/all-GPU pair. A
canonical-10 rerun is therefore required for a fair all-GPU quality
comparison, but it has not been launched here. The historical divergence
should not be attributed to Draft-K placement without matching the original
runtime, binary, and generation provenance.
