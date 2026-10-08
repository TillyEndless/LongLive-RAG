# Group11 / LongLive-RAG wrapper reuse trace

## Scope and source revisions

This is a read-only source trace. No production source, checkpoint, dataset, or
existing result was changed and no inference was run.

| tree | commit | files |
|---|---|---|
| Original LongLive-RAG | `973884a3cd3ad4b314c3d4ab42274c52e7a0b22a` | `wan/modules/causal_model.py`, `wan/modules/causal_model_latentmem.py`, `pipeline/causal_inference.py` |
| Current Group11 | `25d7e7a52a50a28e1c7fbc7d4eaf1493a7fe0098` | `wan/modules/causal_model_latentmem.py`, `pipeline/causal_inference.py`, `utils/draftmap_retrieval.py` |

The phrase “original LongLive wrapper” needs one distinction. The non-latent
`causal_model.py` contains the native local-window wrapper and does not accept
retrieval IDs. The historical LongLive-RAG retrieval implementation is in
`causal_model_latentmem.py`; the reuse target below therefore uses the latter's
fetch/materialization path while preserving the native wrapper semantics.

## Original LongLive dataflow

```
hidden states
  -> q/k/v Linear + optional q/k RMSNorm
  -> causal-online RoPE for query and cached K
  -> local cache direct-insert or roll-and-insert
  -> evicted full BF16 K/V -> CPU frame lists
  -> descriptor selector in pipeline (original RAG path)
  -> selected frame IDs [B,k]
  -> CPU K/V .to(cuda, non_blocking=True)
  -> stack/view [B,k*frame_seqlen,H,D]
  -> causal-online RoPE for retrieved K
  -> sink + retrieved + local K/V torch.cat
  -> attention(q, k_cat, v_cat)
  -> flatten + output projection
  -> deferred cache bookkeeping
```

Verified source locations:

- `wan/modules/causal_model.py:88-154`: `CausalWanSelfAttention.forward`,
  Q/K/V projections, normalization, and the no-retrieval native path.
- `wan/modules/causal_model.py:180-330`: cache materialization, roll/direct
  insert, relative RoPE, and cache update info.
- `wan/modules/causal_model.py:336-356`: sink/local working-set construction,
  `torch.cat`, and BF16 attention dispatch.
- `wan/modules/causal_model.py:430-480`: `CausalWanAttentionBlock.forward`,
  self-attention call and output/FFN path.
- `wan/modules/attention.py:139`: attention dispatch (FlashAttention when
  available, otherwise PyTorch SDPA).
- `wan/modules/causal_model_latentmem.py:86-154`: RAG-capable attention
  constructor and Q/K/V setup.
- `wan/modules/causal_model_latentmem.py:265-360`: original pipeline cache
  roll/offload and update-info construction.
- `wan/modules/causal_model_latentmem.py:427-508`: retrieval working set,
  CPU fetch, RoPE, concatenation, attention, and output.
- `pipeline/causal_inference.py:267-304`: original descriptor retrieval
  decision, cosine scoring, and `torch.topk` selected IDs.
- `pipeline/causal_inference.py:432-469`: per-transformer-block cache
  ownership and CPU frame lists.
- `wan/modules/causal_model_latentmem.py:1186-1260`: deferred cache update,
  CPU history append, and pointer bookkeeping.

### Retrieval seam

The narrowest seam is the producer of `memory_indices` in
`pipeline/causal_inference.py:267-304`. Its contract is a tensor of shape
`[B,k]`, where values index the eligible CPU history pool after the sink and
recent-exclusion policy. The downstream consumer is
`causal_model_latentmem.py:440-465`, which indexes `cpu_k_frames` and
`cpu_v_frames` and performs the H2D copies. This is the preferred A-level
hook: replace only the selector, not fetch, RoPE, concatenation, or attention.

The next-best seam is a selector callback immediately before the existing
`memory_indices` consumer. Replacing the fetch path or the whole attention
wrapper is unnecessarily invasive.

## Current Group11 execution graph

```
hidden states
  -> Q/K/V projection (Group11 wrapper)
  -> _online_memory_indices()
       -> current Q -> _draft_frame_pools()
       -> GPU persistent historical Draft-K
       -> DraftMapChunkIndex.score_history()
       -> top-k selected history IDs
  -> local cache roll/direct insert + CPU full BF16 history append
  -> selected CPU K/V lookup
  -> per-selected-entry H2D
  -> stack/view working set
  -> retrieved/local/sink cat
  -> Group11 prepare_attention_kv()
  -> BF16 attention()
  -> output projection
  -> _apply_cache_updates(): CPU history + Draft-K state + pointers
```

Source-level details:

- `wan/modules/causal_model_latentmem.py:86-163`: Group11 state is attached
  to every attention block (`retrieval_backend`, traces, runtime mode, sparse
  ratio, reuse cache) in addition to the original module state.
- `:165-223`: `_online_memory_indices()` requires
  `retrieval_backend=draftmap_online`, builds candidate records from
  `gpu_draft_k_frames`, checks Draft-K device, calls `score_history`, then
  returns `topk(...).indices`. Draft-K is BF16 and `[B,T,H,D]` before block
  pooling; selected IDs are candidate-history indices.
- `:225-278`: `forward` computes Q/K/V and overrides the pipeline IDs with
  online DraftMap IDs when enabled; `_draft_frame_pools` constructs the
  transient current Draft-Q and update metadata.
- `:350-529`: cache roll/direct insert, evicted CPU full K/V frames, relative
  RoPE, and Draft-K update info.
- `:531-677`: selected archive/CPU lookup, optional reuse cache, H2D copies,
  stack/view, retrieved-history RoPE, and K/V concatenation.
- `:674-685`: Group11 preparation followed by the same BF16 attention call.
- `:701-714`: flatten/output projection and cache-update return.
- `:1174-1260`: `_apply_cache_updates` writes local K/V, appends CPU history,
  updates Draft-K lists and cache indices.
- `utils/draftmap_retrieval.py:15-20,22-46`: `DraftChunkRecord` and BF16
  pooled block registration; `:68-90` computes mean-pooled QK scores and
  row-softmax; `:92-109` aggregates block scores to chunk scores; `:111-115`
  returns `torch.topk` values and indices.

## Semantic mapping

See `results/group11_reuse_longlive_wrapper_mapping.csv` for the complete
stage-by-stage mapping. In brief, QKV, cache rolling, CPU full-history
ownership, selected-K/V fetch, stack/view, RoPE, concatenation, BF16 attention,
output projection, and pointer bookkeeping all already exist in LongLive-RAG.
Group11 genuinely adds Draft-Q pooling, GPU-persistent Draft-K, DraftMap
scoring/top-k, and Draft-K bookkeeping. Its wrapper duplicates the downstream
data path rather than merely supplying IDs to it.

## Selected-ID and ownership compatibility

| contract | original LongLive-RAG | Group11 |
|---|---|---|
| type | `torch.Tensor` | `torch.Tensor` |
| shape | `[B,k]` | `[B,k]` |
| value semantics | eligible CPU-history pool index, after sink/recent exclusion | eligible Draft-K candidate index, `0..eligible-1`, mapped to the same history order |
| consumer | `cpu_k_frames[idx]`, `cpu_v_frames[idx]` | currently used to index CPU/archive lists |

`DIRECTLY_COMPATIBLE = PARTIAL/YES WITH PROVENANCE GATE`. The tensor shape and
index convention are compatible only if Draft-K records and full CPU-history
records are appended, evicted, and excluded in the same order. The smallest
adapter is a provenance map from DraftMap `candidate_id` to canonical
LongLive history ID; it should be an identity map only after a runtime
assertion of aligned history IDs. Do not silently assume list position.

Draft-K lifecycle is not identical to the original cache: Group11 appends
`gpu_draft_k_frames` in `_apply_cache_updates` (`:1239-1253`) while the full
BF16 CPU history is appended at `:1213-1223`. A refactor must commit both
records atomically and preserve the same eviction/recent-exclusion boundary.

## Classification and duplication

| functionality | original implementation | Group11 implementation | classification | reusable? |
|---|---|---|---|---|
| Q/K/V projection | `causal_model*.py:147-154` | `latentmem.py:261-273` | DUPLICATED_LONGLIVE_LOGIC | YES |
| local cache roll/insert | `:265-425` | `:350-529` | DUPLICATED_LONGLIVE_LOGIC | YES |
| full CPU history ownership | `:285-297`, deferred update | `:386-402`, `:1213-1223` | SEMANTICALLY_DIFFERENT but largely duplicated | YES, with provenance |
| Draft-Q | absent | `:274-278`, `_draft_frame_pools` | ESSENTIAL_GROUP11_LOGIC | NO |
| Draft-K ownership | absent | `:451-469`, `:1239-1253` | ESSENTIAL_GROUP11_LOGIC | NO |
| selection | descriptor top-k `pipeline:267-304` | `_online_memory_indices:165-223` | ESSENTIAL_GROUP11_LOGIC | replace producer |
| CPU lookup/H2D | `latentmem.py:440-465` | `:562-635` | DUPLICATED_LONGLIVE_LOGIC | YES |
| stack/view/cat | `:456-481` | `:636-673` | DUPLICATED_LONGLIVE_LOGIC | YES |
| RoPE/layout | `:462-465` and `:408-411` | `:641-647` and `:491-515` | DUPLICATED_LONGLIVE_LOGIC | YES |
| BF16 attention | `:483-500` | `:679-685` | DUPLICATED_LONGLIVE_LOGIC | YES |
| output projection | `:502-508` | `:701-714` | DUPLICATED_LONGLIVE_LOGIC | YES |
| cache bookkeeping | `:1186-1260` | `:1174-1260` | PARTIAL duplicate + Draft-K extension | mostly |

Approximate current Group11 self-attention wrapper span is 165-714 (550 LOC,
including comments/branches). The essential Draft-RAG-specific portion is
approximately 165-223 plus Draft-K update/trace sections (~90-130 LOC). The
duplicated or semantically equivalent LongLive dataflow is approximately
250-330 LOC. These are source-span estimates, not executable LOC.

## Profiling evidence and current problems

The matched bounded capture reports Group11 4.914063 s versus LongLive
1.557214 s (delta 3.356849 s); wrapper CUDA boundary is 4.393256 versus
1.053488 s (delta 3.339768 s). BF16 attention itself is 0.528048 versus
0.551642 s, so the attention kernel is not the regression. Prior diagnostics
reported 30,270 fetch requests, 919 unique `(layer,chunk)` pairs, 96.96%
reuse-cache hit rate, and avoided 58,702 H2D calls / 281.3 GB; v5 also observed
540 `cudaHostAlloc` calls (2.543 s) and 12,282 `cudaStreamSynchronize` calls
(2.113 s). These are evidence of orchestration overhead, not proof that every
second is attributable to one site.

| issue | diagnosis | reuse effect |
|---|---|---|
| ~3.34 s wrapper delta | duplicated downstream wrapper/orchestration | LIKELY_REDUCE |
| repeated CPU lookup | Group11 loops selected entries and adds tracing/cache logic | LIKELY_REDUCE |
| repeated H2D orchestration | per-entry `.to(cuda)` and bookkeeping | LIKELY_REDUCE, not eliminate |
| repeated working-set materialization | separate stack/view/cat path | LIKELY_REDUCE |
| duplicate cat/layout/RoPE | same semantic operations are reimplemented | LIKELY_REDUCE |
| `cudaHostAlloc` activity | staging/allocation behavior | UNKNOWN until matched trace |
| synchronization activity | non-blocking copies still expose sync points | LIKELY_REDUCE, not guaranteed |
| DraftMap score/top-k | genuinely Group11-specific | NOT_AFFECT |
| Draft-K maintenance | genuinely Group11-specific | NOT_AFFECT |
| BF16 attention kernel | already comparable/slightly faster in Group11 capture | NOT_AFFECT |

Maximal reuse cannot remove DraftMap, Draft-Q, Draft-K maintenance, or the
retrieval-policy bookkeeping. The intended cost model is
`T_LongLive_wrapper + T_DraftMap_selector + small ID adapter`, not zero
overhead.

## Minimal redesign (design only)

1. Keep LongLive's `CausalWanSelfAttention.forward` and its cache/fetch,
   RoPE, working-set, attention, output, and update path.
2. Add a `retrieval_policy.select(query, cache_metadata) -> [B,k]` callback at
   the existing `memory_indices` producer seam.
3. Maintain a per-block Draft-K sidecar with canonical history IDs. On each
   cache commit, append the full BF16 history record and Draft-K record in one
   transaction; on eviction, remove both together.
4. Build Draft-Q from the current Q without changing the attention Q/K/V.
5. Convert DraftMap top-k candidate IDs through the sidecar provenance map.
6. Pass those IDs into unchanged LongLive fetch/materialization code.
7. Add assertions for device, dtype, candidate count, recency exclusion, and
   ID alignment. Do not insert an independent `torch.cat`/RoPE/attention path.

### Validation gates

1. Selected IDs equal current Group11 for the same deterministic call.
2. Final BF16 attention Q/K/V inputs match current Group11.
3. Attention output matches.
4. Case01 latent/video output matches.
5. Matched blocks 6-7 profile: wrapper, DraftMap, fetch, H2D, working-set,
   and BF16 attention separately. Success is semantic equivalence plus wrapper
   overhead moving toward LongLive, not speed alone.

## Final design verdict

`CURRENT_GROUP11_WRAPPER_ARCHITECTURE = custom latentmem attention wrapper that embeds DraftMap selection and reimplements LongLive cache/fetch/working-set/attention dataflow.`

`MAIN_CURRENT_IMPLEMENTATION_PROBLEMS = duplicated fetch/materialization/RoPE/cat/attention orchestration, per-entry transfer/bookkeeping, and exposed allocation/synchronization overhead.`

`CURRENT_WRAPPER_DUPLICATES_LONGLIVE = YES (substantial downstream path)`

`MAXIMAL_LONGLIVE_REUSE_FEASIBLE = YES, with a canonical-ID/provenance adapter`

`EXPECTED_TO_REMOVE_CUSTOM_FETCH_PATH = YES`

`EXPECTED_TO_REMOVE_CUSTOM_WORKINGSET_PATH = YES`

`EXPECTED_TO_REMOVE_CUSTOM_LAYOUT_PATH = YES`

`EXPECTED_TO_REMOVE_CUSTOM_ATTENTION_DISPATCH = YES`

`EXPECTED_TO_REDUCE_WRAPPER_LATENCY = LIKELY, not guaranteed`

`UNAVOIDABLE_REMAINING_GROUP11_OVERHEAD = Draft-Q, Draft-K, DraftMap scoring/top-k, selected-ID provenance adapter, Draft-K maintenance, and policy bookkeeping.`

`CAN_TARGET_ARCHITECTURE_BE: LongLive wrapper + Draft-RAG retrieval-policy plugin = YES`

`PRIMARY_IMPLEMENTATION_RECOMMENDATION = refactor at the selected-ID producer seam; preserve LongLive's downstream path and validate exact IDs/QKV/outputs before measuring latency.`
