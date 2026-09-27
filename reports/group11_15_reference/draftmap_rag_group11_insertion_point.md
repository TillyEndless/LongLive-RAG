# Group 11 DraftMap-RAG insertion-point audit

Date: 2026-09-26
Scope: read-only audit before implementation. No inference was run and no
persistent-cache source or result was modified.

## Original call graph

```text
CausalInferencePipeline.inference
  -> denoise current block
  -> append current latent descriptors
  -> compute memory_indices with latent cosine + torch.topk
  -> WanDiffusionWrapper.forward(memory_indices)
  -> CausalWanModelLatentMem._forward_inference
  -> CausalWanSelfAttention.forward
  -> fetch selected cpu_k_frames/cpu_v_frames
  -> causal_online_rope historical K
  -> concatenate sink + retrieved + local BF16 K/V
  -> original wan.modules.attention.attention
  -> _apply_cache_updates
  -> evicted BF16 frames appended to CPU archive
```

## Exact symbols and boundaries

| Responsibility | Source symbol and lines | Finding |
|---|---|---|
| Latent/context registration | `pipeline/causal_inference.py:300-308`, `CausalInferencePipeline.inference` | Encodes each completed denoised frame with AE or `avg_pool`; appends one descriptor per frame. |
| Historical cache registration | `wan/modules/causal_model_latentmem.py:925-980`, `_apply_cache_updates` | Rolls/inserts BF16 K/V and appends evicted frame tensors to `cpu_k_frames`/`cpu_v_frames`. |
| Candidate construction | `pipeline/causal_inference.py:225-240`, `CausalInferencePipeline.inference` | Candidate pool is the evicted archive after `recent_exclude`; IDs are descriptor/archive indices. |
| Original score | `pipeline/causal_inference.py:241-247` | L2-normalized descriptor cosine via `torch.bmm`. |
| Top-K | `pipeline/causal_inference.py:247`, same symbol | `torch.topk(sims, k=min(memory_size,num_eligible))`. |
| Selected IDs | `pipeline/causal_inference.py:249-257` | Converts pool indices to global frame indices and logs selection. |
| CPU fetch | `wan/modules/causal_model_latentmem.py:405-426`, `CausalWanSelfAttention.forward` | Uses selected archive indices to restore one frame per selected chunk for every batch item. |
| Context ordering | `wan/modules/causal_model_latentmem.py:436-450` | Concatenates sink, retrieved memory, then local BF16 window. |
| Final attention | `wan/modules/causal_model_latentmem.py:451-464` | Calls the existing `attention(roped_query,k_cat,v_cat)` path. |

## Narrow replacement boundary

The smallest valid boundary is the score/ranking block in
`CausalInferencePipeline.inference:225-257`:

```text
candidate archive + recent exclusion
  -> retrieval_backend.score(candidate metadata, current draft query)
  -> existing torch.topk
  -> existing memory_indices
  -> unchanged CausalWanSelfAttention fetch/assembly/BF16 attention
```

The original backend remains the default. DraftMap must replace only the
values used to form `memory_indices`; it must not replace `topk`, CPU fetch,
cache update, context assembly, or attention.

## Anemoi semantics available for reuse

The native DraftMap implementation is in
`repos/anemoi/anemoi/layers/attention/mpa/executor.py:1013-1150`:
it prepares pooled Q/K, optionally max-pooled Q/K, and calls
`sm120_h3_draft_probability`. The extension-free reference formula is
`anemoi/layers/attention/mpa/routing.py:64-103`: pooled Q/K dot product,
`1/sqrt(head_dim)` scaling, row softmax, and optional diagonal-Jensen second
moment correction. This Group 11 path must use BF16 source Q/K and must not
invoke Anemoi attention execution or low-bit preparation.

## Block-to-chunk aggregation status

No authoritative block-to-chunk score rule was found in the LongLive source,
reports, prior campaign artifacts, or Anemoi utilities. The prior deferred
note `results/draft_v2_campaign/07_draft_retrieval/token_level_next_step.md`
requires deterministic whole-chunk aggregation but does not define mean, max,
sum, or another rule.

Therefore:

```text
CHUNK_AGGREGATION = BLOCKED_NEEDS_DEFINITION
```

The implementation may still prepare transient current Draft-Q and persistent
historical Draft-K storage,
candidate block scores, and exact block-to-LongLive-chunk mapping, but the
canonical 10-case Group 11 run must not start until the aggregation rule is
authoritatively defined. No rule is guessed here.

## Controlled-comparison invariants

The intended comparison keeps the following identical: archive ownership,
chunk IDs, candidate pool, recent exclusion, Top-K budget and implementation,
CPU offload/fetch, context ordering, BF16 K/V/Q, and original attention. Only
the score source changes from latent cosine to the approved DraftMap-derived
chunk score.
