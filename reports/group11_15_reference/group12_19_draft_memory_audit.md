# Group 12–19 DraftMap memory audit

Date: 2026-09-27. Read-only audit of the corrected campaign definitions.

## Validated DraftMap implementation

The only validated DraftMap implementation is the Group 11 BF16 retrieval
path:

- `repos/LongLive-RAG/wan/modules/causal_model_latentmem.py:85-100`
  mean-pools BF16 K into 64-token blocks. The per-frame pooled layout is
  `[B, H, ceil(frame_tokens/64), D]`, BF16.
- `pipeline/causal_inference.py:431-432` owns `cpu_draft_k_frames` and
  `local_draft_k_frames` in the latent-memory cache.
- `causal_model_latentmem.py:145-224` uses historical Draft-K plus transient
  current Draft-Q for scoring. Current Draft-Q is not persisted.
- `utils/draftmap_retrieval.py:31-44,48-91` confirms BF16 storage, 64-token
  pooling, and Anemoi-style block score construction.

The validated Group 11 result measured persistent Draft-K memory of
`0.007724761963 GiB`; that value is not blindly reused here because the
corrected native cache path has a different owner and currently has no Draft
state.

## Native persistent path audit

`pipeline/causal_inference.py:431-445` creates only
`PersistentAnemoi8BitCache`; `utils/persistent_anemoi_8bit_cache.py:12-80`
owns packed K8/V8, scales, validity metadata, and chunk IDs, with no Draft-K
or Draft-Q fields. `wan/modules/causal_model.py:220-249` sends the current
chunk and persistent chunks directly to `persistent_anemoi_int8_attention`;
there is no DraftMap computation, active-window importance tensor, or route
observer in that branch.

Therefore, for the corrected Groups 12–19 definitions:

| Item | Finding |
|---|---|
| Historical Draft-K persisted | **No** in the native persistent cache |
| Historical Draft-Q persisted | **No**; not required by the validated Group 11 algorithm |
| Current Draft-Q transient | **No** in the native persistent branch; it is absent |
| Block size | 64 tokens in the separate Group 11 implementation |
| Draft dtype/layout | BF16 `[B,H,blocks,D]` after pooling in Group 11 |
| Native 8-bit route | Fixed 100% INT8/FP8 in `persistent_anemoi_int8_attention` |
| Native 4-bit route | No persistent direct consumer found |
| Mixed 8/4/0 route | No persistent direct consumer found |
| RAG retrieval | Disabled in the corrected Group 12–19 configs |

## Consequence

No Group 12–19 experiment can truthfully be launched from the current native
persistent worktree: doing so would omit the required active DraftMap
infrastructure and would turn Groups 14–19 into unsupported semantics.
Adding DraftMap state to the native cache is an implementation change, not a
configuration-only launch. No source was modified and no Group 12–19
inference was run.
