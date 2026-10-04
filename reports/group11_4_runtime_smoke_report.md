# Group11.4 runtime smoke report

## Result

**PASS — independent runtime smoke completed.**

No canonical 10-case campaign was started. No Group12–15 inference was run.

## Configuration

- Repository commit: `01212440f3d75e237d89cf3c3b14ea8f92247cbc`
- GPU: H200 NVL, GPU0
- GPU free before launch: approximately 143 GiB
- Case: case01 prompt
- `group_id: 11`
- `retrieval_backend: draftmap_online`
- `retrieval_query_mode: current_q`
- `group11_fetch_mode: next_layer_prefetch`
- `local_attn_size: 12`
- `memory_size: 6`
- `recent_exclude: 5`
- `num_output_frames: 30`
- `smoke_10block: true`
- seed: 0
- final attention: BF16

Output directory:

`/data/zxl/LongLive-RAG-group11_15_h200/results/group11_4_runtime_smoke/`

## Runtime counters

| Counter | Value |
|---|---:|
| attention calls | 1,500 |
| DraftMap retrieval calls | 720 |
| prefetched IDs | 3,161 |
| prefetch hits | 2,620 |
| correction fetches | 3,270 |
| wasted prefetched IDs | 541 |
| peak transient prefetch GPU bytes | 115,015,680 |
| persistent GPU KV bytes | 3,450,470,400 |
| persistent GPU Draft-K bytes | 41,472,000 |
| native low-bit kernel | NO |

The predicted-ID accounting is internally consistent: `3,161 = 2,620 + 541`.
Correction fetches are separate because they are IDs in the next layer's true
selection that were not available as a prefetch hit.

## Correctness checks

- The current layer computes authoritative current-Q IDs before scheduling the next-layer prefetch.
- The next layer reconciles stale entries before consumption.
- Only exact `(layer, history_id)` hits are consumed from transient prefetch state.
- Missing IDs take the normal correction-fetch path.
- Final attention is constructed from the current layer's `memory_indices`, not from a union with prefetched IDs.
- Prefetch bytes are reported separately as transient memory and are not included in persistent GPU KV accounting.
- Group12–15 compressed-history archive and persistent low-bit ownership paths are unchanged and statically isolated from `next_layer_prefetch`.

The smoke artifact does not serialize every layer's selected-ID list. Therefore
the exact per-ID equality is established by the static contract test and the
runtime hit/correction/waste lifecycle counters; this smoke does not claim a
separate serialized per-layer ID audit.

## Errors

- Runtime exit code: 0
- OOM: none
- stale-entry/lifecycle error: none observed
- acceleration claim: none

The first attempted launch was rejected before model loading because the
temporary 30-frame config lacked the required `smoke_10block: true` flag. It
was not a runtime smoke attempt and was corrected only in the temporary config.
