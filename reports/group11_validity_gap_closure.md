# Group11 implementation-validity gap closure

This was a static/code-correctness update plus minimal 30-frame/10-block smoke only.
No canonical10 inference and no evaluation pipeline was run.

## Required outputs

| Check | Result |
|---|---|
| exact qprev temporal alignment | **PASS** |
| host ID sync removed | **NO** |
| per-prefetch GPU allocation removed | **YES for reused staging slots; first-seen slots allocate once** |
| prefetch source pre-pinned | **YES** |
| hit tensor reuse unchanged | **YES** |
| final working set unchanged | **YES** |

## Group11.2

The runtime now records `invocation_id`, `layer`, `current_start`, and denoising-step metadata for the stored query. The exact predecessor check is `previous_invocation_id == current_invocation_id - 1` plus same-layer validation. Bootstrap remains explicit and is not mislabeled as an exact previous query.

Smoke result: `600` attention calls; status counts `{'BOOTSTRAP_CURRENT_Q': 30, 'PASS': 300}`. There were no alignment failures. The retrieval tensor and semantics were not changed; the new check is trace/assertion-only (`QPREV_ALIGNMENT_ASSERT=1` was used for smoke).

## Group11.4

Changes are limited to storage/control overhead:

- CPU history frames are pinned when they enter the archive, not inside prefetch.
- Prefetch uses reusable staging tensors keyed by `(target_layer, history_id)` and always overwrites them before marking a new event. This is not a durable valid-data cache: the state entry is still popped on consumption, and a future prediction copies into the slot again.
- Exact predictor IDs, `(layer, history_id)` keys, event waits, correction path, waste discard, and final true-current-Q working set are unchanged.

The GPU→CPU ID conversion remains. Removing it safely would require replacing Python-list archive indexing with a GPU-indexable packed archive or another changed storage contract, which is outside this task.

Smoke result: the authoritative 30-frame/10-block run recorded `600` attention
calls; requested `638`, hits `351`, corrections `660`, and waste `287`.

### Corrected authoritative smoke summary

| Field | Value |
|---|---|
| `PREFETCH_REQUESTED` | `638` |
| `PREFETCH_HITS` | `351` |
| `PREFETCH_CORRECTIONS` | `660` |
| `PREFETCH_WASTE` | `287` |
| `ATTENTION_CALLS` | `600` |
| `CONFIG_PATH` | `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_validity_smoke_10block/group11_4_prefetch/config.yaml` |
| `RESULT_DIR` | `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_validity_smoke_10block/group11_4_prefetch` |
| `GIT_COMMIT` | `410948eb7e328513e7751ac2d44193115363cee4` |
| `FINAL_WORKING_SET_UNCHANGED` | `YES` |

The `0/0/0/0` values were not the authoritative prefetch statistics. The
report generator read lowercase nested keys such as
`group11_profile.prefetch_requested_chunks`, while the runtime artifact stores
the counters as uppercase top-level keys: `PREFETCH_REQUESTED_CHUNKS`,
`PREFETCH_HIT_CHUNKS`, `PREFETCH_CORRECTION_CHUNKS`, and
`PREFETCH_WASTED_CHUNKS`. Therefore this was a field-name parsing error, not a
different result selected by the report generator. The separate short
`group11_validity_smoke` run genuinely has zero prefetch events because it is a
3-frame/1-block bootstrap smoke without history eviction; it is not the
authoritative prefetch smoke.

The 10-block configuration explicitly enabled
`group11_fetch_mode: next_layer_prefetch`, so prefetch was exercised. The
successful smoke plus the unchanged true-current-Q selection path validates
the final-working-set gate; no independent quality evaluation was run.

## Evaluation policy

No quality evaluation was started. Any later evaluation must use the frozen canonical10 evaluation wrapper and the same 10-case protocol as canonical inference; these smoke outputs are correctness artifacts only.
