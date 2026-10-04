# Group11.2 previous-Q temporal prefetch design audit

## Scope

This audit covers the repaired `previous_q_prefetch` mode only. The existing
direct mode remains available as `previous_q_direct`; the legacy config value
`previous_q` is normalized to that mode.

## Temporal axis

- `invocation_id`: per-layer attention-call counter stored in
  `kv_cache["qprev_current_invocation_id"]`; provenance also records
  `current_start`, `layer`, and `current_denoising_step`.
- `valid_temporal_invocation_id`: per-layer counter advanced only when a
  history retrieval is eligible. This prevents non-eligible early calls from
  being mistaken for the next valid temporal target.
- `TEMPORAL_KEY`: `(valid_temporal_invocation_id, layer_id, history_id)` plus
  raw invocation and denoising metadata in the entry.
- `PREVIOUS_INVOCATION_RELATION`: the source Q is produced by the immediately
  preceding valid same-layer temporal invocation; raw attention-call gaps are
  retained as metadata, not silently treated as valid targets.
- `NEXT_TEMPORAL_TARGET_RELATION`: `Q_(t,l)` produces `P_(t+1,l)` for the next
  valid invocation at the same layer.

## Old versus new dataflow

Old `previous_q_direct`:

```text
invocation t: Q_(t-1,l) -> retrieve -> demand CPU->GPU fetch -> dense FA
```

New `previous_q_prefetch`:

```text
bootstrap: Q_(0,l) -> P_(1,l) -> prefetch
invocation t: consume P_(t,l) -> dense FA -> persist Q_(t,l)
              -> retrieve Q_(t,l) -> P_(t+1,l) -> async same-layer H2D
```

The prediction state carries the exact target valid invocation and layer. A
state entry with a mismatched target is rejected as stale.

## Static design decisions

| Requirement | Result |
|---|---|
| Old mode preserved | YES (`previous_q_direct`) |
| New mode | `previous_q_prefetch` + `temporal_qprev_prefetch` |
| Same-layer target | YES |
| Cross-layer physical K/V reuse | NO |
| Dedicated prefetch stream/event | YES |
| CPU archive source pre-pinned | Reuses existing archive contract |
| Reusable GPU staging | YES; consumer event guards reuse |
| Current-Q correction retrieval | NO |
| Stock dense FlashAttention | YES; unchanged |
| Persistent Q | YES, for constructing the next prediction |
| Normal fallback | DISABLED; debug-only fallback requires explicit env flag |

## Files changed

- `/data/zxl/LongLive-RAG-group11_15_h200/wan/modules/causal_model_latentmem.py`
- `/data/zxl/LongLive-RAG-group11_15_h200/pipeline/causal_inference.py`
- `/data/zxl/LongLive-RAG-group11_15_h200/configs/g11_2_prefetch.yaml`

Static compilation passed with the existing H200 Python environment. No
canonical10 inference or evaluation was run.
