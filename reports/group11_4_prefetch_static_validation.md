# Group11.4 next-layer prefetch static validation

This report records the source-level port only. No H200 inference was run.

## Contract

- The authoritative selection at layer `l` is `I_l` from current-Q DraftMap retrieval.
- The transient prefetch requests `(l+1, id)` for `id in I_l` from the next layer's CPU BF16 archive.
- At layer `l+1`, stale IDs are discarded before attention; only `I_{l+1}` is consumed.
- Hits are `I_l ∩ I_{l+1}`; correction fetches are `I_{l+1} - I_l`; waste is `I_l - I_{l+1}`.
- Prefetched IDs are never unioned into the final attention working set.

## Ownership and accounting

The current Group12–15 `compressed_history_archive` path was not changed. It remains the persistent low-bit owner and still dequantizes selected values for BF16 attention. The Group11.4 prefetch state is a separate transient dictionary attached to the shared cache owner. Its bytes are reported as transient prefetch memory and are not added to `GPU_KV_ACTUAL_PERSISTENT_BYTES` or the compression denominator.

## Static results

| Check | Result |
|---|---|
| Python compilation of attention/pipeline/inference/memory modules | PASS |
| Set-contract test for hit/correction/waste/final IDs | PASS |
| Group12–15 archive/low-bit owner path retained | PASS |
| Group11.4 config selects current-Q and `next_layer_prefetch` | PASS |
| H200 inference/runtime validation | NOT RUN BY REQUEST |

The implementation is safe to proceed to an isolated smoke test; it is not runtime-validated by this change.
