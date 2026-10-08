# Group11 non-FlashFetch algorithm consistency

Static-only audit; no inference or benchmark was run. Source snapshot: `410948eb7e328513e7751ac2d44193115363cee4`.

| Strategy | Retrieval query | Prediction | Final attention selection | Main extra work | Static semantic status |
|---|---|---|---|---|---|
| 11.1 CURRENT_Q | current Q_l | none | I_l directly + local/resident | provenance CPU ID conversion | REFERENCE: YES |
| 11.2 PREVIOUS_Q | stored Q from same layer's prior cache state; first use bootstraps current Q | none | I'_l directly + local/resident | detach/clone/persist query state; exact t-1 not proven | DIRECT QPREV after bootstrap, temporal alignment caveat |
| 11.4 NEXT_LAYER_PREFETCH | current Q_l for current layer and again current Q_(l+1) next layer | P_(l+1)=I_l | true I_(l+1), not prediction | CPU ID list/Python indexing, per-block pin/alloc/H2D, event wait, correction/waste handling | TRUE_ASYNC_PREFETCH structurally |

## Answers to final questions

1. **Does Group11.1 exactly implement current-Q DraftMap retrieval?** Yes for its enabled configuration; selected IDs are passed directly into the final working-set assembly.
2. **Is Group11.2 only replacing Q_t with Q_(t-1)?** Algorithmically it adds the intended previous-query state, but also adds clone/persistence/metadata overhead and a current-Q bootstrap; exact temporal predecessor alignment is not enforced.
3. **Does Group11.4 preserve current-Q retrieval for current and next layer?** Yes. It predicts from I_l, but recomputes I_(l+1) using Q_(l+1).
4. **Is prefetch launched early enough?** Structurally yes: after current retrieval and before current-layer history assembly/attention. The launch itself is preceded by host ID conversion.
5. **Separate CUDA stream?** Yes, a dedicated `torch.cuda.Stream`; compute uses the current stream.
6. **Direct hit consumption?** Yes, the exact prefetched K/V tensors are returned from the state entry.
7. **Can a hit trigger another H2D?** Not within the same retained state entry; after the entry is popped, later independent invocations have no durable cache and may fetch again.
8. **Correction fetch exactly once?** The visible selected-ID loop issues one correction for each missing `(layer, history_id)` key; top-k selection is unique.
9. **Waste postprocessing?** No RoPE/repack before discard, but stale predictions already paid pinning, allocation, and H2D.
10. **GPU→CPU sync/Python indexing?** Yes: selected IDs are converted with `.detach().cpu().tolist()` and used to index Python CPU lists.
11. **Physical K/V reused across layers?** No; only IDs cross layers, while target-layer archive supplies target-layer K/V.
12. **Structurally capable of beating CURRENT_Q?** Yes, conditionally: a separate prefetch stream and event wait create a plausible hideable interval.
13. **What may prevent a win?** CPU ID synchronization, Python per-block control, per-prefetch allocation/pinning, stale/wasted H2D, correction traffic, and no durable cache after hit consumption. These are structural cost sources, not proof of a measured slowdown.

## Important non-conclusions

This audit does not claim a timing win, does not claim exact qprev temporal alignment, and does not claim that all H2D is hidden. Those require a matched runtime benchmark with explicit exposed/hidden event accounting.
