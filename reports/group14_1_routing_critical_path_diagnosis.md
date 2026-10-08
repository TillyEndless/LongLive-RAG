# Group14.1 Sparse-Selection Critical Path Diagnosis

## Scope

Matched `case_01`, 6,000 attention calls, Group12 baseline, Group14.1 baseline, and a separate Group14.1 `GROUP14_GPU_ONLY_ROUTING=1` run. No canonical10 and no Group14.2–14.4 were run. Final attention remains BF16 and selected IDs/semantics are unchanged.

## Phase A: routing decomposition

Host/exposed seconds are CPU-observed phase wall times. CUDA seconds are CUDA-event work times; they are reported separately and are not summed into E2E.

| field | host/exposed s | CUDA work s |
|---|---:|---:|
| ROUTING_TOTAL_S | 6.660410 | 5.589255 |
| Q_POOL_S | 0.392085 | 0.471410 |
| K_POOL_S | 0.512590 | 1.562285 |
| SCORE_MATMUL_S | 1.259805 | 0.124286 |
| SOFTMAX_S | 0.327593 | 0.120925 |
| IMPORTANCE_REDUCTION_S | 0.179143 | 0.075623 |
| TOPK_S | 0.512699 | 0.295285 |
| ID_CONSTRUCTION_S | 0.859514 | 0.575100 |
| ROUTING_OTHER_S | 0.470105 | 0.733651 |

`ROUTING_OTHER_S` is the parent-minus-independent-child residual; it is not silently assigned to a named component. Token clamp, metadata construction, provenance conversion, and parent bookkeeping are retained in the closure.

## Phase B: ID consumption audit

The sparse route keeps `ids` on CUDA through sorted selected-block IDs, token-ID expansion, and `torch.gather(k/v)`. The correctness gate passed for exact selected IDs/order, token IDs, K/V tensors, retained fraction, and final BF16 attention.

However, the historical retrieval path is separate: `_online_memory_indices` and fetch tracing call `.detach().cpu().tolist()`, then the attention path executes `compressed_entries[int(k_idx)]`. Because `compressed_entries` is a Python list of archive records, full historical GPU-only selection is not safe without changing the storage/indexing contract. The tested flag only removes sparse-route provenance conversion; it does not pretend to eliminate archive-index CPU work.

## Baseline vs GPU-only provenance path

| metric | baseline | GPU-only provenance | delta |
|---|---:|---:|---:|
| CPU_SYNC_BASELINE_S | 1.100395 | 0.000000 | -1.100395 |
| GATHER_BASELINE_S | 0.517017 | 0.324335 | -0.192682 |
| ATTENTION_KERNEL_BASELINE_S | 8.528731 | 8.149476 | -0.379255 |
| WRAPPER_BASELINE_S | 79.582254 | 78.435309 | -1.146945 |

* `NET_WRAPPER_SAVING_S = 1.146945`
* `NET_WRAPPER_SPEEDUP = 1.014623x`

The measured wrapper improvement is positive, so this is a real measured improvement for the limited provenance-only path. It is not a claim of full GPU-only historical retrieval.

## Required fields and decisions

* `SPARSE_OVERHEAD_BUDGET_S = 2.061651`
* `CURRENT_POSITIVE_OVERHEAD_S = 6.373226`
* `OVERHEAD_REDUCTION_REQUIRED_FOR_BREAK_EVEN_S = 4.311575`
* `GPU_ONLY_SELECTION_FEASIBLE = YES` for sparse route IDs through token construction/gather; `NO` for full archive retrieval without a storage/index redesign.
* `CPU_SYNC_REMOVAL_EFFECTIVE = YES` for provenance conversion; `NO` for archive-list indexing.
* `ROUTING_DOMINANT_SUBCOMPONENT = parent routing residual plus score/pooling/selection aggregate; parent host time is 6.660410 s.
* `BREAK_EVEN_REACHED = NO`: the measured wrapper saving is below the 4.311575 s required to reach the 2.061651 s kernel-saved budget.
* `NEXT_OPTIMIZATION = vectorize/batch DraftMap scoring and replace Python archive-list indexing with a device-addressable archive index; selective decode remains lower priority because prior full-decode delta was negative.`

No inference semantics, selected IDs, sparse ratio, or final BF16 attention semantics were changed.
