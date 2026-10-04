# Group12 vs Group14.1 Attention Latency Probe

## Scope

This audit compares the same `case01`, W12, seed 0, 120 latent output-frame
configuration and the same Wan2.1-T2V-1.3B/LongLive checkpoints. Group14.1 is
the corrected `group14_sparsity30` configuration: `group_sparse_ratio=0.70`,
which means 70% of block interactions are retained and 30% are skipped.
Groups 14.2/14.3/14.4 were not run.

## Source-level execution path

1. `wan/modules/causal_model_latentmem.py:936-939` calls
   `prepare_attention_kv(...)`.
2. `utils/h200_group_runtime.py:154-159` applies Group14 routing through
   `utils/persistent_draftmap.py:27-43`.
3. The router mean-pools Q/K in 64-token blocks, computes softmax scores,
   stable-sorts block IDs, and gathers selected K/V blocks.
4. `wan/modules/causal_model_latentmem.py:944-950` then calls the unchanged
   `attention(roped_query, k_cat, v_cat)`.
5. `wan/modules/attention.py:154-169` selects FlashAttention when installed;
   otherwise `:181-182` uses PyTorch SDPA. The H200 environment has
   `flash_attn 2.8.3`; the runtime JSON does not record the exact selected
   backend per call.

Therefore Group14.1 is **B**: it gathers a smaller dense K/V tensor and then
calls dense attention. It is not a block-sparse attention kernel and it does
not apply only a sparse mask to a full K/V sequence.

## Measured trace summary

| Variant | Attention calls | Q tokens | K/V tokens observed | QK elements | QK ratio | Attention wrapper phase | DraftMap decision | Fetch phase |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| Group12 | 6000 | 4680 | 4680;9360;10920;15600;18720 | 5766463872000 | 1.000 | 75.325s | 6.649s | 11.207s |
| Group14.1 | 6000 | 4680 | 3328;6592;7680;10944;13184 | 4060988006400 | 0.704 | 120.676s | 8.350s | 18.179s |

The observed QK element count is reduced to approximately
`0.704` of Group12, i.e. about
`29.6%` fewer QK elements. The
runtime block shapes show Group12 K/V lengths of `4680;9360;10920;15600;18720`
and Group14.1 lengths of `3328;6592;7680;10944;13184` because historical
block rounding varies by step.

## Why there is no end-to-end speedup

The existing Group14.1 trace does show less dense-attention work, but the
measured attention wrapper is *slower* (`120.676s`
vs `75.325s`). This wrapper timing is a CPU
wall-clock phase around the attention call, not a CUDA-event-only kernel time.
It therefore cannot prove the pure FlashAttention kernel time or a precise
kernel speedup.

The source and trace do establish the overhead mechanism:

- Group14 performs DraftMap pooling, score computation, softmax, sorting/top-k,
  token-ID construction, and K/V gather before the unchanged dense attention.
- The selected tensor is repacked into a contiguous dense tensor through
  `torch.gather` and then passed to the same dense attention API.
- The saved QK elements do not imply proportional wall-time savings because
  the kernel is launched on a dynamically gathered, irregular-length tensor;
  launch/setup and memory movement can dominate.
- The existing `attention_rows` and `model_phase_ms` fields are CPU wall times;
  no per-operation CUDA events or kernel names were saved. Consequently the
  exact split between pure kernel saving and sparse-only overhead is
  **NOT_AVAILABLE** from existing artifacts.

## Critical status

`SPARSITY_DOES_NOT_REDUCE_ATTENTION_COMPUTE = NO`: QK element count does fall.

`NATIVE_SPARSE_ATTENTION_KERNEL = NO`: the implementation gathers selected
blocks and invokes dense FlashAttention/SDPA.

`PURE_CUDA_KERNEL_SPEEDUP = NOT_AVAILABLE`: a new synchronized CUDA-event
trace around the attention backend is required to measure it correctly.

`E2E_GROUP14.1 = NOT_AVAILABLE in the stored runtime JSON`: the Group14.1
artifact predates the unified E2E runtime fields. It must not be reconstructed
from phase sums and presented as a wall-time measurement.

The H2D rows in this historical Group14.1 artifact also contain zero per-row
copy times; they are not used to claim a H2D speedup.

## Conclusion

The current evidence supports: sparse routing reduces the dense K/V sequence
and QK element count, but it is implemented as routing plus gather/repack plus
ordinary dense attention. The added routing/materialization and irregular
kernel-launch costs can cancel the arithmetic reduction. A definitive numeric
`kernel_saved` and `sparse_overhead` requires a new same-case CUDA-event probe;
those two values are not fabricated here.

Measured latency equation for Group12:

`Group12 attention = KV decode/materialize + dense attention kernel + other`

The existing trace provides the wrapper phase, but does not isolate the pure
CUDA kernel term.
