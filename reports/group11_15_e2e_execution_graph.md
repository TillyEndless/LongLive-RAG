# Group11–15 E2E Execution Graph Audit

## Scope and evidence

Trace-only audit for Group11.1, Group12, and Group14.1. No inference was run,
no retrieval/sparse/quantization code was changed, and no final profiler was
added. Evidence is the current source plus existing case01 runtime traces:

- Group12 runtime: `/data/zxl/strict_latency_case01_20260928/aligned_runtime_v4/group12_gpu0/case_01/rank0-0-0_lora_runtime.json`
- Group14.1 runtime: `/data/zxl/LongLive-RAG-group11_15_h200/results/group14_15_corrected_sparse/group14_sparsity30/case01/rank0-0-0_lora_runtime.json`

## 1. Full E2E call graph

`inference.py:257-523` loop → `CausalInferencePipeline.inference()`
(`pipeline/causal_inference.py:180-461`) → text encoder/init → 40 generation
units (`:276-405`) → 4 denoising calls plus one context-timestep cache update
per unit (`:326-394`) → generator/transformer blocks → self-attention,
cross-attention, FFN/residual → latent output → VAE decode (`:432-439`) →
return video → `inference.py:344-374` GPU-to-CPU conversion and `write_video`.

`PROCESS_WALL` includes model setup, pipeline inference, CPU conversion and
video writing. `E2E_INFERENCE` is `e2e_start` to `e2e_end`. `TRANSFORMER_TOTAL`
is the `transformer_start` to `transformer_end` interval and includes all
generation units, denoising, routing, H2D, attention and cache work.

## 2. Representative late-stage dataflow

For each generation unit, the pipeline sets the denoising timestep on all
self-attention blocks, builds the timestep tensor, calls the generator, adds
noise for the next timestep, then performs the final denoising call. It then
stores denoised latent descriptors and calls the generator once more at the
context timestep to update the KV cache.

At a transformer block, the code performs Q/K/V projection, cache roll/direct
insert and CPU offload of evicted BF16 K/V, causal RoPE, DraftMap retrieval
and selected history fetch, concatenates sink/history/local K/V, invokes
`prepare_attention_kv`, calls `attention(...)`, applies `self.o`, then returns
to the block's residual/cross-attention/FFN path.

## 3. Self-attention detailed trace

The detailed source sequence is:

1. Q/K/V projections in `wan/modules/causal_model_latentmem.py`.
2. Local cache update/eviction and causal online RoPE (`:570-740`).
3. Historical fetch loop (`:773-905`): archive lookup, selected CPU K/V
   materialization or `.to(device, non_blocking=True)`, then stack/view and
   RoPE for fetched K.
4. Sink + historical + local tensors are concatenated (`:910-928`).
5. `prepare_attention_kv` (`:936-942`) applies persistent-owner handling and,
   for Group14.1, DraftMap routing.
6. Attention call (`:944-950`) invokes the unchanged `attention()` function.
7. Output projection (`:969-977`).

The existing trace records 6000 attention rows. Group12 Q length is 4680 and
observed K/V lengths are 4680, 9360, 10920, 15600, 18720. Group14.1 Q length
is 4680 and actual dense-kernel input K/V lengths are 3328, 6592, 7680,
10944, 13184. The existing rows therefore show a smaller dense input, not a
true sparse kernel.

## 4. Group12 low-bit KV lifecycle

The intended corrected lifecycle is CPU BF16 historical K/V → selected fetch /
H2D → persistent GPU low-bit archive owner → temporary BF16 materialization →
BF16 attention. `inference.py:400-407` records Group12's persistent owner as
INT8 K / FP8 E4M3 V and final attention as BF16. The attention call itself is
made at `causal_model_latentmem.py:936-950`.

The current artifacts confirm the storage contract metadata, but do not expose
a separate per-chunk quant-pack/dequant CUDA event boundary. The temporary
BF16 lifetime is therefore: created for the current attention operands,
consumed by the attention call, and not intended to remain as a persistent
full-history BF16 shadow. Exact allocator lifetime is `NOT_AVAILABLE` from the
stored trace.

## 5. Group14.1 sparse lifecycle

The config uses `group_sparse_ratio: 0.70` and the contract comment says this
means retain 70% / prune 30%. `persistent_draftmap.py:27-43` pools Q/K in
64-token blocks, computes pooled QK scores, softmaxes and averages them,
retains `ceil(0.70*K)` blocks, constructs token IDs, then `torch.gather`s K/V.
The result is passed to the same dense attention function.

This is **GATHER_THEN_DENSE**, not a sparse CUDA attention kernel and not a
dense masked kernel. Existing trace QK element totals are:

- Group12: `5,766,463,872,000`
- Group14.1: `4,060,988,006,400`
- ratio: `0.7042` (approximately 29.6% fewer QK elements)

## 6. CUDA stream and overlap graph

For Group11.1/12/14.1 there is no explicit copy stream in the active path.
The H2D rows identify `copy_stream: default`; target rows report
`source_pinned_memory: false`. The prefetch helper creates a separate stream
in `causal_model_latentmem.py:286-321`, but it is not active for these target
groups (`group11_fetch_mode=serial_full`, and target configs disable prefetch).

Thus the verified dependency is routing → H2D/decode → gather/repack → dense
attention → output projection, all on the default stream. H2D/compute overlap
is not established by the current implementation. No D2H/compute overlap is
used in the target attention call.

## 7. Synchronization inventory

- `pipeline/causal_inference.py:273`: explicit `torch.cuda.synchronize()` at
  initialization/profile boundary.
- `:398`: explicit synchronization after each generation unit when profile is
  enabled.
- `:410`: synchronization at diffusion end.
- `:420` and `:437`: synchronization around VAE timing.
- `.cpu()`, `.numpy()`, `.item()` calls occur in routing, metadata logging,
  scheduler/cache indexing and final video conversion; these can create
  implicit host waits when the value is GPU-produced.
- `non_blocking=True` appears on several transfers, but non-blocking alone
  does not establish overlap without pinned memory and an independent stream.

## 8. Attention backend

`wan/modules/attention.py:139-185` selects FlashAttention when the import is
available; otherwise it uses `torch.nn.functional.scaled_dot_product_attention`.
The H200 environment has `flash_attn 2.8.3`. The stored runtime metadata does
not record the exact backend dispatch per call, so the definitive backend for
the historical videos is `FlashAttention if dispatch predicate was true,
otherwise SDPA`; kernel-level CUDA timing is `NOT_AVAILABLE`.

## 9. Operation taxonomy and timer nesting

The complete inventory is in `results/group11_15_operation_inventory.csv`.
The nesting tree is in `results/group11_15_timer_nesting_tree.json`.
Parent timers contain children; future aggregation must not add
`SELF_ATTN_WRAPPER` to its child timers. A future profiler should report both
work time and exposed critical-path time because H2D/routing may be nested or
overlapped.

## 10. Current evidence limits

Existing runtime rows use CPU wall timers for the attention wrapper and phase
bookkeeping. They do not contain synchronized CUDA-event timings for pure
QK, softmax, PV, quant/dequant, or gather kernels. Therefore this audit does
not claim a kernel speedup or slowdown and does not finalize the unified
profiler timers.

## Required fields

| Field | Value |
|---|---|
| E2E_ENTRY | `inference.py:257-321` → `CausalInferencePipeline.inference` |
| E2E_EXIT | `pipeline/causal_inference.py:449-461`; then `inference.py:344-374` |
| GENERATION_LOOP | `pipeline/causal_inference.py:276-405` |
| DENOISING_STEP | `pipeline/causal_inference.py:326-367`, plus context update `:381-394` |
| TRANSFORMER_BLOCK | generator model block forward; self-attention source `wan/modules/causal_model_latentmem.py` |
| SELF_ATTN_ENTRY | `wan/modules/causal_model_latentmem.py:505`-projection path / attention wrapper |
| SELF_ATTN_EXIT | `wan/modules/causal_model_latentmem.py:969-977` |
| GROUP12_ATTENTION_BACKEND | FlashAttention if available, otherwise SDPA; exact historical dispatch NOT_AVAILABLE |
| GROUP14_1_ATTENTION_BACKEND | Same dense attention API after gather; exact historical dispatch NOT_AVAILABLE |
| GROUP12_Q_LEN | 4680 |
| GROUP12_K_LEN | 4680/9360/10920/15600/18720 by step |
| GROUP14_1_Q_LEN | 4680 |
| GROUP14_1_ORIGINAL_K_LEN | corresponding Group12 K lengths |
| GROUP14_1_ACTUAL_KERNEL_K_LEN | 3328/6592/7680/10944/13184 |
| GROUP14_1_SPARSE_IMPLEMENTATION | GATHER_THEN_DENSE |
| GROUP14_1_RETAINED_RATIO | 0.70 target; 0.7042 QK-weighted observed |
| GROUP14_1_PRUNED_RATIO | 0.30 target; approximately 0.2958 QK-weighted observed |
| PERSISTENT_LOWBIT_OWNER_GROUP12 | GPU INT8 K / FP8 E4M3 V archive |
| TEMP_BF16_LIFETIME | attention operand materialization; exact allocator lifetime NOT_AVAILABLE |
| H2D_STREAM | default stream |
| COMPUTE_STREAM | default stream |
| D2H_STREAM | no dedicated D2H stream; final `.cpu()` implicit/default path |
| H2D_COMPUTE_OVERLAP_POSSIBLE | not established; no active independent copy stream and source rows unpinned |
| D2H_COMPUTE_OVERLAP_POSSIBLE | not used/NOT_AVAILABLE |
| MAJOR_SYNC_POINTS | `pipeline/causal_inference.py:273,398,410,420,437`; plus implicit `.cpu()/.item()/.numpy()` waits |
| OPERATION_COUNT_BY_CATEGORY | see CSV; 21 inventoried categories |
| UNCLASSIFIED_OPERATIONS | exact cross-attention/norm-residual source boundaries and exact quant/dequant kernel boundaries are NOT_AVAILABLE in current trace |
| TRACE_REPORT | `/data/zxl/LongLive-RAG-group11_15_h200/reports/group11_15_e2e_execution_graph.md` |
| OPERATION_CSV | `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_15_operation_inventory.csv` |
| NESTING_JSON | `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_15_timer_nesting_tree.json` |
| STREAM_GRAPH_JSON | `/data/zxl/LongLive-RAG-group11_15_h200/results/group11_15_stream_dependency_graph.json` |
