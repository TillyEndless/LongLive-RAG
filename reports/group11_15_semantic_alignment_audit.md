# Group11–15 v2 semantic alignment audit

审计对象：`/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2`

本次只做 source/config 静态审计和 Python 静态编译检查；没有运行 smoke、canonical inference、VBench、DINO 或其他 evaluation。

## 结论

| 项目 | 结论 |
|---|---|
| Group11.1 current-Q DraftMap retrieval | PASS |
| Group11.2 previous-Q prefetch | PASS |
| Group11.3 online chunk/flash fetch | PASS（独立的在线 BF16 streaming attention 路径） |
| Group11.4 next-layer prefetch | PASS |
| Group12 local INT8-K/FP8-V owner | PASS |
| Group13 local NVFP4 K/V owner | PASS |
| Group14 Group12 + Q-side sparse | PASS |
| Group15 Group13 + Q-side sparse | PASS |
| CPU authoritative BF16 history K/V | PASS |
| GPU historical persistent KV = 0 for v2 | PASS |
| Group14/15 sparse-before-promotion ordering | PASS |
| Group14/15 exact full-candidate routing | PASS |

## Group11 execution mapping

### Group11.1

Representative configs:

- `/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/configs/group11_15_reference/group11_draftmap_online_w12_10case.yaml`
- `/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/configs/final_group11_gpu1.yaml`

These use `retrieval_backend: draftmap_online`, `retrieval_query_mode: current_q` by default, and `group11_fetch_mode: serial_full` by default. The source path is:

`WanAttentionBlock.forward()` → `_online_memory_indices()` → persistent GPU Draft-K scoring/top-k → CPU `cpu_k_frames/cpu_v_frames` fetch → BF16 candidate assembly → attention.

Evidence:

- `wan/modules/causal_model_latentmem.py:443-525`: DraftMap candidate construction, GPU Draft-K score, top-k and selected history IDs.
- `wan/modules/causal_model_latentmem.py:1089-1227`: selected CPU history K/V are copied and assembled before attention.
- `wan/modules/causal_model_latentmem.py:1343-1373`: candidate tensors are concatenated and passed to the unchanged attention call.

The implementation does not keep full historical K/V persistently on GPU. `gpu_draft_k_frames` is the persistent Draft-K metadata used for retrieval; the full historical K/V source is CPU BF16.

### Group11.2

Config:

`/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/configs/qprev_case01.yaml`

It sets `retrieval_query_mode: previous_q`. The source path queues the current Q after projection, uses the previous Q for the next retrieval decision, and schedules same-layer prefetch:

- `wan/modules/causal_model_latentmem.py:599-615`: Q history queue/commit.
- `wan/modules/causal_model_latentmem.py:452-472`: previous-Q selection and bootstrap behavior.
- `wan/modules/causal_model_latentmem.py:761-765`: same-layer prefetch scheduling.
- `wan/modules/causal_model_latentmem.py:588-597`: event-based consumption; a miss falls back to a correctness fetch.

Thus Group11.2 changes the retrieval timing/query source, not the retrieved representation or attention semantics.

### Group11.3

Config:

`/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/configs/flashfetch_case01.yaml`

It sets `group11_fetch_mode: flashfetch_serial`. The path is:

`forward()` → `_flashfetch_online_attention()` → pinned CPU BF16 K/V → async H2D for the next chunk → event wait → online softmax merge.

Evidence: `wan/modules/causal_model_latentmem.py:616-708`.

This is a chunk-pipelined streaming attention implementation, not the ordinary dense `attention()` call. It preserves the full BF16 attention result through online softmax accumulation, while changing fetch scheduling. It does not use the v2 low-bit local store because Group11 is the BF16 baseline.

### Group11.4

Config:

`/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/configs/group11_4_next_layer_prefetch_case01.yaml`

It sets `group11_fetch_mode: next_layer_prefetch` and `retrieval_query_mode: current_q`. The path schedules `(layer+1, selected history IDs)` on a CUDA stream, reconciles predictions against the next layer's exact IDs, and consumes only exact hits; misses are corrected by a normal fetch.

Evidence:

- `wan/modules/causal_model_latentmem.py:551-586`: next-layer CPU BF16 K/V prefetch.
- `wan/modules/causal_model_latentmem.py:588-597`: exact-hit event consumption and fallback behavior.
- `wan/modules/causal_model_latentmem.py:766-772`: schedule/reconcile integration.

## Group11 local promotion boundary

All Group11 configs expose `local_kv_promotion_ratio: 0.2`, and the model computes/records local promotion IDs through:

`wan/modules/causal_model_latentmem.py:386-405` and `wan/modules/causal_model_latentmem.py:2031-2055`.

However, Group11 is the uncompressed BF16 baseline: `_initialize_kv_cache()` allocates dense BF16 `cache["k"]`/`cache["v"]` when `local_lowbit_mode is None` (`pipeline/causal_inference.py:541-549`). Therefore Group11 promotion is an identity operation: the local owner is already BF16, so no low-bit owner is replaced and no separate promotion H2D is needed. This is consistent with “Group12 = Group11.1 + local-window compression.”

If “promotion” were intended to mean an actual precision conversion in Group11, that would contradict the stated distinction that Group12/13 add local-window compression. The current implementation keeps the policy hook and provenance IDs while correctly making Group11 promotion numerically a no-op.

## Group12 and Group13

Active configs:

- `/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/configs/group12_15_v2/group12_v2_local_int8_fp8.yaml`
- `/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/configs/group12_15_v2/group13_v2_local_nvfp4.yaml`

Both use `retrieval_backend: draftmap_online`, CPU BF16 history, `local_kv_promotion_ratio: 0.2`, `q_sparse_ratio: 0.0`, and `attention_compute: bf16`.

The only intended precision difference is:

- Group12: persistent local K INT8, local V FP8 E4M3.
- Group13: persistent local K/V NVFP4.

Evidence:

- `pipeline/causal_inference.py:541-550`: v2 creates `LocalLowbitKVStore` instead of dense GPU BF16 local K/V.
- `utils/local_lowbit_kv.py:46-84`: new projected K/V are quantized at insertion and stored in low-bit slots.
- `utils/local_lowbit_kv.py:136-194`: attention receives call-local BF16 materialization; promotion is transient.
- `utils/local_lowbit_kv.py:196-210`: persistent-byte accounting counts packed owner storage, not the temporary BF16 tensor.
- `utils/h200_group_runtime.py:18-54`: BF16 input, low-bit storage transform, BF16 compute metadata; no native low-bit attention kernel.

Historical K/V are not stored as persistent GPU low-bit KV in this v2 contract. They are appended to `CompressedHistoryArchive` as CPU BF16 (`utils/compressed_history_archive.py:61-89`). `gpu_persistent_bytes()` is explicitly zero (`utils/compressed_history_archive.py:40-43`).

## Group14 and Group15

Active configs:

- `/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/configs/group12_15_v2/group14_v2_local_int8_fp8_sparse.yaml`
- `/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/configs/group12_15_v2/group15_v2_local_nvfp4_sparse.yaml`

The two configs are identical in routing/promotion settings (`q_sparse_ratio`, `local_kv_promotion_ratio`, DraftMap backend); they differ only in local low-bit storage mode:

- Group14: INT8 K / FP8 E4M3 V.
- Group15: NVFP4 K / NVFP4 V.

The exact current path is:

`current Q` → CPU-history fetch plan + local low-bit materialization → full candidate set (history + local) → `route_candidate_chunks()` → retained IDs → promotion selection inside retained local IDs → temporary CPU BF16 overlays → ordered BF16 candidate tensors → unchanged attention.

Evidence:

- `wan/modules/causal_model_latentmem.py:211-327`: v2 exact full-candidate unified path.
- `utils/persistent_draftmap.py:142-210`: exact full-candidate chunk routing; descriptor-only routing is not used by this path.
- `wan/modules/causal_model_latentmem.py:1290-1336`: serial fallback preserves sparse-then-promotion ordering.
- `wan/modules/causal_model_latentmem.py:1349-1373`: final tensors are BF16 and use the stock attention call.

The same-state replay already recorded in `/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2/results/group14_15_same_state_replay_ab.json` passed exact history/retained/promotion/final-ID equality and attention allclose for both Group14 and Group15. That replay is a correctness artifact, not a new inference/evaluation run.

## Ownership and memory contract

`utils/runtime_memory_measurement.py:176-228` deduplicates storage identities and separately accounts for:

- GPU local low-bit packed owner bytes;
- CPU BF16 historical archive and CPU local promotion sources;
- GPU Draft-K persistent bytes;
- transient BF16 materialization/fetch bytes.

For v2, `GPU_HISTORICAL_PERSISTENT_RESIDENT_KV_BYTES = 0` and `GPU_KV_ACTUAL_PERSISTENT_BYTES` is the local low-bit store owner (`utils/runtime_memory_measurement.py:261-268`). This matches the requested architecture: no persistent full-history BF16 GPU shadow.

## Static validation

The following passed without model execution:

```text
python -m py_compile wan/modules/causal_model_latentmem.py utils/same_state_replay.py
```

No source change was required by this audit. The only semantic caveat is the intentional identity-promotion behavior of the uncompressed Group11 BF16 baseline; Groups12–15 implement the actual low-bit-owner + transient BF16 promotion/dequantization contract.

## Final status

`GROUP11_15_SEMANTIC_ALIGNMENT = PASS`

`GROUP14_15_EXACT_ROUTING = PASS`

`GROUP12_15_LOWBIT_OWNER = PASS`

`GPU_HISTORICAL_PERSISTENT_KV_V2 = 0`

`INFERENCE_RUN = NO`

`QUALITY_EVALUATION = NO`
