# H200 Group12–15 campaign audit

## Audit status

The campaign was paused after inspecting the first formal Group12 case and the active runner. No unrelated process was terminated. Existing videos and logs were preserved but are not promoted to final results.

| Item | Finding | Status |
|---|---|---|
| Window | 12 | PASS |
| Retrieval | `draftmap_online`, current layer Q path | PASS by source trace |
| CPU historical K/V | Source archive stores BF16 tensors | PASS semantically; measurement labels were stale |
| GPU low-bit persistence | Current pre-fix code quantized and immediately dequantized, then retained BF16; dequantization itself is valid, but persistent BF16 ownership is not | INVALID; target is persistent low-bit owner plus bounded BF16 materialization |
| Final attention | BF16 attention | PASS |
| Memory accounting | CPU archive bytes omitted from `CPU_KV_MEASURED_BYTES`; GPU KV reports BF16 local cache | INVALID / must fix |
| Scheduler threshold | Active runner used 30 GiB | INVALID against 35 GiB normal-mode policy |
| Evaluation | Batch evaluator not yet reached; no final row is valid | PENDING |

## Evidence

The active Group12 case metadata reported `CPU_HISTORY_STORAGE_DTYPE=bf16`, `H2D_SOURCE_DTYPE=bf16`, and `FINAL_ATTENTION_DTYPE=bfloat16`, but also reported `CPU_KV_MEASURED_BYTES=0`, `CPU_KV_TOTAL_BYTES=132843110400`, and `GPU_LOCAL_BF16_KV_BYTES=3450470400`. The runtime path in `wan/modules/causal_model_latentmem.py` calls `prepare_attention_kv()` on the concatenated BF16 operands; `utils/h200_group_runtime.py` quantizes and immediately dequantizes them. This is numerical fake quantization, not persistent GPU low-bit KV storage.

The source-level archive in `utils/compressed_history_archive.py` is correctly BF16-authoritative. The accounting producer must therefore count archive payloads as CPU KV, while not calling them compressed CPU storage. Dequantization is expected and valid; only immediate dequantization followed by persistent BF16 ownership is invalid.

## Required corrective actions before resume

1. Keep CPU historical K/V as authoritative BF16 and count those payloads in `CPU_KV_MEASURED_BYTES`.
2. Wire a real persistent GPU low-bit working-set owner for Groups12–15; temporary BF16 dequant buffers are valid and expected, but must not become the long-lived owner.
3. Report `persistent_packed_k_bytes`, `persistent_packed_v_bytes`, `persistent_scale_bytes`, `persistent_metadata_bytes`, `transient_bf16_dequant_peak_bytes`, `bf16_equivalent_gpu_kv_bytes`, `gpu_kv_actual_persistent_bytes`, and the GPU-only compression ratio separately.
4. Use normal-mode free-VRAM threshold 35 GiB; use drain-mode threshold 30 GiB only after external processes disappear.
5. Requeue the preserved Group12 case01 output and every case generated under the transient/incorrect accounting contract. Do not reuse them as final rows.
6. Add case-level evaluation state; a row is final only after 10/10 valid inference and 10/10 canonical evaluations.

## Current campaign state at pause

- Group12/case01: video and runtime artifacts preserved, marked `INVALID_SEMANTICS`.
- Group12/case02: interrupted before completion; requeue.
- Group13–15: no valid canonical case promoted.
- Smoke outputs remain separate and are not included in final means.
- Unrelated H3 processes were not modified.

## Final conclusion

The old campaign was not scientifically complete. The retrieval algorithm and CPU BF16 archive source semantics are aligned, but its GPU storage contract and memory accounting were not. It remains stopped and excluded.

## Corrected static gate

The corrected source now implements the intended lifecycle: CPU BF16 archive, BF16 H2D on first selected fetch, persistent GPU INT8/FP8 or NVFP4 packed owner, and temporary BF16 materialization for attention. `prepare_attention_kv()` skips a second quantize/dequantize cycle when the archive has already materialized the persistent owner. The runtime memory producer separates packed K/V, scale metadata, transient BF16 dequant peak, BF16-equivalent bytes, and actual persistent GPU bytes.

Static validation completed:

- 100 corrected configs generated under `results/group12_15_persistent_campaign/`.
- 10 cases for Group12/13 and 10 cases for each Group14/15 retained ratio.
- All configs: window 12, `draftmap_online`, memory size 6, recent exclusion 5, 474 frames, seed 0.
- Manifest SHA256: `db9462c0954186b9128543b58fb6d1d496a8b89e0754333beca698d3cf461fec`.
- Python compilation passed for archive, runtime, model, pipeline, inference, and accounting modules.
- Forbidden q-prev/prefetch/Flash Fetch/hot-cache paths were not found in the corrected execution path.

The persistent-owner smoke was paused intentionally after Group12/13 generation while reviewing static semantics. It must be rerun after the final evaluator and scheduler audit; no canonical case has been promoted.
