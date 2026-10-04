# Phase BA-1.3 — Real AO/BA Attention Parity

## Scope

This validation used isolated copies only. The formal AO and BA worktrees were not modified. Both runs used the same case prompt, checkpoint, LoRA, seed, `serial_full` historical fetch, layer-0 capture, and F30/smoke-10-block workload. No F120 run or Unified Evaluation was executed.

AO: original W20/R16/S1, physical cache 20.

BA: logical W20/R16/S1, physical cache 4.

Captured blocks: 0, 1, 7, 9. The capture hook recorded the first layer-0 attention call at each selected block.

## Results

| status | result |
|---|---|
| MATCHED_REAL_QKV_CAPTURE | PASS |
| RAW_QKV_PARITY | PASS — raw Q/K/V exact at all four captures |
| SELECTED_RETRIEVAL_ID_PARITY | PASS — exact where retrieval was active |
| ROPE_TENSOR_PARITY | PASS — transformed query and final K/V exact |
| ROPE_INDEX_METADATA_PARITY | FAIL — physical BA cache index metadata differs from AO logical metadata |
| FINAL_KV_PARITY | PASS — exact `final_k` and `final_v` |
| NATIVE_FA2_OUTPUT_PARITY | PASS — exact captured output |
| ARCHIVE_DIAGNOSTIC_STATE_PARITY | PARTIAL — unselected archive/pending representation differs at later blocks |
| D2H_BATCH_LIFETIME | PASS for event-guarded lifetime test; overlap NOT PROVEN |
| READY_FOR_SHORT_PERFORMANCE_TEST | NO |
| READY_FOR_F120 | NO |

All exact-equal tensor comparisons had max absolute error 0 and RMSE 0. No nonfinite values were observed in the compared tensors.

## First strict divergence

The first non-exact captured field is:

```text
block_001_layer_000.cache_rope_indices
AO shape: [6], values: [0, 1, 2, 3, 4, 5]
BA shape: [4], values: [0, 3, 4, 5]
```

This is a metadata/physical-residency difference, not a transformed-K/V difference: the corresponding `roped_query`, `final_k`, `final_v`, and attention output are exact. At block 7 and block 9 the same physical-vs-logical index distinction remains (`AO` has the full logical W20 index vector while BA records the four resident physical indices).

At block 9, diagnostic archive lists also differ for later unselected entries. The selected IDs remain `[0, 1]`, and the assembled `final_k/final_v` remain exact. Therefore this does not establish a retrieval-content failure, but it does mean the entire internal archive representation is not bytewise identical and must not be reported as full archive-state parity.

## Per-capture numerical evidence

For block 0, 1, 7 and 9 respectively, the following are exact between AO and BA: raw Q/K/V, roped query, final K/V, query RoPE indices, and captured Native FA2 output. `memory_indices` and `selected_ids` are exact whenever present.

The complete machine-readable comparison is in:

```text
results/phase218ba13_raw_qkv_parity.csv
results/phase218ba13_retrieval_parity.csv
results/phase218ba13_final_kv_parity.csv
results/phase218ba13_native_fa2_parity.csv
```

## D2H lifetime

The existing real CUDA test reports pinned destinations and event completion before promotion. Both the inherited nonblocking path and an explicit pinned copy had `event_query_before_stream_sync=false` and `event_query_after_stream_sync=true`. BA retains the exact event with its K/V batch, synchronizes that batch before promotion, retains the batch/event for partial promotion, and retires it only after all batch frames are consumed. This validates lifetime protection, not useful transfer/compute overlap.

## Provenance

The capture sources are isolated copies with capture-only hooks. Their hashes, configs, commands, and artifact hashes are in `results/phase218ba13_capture_manifest.json`. The BA capture source contains only the approved BA-1.2 source plus the isolated capture hook; the formal AO/BA worktrees were not changed.

## Conclusion

The tested physical4 BA path produces the same captured attention operands and Native FA2 result at the four requested real-model states. Strict metadata parity is not complete because BA necessarily exposes physical-cache RoPE indices rather than AO's full logical-cache index vector, and later diagnostic archive lists are not bytewise identical in unselected entries. The numerical attention gate is therefore PASS for the captured outputs, while the broader full-state parity gate remains NO/partial. No performance test or F120 should start from this result without deciding whether the metadata distinction is an accepted physical-cache representation or a bug.
