# H200 Group11–15 Static Alignment Audit

Status: **logic/config alignment completed; smoke and inference intentionally not started**.

## Frozen contract

- W12, `retrieval_backend=draftmap_online`, `memory_size=6`, `recent_exclude=5`.
- CPU historical K/V remain authoritative BF16.
- Group11 uses BF16 GPU KV; Groups12/14 use persistent GPU K=INT8, V=FP8 E4M3.
- Groups13/15 use persistent GPU K/V=NVFP4.
- Groups12–15 dequantize selected packed KV into bounded temporary BF16 operands; final attention remains BF16.
- Group14/15 ratios are explicit retained interaction ratios (30/20/10/5%), and routing metadata records retained and skipped interactions.
- `num_output_frames=120` is the latent-frame configuration; the frozen evaluator requires the decoded 474-frame, 16-FPS, 832x480 protocol and frame 0 vs 237.

## Code changes

- `inference.py`: fail-fast contract validation for Group12–15, explicit query/prefetch flags, owner/temporary-memory metadata, and the latent-to-decoded frame contract.
- `utils/h200_group_runtime.py`: low-bit quantization is applied only to archive-backed persistent ownership; local/sink/current transient tensors stay BF16 and are not quantize→dequantized accidentally.
- `wan/modules/causal_model_latentmem.py`: local-only attention path explicitly disables storage quantization.
- `utils/runtime_memory_measurement.py`: cache-local counters are aggregated per cache instead of using a stale loop variable; persistent packed, BF16-equivalent, transient, CPU archive, and Draft fields remain separate.
- `scripts/evaluate_group12_15_corrected.py`: points to the new persistent campaign root and rejects stale outputs unless runtime metadata proves the corrected owner semantics, final BF16 attention, and disabled qprev/FlashFetch/prefetch paths.

## Static gates

| Gate | Result |
|---|---|
| New campaign configs | 100 jobs, Group12/13 + Group14/15 sparse ratios |
| Canonical manifest | SHA256 `db9462c0954186b9128543b58fb6d1d496a8b89e0754333beca698d3cf461fec` |
| Seed | 0 |
| Latent frame config | 120 |
| Corrected storage mode | `LOWBIT_STORAGE_BF16_COMPUTE` |
| CPU history dtype | BF16 |
| Final attention dtype | BF16 |
| Native low-bit kernel | Disabled (`NO`) |
| qprev / FlashFetch / next-layer prefetch / hot cache | Disabled for Groups12–15 |
| Python compile | PASS |
| GPU smoke/inference | **NOT RUN** |

## Group11 variants

Group11.1 remains the current-Q BF16 reference. Group11.2 is previous-Q direct retrieval, Group11.3 is the serial Flash-Fetch correctness prototype (not proven async overlap), and Group11.4 is next-layer prediction with true current-Q correction. These are separate worktrees and are not silently relabeled as the Group12–15 storage experiments. Their independent 10-case quality/exactness gates remain required before inheriting Group11.1 quality.

