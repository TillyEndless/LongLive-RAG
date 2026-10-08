# Group12 vs Group14.1 overhead attribution

## Scope and contract

Matched existing case_01, W12, seed=0, 6000 attention calls per variant. No canonical10 or Group14.2/14.3/14.4 inference was run by this analysis.

## Known equation

| quantity | Group12 | Group14.1 |
|---|---:|---:|
| self-attention wrapper (s) | 75.487 | 80.519 |
| attention CUDA work (s) | 10.717 | 8.655 |
| non-kernel wrapper (s) | 64.770 | 71.864 |
| additional non-kernel (s) | colspan=2: 7.094 |

The kernel saving is 2.062 s. The remaining 7.094 s is a critical-path quantity, not the sum of nested component timers.

## Measured work-time components

| component | Group12 work (s) | Group14.1 work (s) | delta (s) |
|---|---:|---:|---:|
| Draft Q pooling | 0.335034 | 0.346830 | 0.011797 |
| DraftMap score (matmul+softmax combined) | 6.144553 | 5.764901 | -0.379652 |
| importance reduction | 0.000000 | 0.000000 | 0.000000 |
| TopK/block selection | 0.149882 | 0.147436 | -0.002446 |
| gather + cat/repack | 0.294288 | 0.297687 | 0.003399 |
| K/V gather split | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| dequant/materialization | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| RoPE/mask/layout split | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| host sync wait | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |

Measured work-time delta for the available routing+selection+gather fields is **-0.366903 s**, which is -5.2% of the 7.094 s critical-path delta. This over-coverage proves nested/overlapping accounting; it cannot be reported as exposed overhead.

## Decode order and synchronization audit

`archive.fetch` dequantizes/materializes the selected historical record at `wan/modules/causal_model_latentmem.py:802-813`; only afterwards does `route_draftmap` pool/score/select and gather BF16 K/V at `utils/persistent_draftmap.py:32-43`. Therefore the measured code order is **FULL_DECODE_THEN_GATHER**. The route implementation also materializes selected IDs/scores through `.detach().cpu().tolist()` at lines 58-59, a definite potential GPU→CPU wait site; its wait duration was not measured in the saved trace.

**Default stream:** recorded attention events use the current/default stream; exact overlap among routing subphases is not measured. The saved trace has no independent dequant, host-wait, RoPE, or mask-layout timers.

## Conclusion

- Attention arithmetic is reduced: QK elements fall to 0.704242 of Group12 and CUDA attention work falls by 2.062 s.
- The available routing score, selection, and gather/repack timers do not explain the 7.094 s delta: their measured work delta is -0.366903 s.
- The exact 90% critical-path attribution is **not achieved** by existing artifacts. Residual and top-3 exposed contributors are therefore **NOT_IDENTIFIABLE**, not guessed.
- Root cause classification: **UNRESOLVED_WITH_EXISTING_TRACE**.
- Break-even retained ratio: **NOT_IDENTIFIABLE_YET**; one retained-ratio point cannot fit a model.

## Outputs

- Raw per-call CSV: `/data/zxl/LongLive-RAG-group11_15_h200/results/group12_vs_group14_1_overhead_raw.csv`
- Aggregate JSON: `/data/zxl/LongLive-RAG-group11_15_h200/results/group12_vs_group14_1_overhead_aggregate.json`
- Report: `/data/zxl/LongLive-RAG-group11_15_h200/reports/group12_vs_group14_1_overhead_attribution.md`
