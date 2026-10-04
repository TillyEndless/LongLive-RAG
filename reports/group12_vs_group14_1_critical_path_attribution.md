# Group12 vs Group14.1 Critical-Path Attribution

## Scope and validity

This report uses the matched `case_01` workload and verifies 6,000 attention calls for each group. It does not run canonical10, Group14.2–14.4, or change inference semantics. The first instrumented attempt was retained separately but excluded because the executable entry point was `pipeline/causal_inference.py`; the v2 run used the corrected entry point and `CRITICAL_PATH_PROFILE.enabled=true`.

Execution order remained:

`full low-bit decode -> BF16 materialization/contiguous -> DraftMap routing -> K/V gather -> post-gather repack -> RoPE -> dense BF16 attention`.

The parent wrapper and attention-kernel authoritative values are:

* Group12 wrapper: `75.486982 s`; Group14.1 wrapper: `80.519119 s`.
* Kernel delta is `-2.061651 s` (saved `2.061651 s`).
* The target additional non-kernel overhead is `7.094000 s`.

## Independent measurements

CUDA columns are CUDA event work time. Host columns are CPU-observed wall time for the same phase. They are not interchangeable and nested timers are not added twice.

| Component | CUDA delta (s) | Host-observed delta (s) |
|---|---:|---:|
| full_decode | -0.110180 | -0.134871 |
| materialization | -0.037306 | -0.062085 |
| routing | 3.054777 | 4.134837 |
| cpu_detach_cpu | 0.311895 | 1.104757 |
| tolist | 0.161922 | 0.275624 |
| gather | 1.069220 | 0.858008 |
| repack | -0.004121 | -0.013477 |
| rope | -0.000394 | -0.012045 |
| h2d | -0.009005 | -0.011072 |

For additive closure, `full_decode` is split into `decode-exclusive = full_decode - BF16 materialization`, then materialization is added once. This avoids double-counting the nested materialization timer.

## Required attribution fields

* `MEASURED_POSITIVE_OVERHEAD_S = 6.373226`
* `EXPLAINED_OVERHEAD_S = 6.373226`
* `EXPLAINED_PERCENT = 89.84%`
* `RESIDUAL_S = 0.720774`
* `RESIDUAL_PERCENT = 10.16%`
* `FULL_DECODE_DELTA_S = -0.134871`
* `MATERIALIZATION_DELTA_S = -0.062085`
* `CPU_SYNC_WAIT_DELTA_S = 1.104757`
* `GATHER_K_DELTA_S = 0.451179`
* `GATHER_V_DELTA_S = 0.406829`
* `REPACK_DELTA_S = -0.013477`
* `ROPE_DELTA_S = -0.012045`
* `MASK_LAYOUT_DELTA_S = NOT_AVAILABLE`

`TOP1_OVERHEAD = draftmap_routing (4.134837 s)`, `TOP2_OVERHEAD = cpu_sync_wait (1.104757 s)`, `TOP3_OVERHEAD = gather_k_v (0.858008 s)`.

## Interpretation

The largest measured positive host-wall contributors are DraftMap routing, the CPU `.detach().cpu()` exposed wait, and K/V gather. Full decode, materialization, repack, RoPE, and H2D deltas are negative or near-zero in this matched run and are preserved as such. `.tolist()` is reported separately and is not folded into `.detach().cpu()`.

The current closure explains `89.84%`, below the requested 90% target, with `0.720774 s` residual. Therefore the evidence is insufficient to claim >=90% attribution. Mask/layout preparation has no independent boundary in this path, and explicit/implicit synchronization cannot be isolated beyond the observed CPU detach/copy wait. No value is fabricated to close the residual.

`SELECTIVE_DECODE_AS_NEXT_OPTIMIZATION = INCONCLUSIVE`: the measured full low-bit decode delta is negative (`-0.134871 s` host, `-0.110180 s` CUDA), so this experiment does not show that selective decode would reduce the Group14.1 overhead. The dominant measured positive overhead is routing/selection and host synchronization/gather, not full decode.
