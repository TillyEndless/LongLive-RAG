# Native W6/W12 profile field-mapping audit

## Conclusion

`WRAPPER_MISLABEL_CONFIRMED = YES` for the later old-vs-new timer audit
report. The values 12.669093 s and 12.823286 s are not self-attention
wrapper times. They are derived residuals:

`NON_TRANSFORMER_E2E_S = E2E_INFERENCE_S - TRANSFORMER_S`

The raw profiling outputs already contain the correct self-attention wrapper
field and were not modified.

## Authoritative mapping

| Window | E2E_INFERENCE_S | TRANSFORMER_S | SELF_ATTN_WRAPPER_S | NON_TRANSFORMER_E2E_S | ATTENTION_KERNEL_S |
|---:|---:|---:|---:|---:|---:|
| W6 | 40.435522 | 27.766429 | 14.990465 | 12.669093 | 6.185424 |
| W12 | 43.004939 | 30.181653 | 17.479825 | 12.823286 | 10.659426 |

## Field origins

- `E2E_INFERENCE_S` and `TRANSFORMER_S`: runtime metadata fields aggregated
  into `results/native_w6_w12_case01_profile.csv`.
- `SELF_ATTN_WRAPPER_S`: raw
  `results/native_w6_w12_case01_profile/w{6,12}_v2/native_case01_profile.json`,
  field `self_attention_wrapper_s`.
- Wrapper source: `profiling_configs/native_case01_profiler_v3.py:44`,
  `sum(x['host_s'] for x in forward_rows)`, where `host_s` is measured around
  `CausalWanSelfAttention.forward`.
- `ATTENTION_KERNEL_S`: raw profile field `attention_kernel_s`, accumulated
  from CUDA event pairs around the attention call.
- `NON_TRANSFORMER_E2E_S`: derived only as `E2E - Transformer`; it is not a
  direct profiler component timer.

## Corrections

The later `native_old70_vs_new40_timer_audit.md` and JSON had labeled the
derived residual as `Wrapper`. That label has been corrected to
`NON_TRANSFORMER_E2E_S`, while retaining the measured raw wrapper values.

No other field-mapping error was found in the inspected CSV, JSON, raw profile,
and audit artifacts. `NOT_AVAILABLE` remains distinct from zero; no residual
has been promoted to a measured component.

## Required answers

1. 12.669093 s and 12.823286 s are non-transformer E2E residuals.
2. Authoritative self-attention wrapper times are 14.990465 s and 17.479825 s.
3. Yes, the later audit report was mislabeled.
4. No additional mapping error affecting Group11 comparison was found.
