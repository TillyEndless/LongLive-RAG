# Final Group11 latency root-cause report

Matched case01, seed=0, W12, GPU0, same checkpoint/LoRA environment, and temporal blocks 6–7. Every measured phase used CUDA Event start/end pairs, followed by one final synchronization after inference.

Wrappers: Group11=`wan/modules/causal_model_latentmem.py:CausalWanSelfAttention.forward`; LongLive=`wan/modules/causal_model.py:CausalWanSelfAttention.forward`. Both call `wan/modules/attention.py:attention`.

| phase | Group11 CUDA s | LongLive CUDA s | delta s |
|---|---:|---:|---:|
| Q projection | 0.020343 | 0.015128 | 0.005215 |
| K projection | 0.019567 | 0.015961 | 0.003606 |
| V projection | 0.026850 | 0.015969 | 0.010881 |
| BF16 attention | 0.528048 | 0.551642 | -0.023594 |
| output projection | 0.014574 | 0.014036 | 0.000538 |
| wrapper boundary | 4.393256 | 1.053488 | 3.339768 |

The BF16 attention kernel is not materially slower in Group11: the measured CUDA delta is -0.023594 s over 300 calls. The wrapper boundary is larger by 3.339768 s, but this includes orchestration/materialization outside the attention call.

DraftMap, fetch, working-set subphases, runtime API counts, pinned allocation, and synchronization provenance were not separately captured in this final event run. They are marked `NOT_AVAILABLE`; old Group11-only numbers were not substituted. Matched torch.profiler with_stack was also not run in this final pass. Therefore final root-cause coverage is below 90%.

FINAL_BOUNDED_GROUP11_S = 4.914063
FINAL_BOUNDED_LONGLIVE_S = 1.557214
FINAL_BOUNDED_DELTA_S = 3.356849
FINAL_ACCOUNTED_DELTA_S = 3.339768
FINAL_ROOT_CAUSE_COVERAGE_PERCENT = NOT_COMPUTABLE
DRAFTMAP_INCREMENT_S = NOT_AVAILABLE
FETCH_INCREMENT_HOST_S = NOT_AVAILABLE
FETCH_INCREMENT_CUDA_S = NOT_AVAILABLE
WORKINGSET_INCREMENT_S = NOT_AVAILABLE
CUDAHOSTALLOC_INCREMENT_S = NOT_AVAILABLE
SYNC_INCREMENT_S = NOT_AVAILABLE
ATTENTION_KERNEL_INCREMENT_S = -0.023594
OTHER_FRAMEWORK_INCREMENT_S = NOT_AVAILABLE
PRIMARY_MEASURED_ROOT_CAUSE = NOT_ESTABLISHED
SECONDARY_MEASURED_ROOT_CAUSE = NOT_ESTABLISHED
THIRD_MEASURED_ROOT_CAUSE = NOT_ESTABLISHED
BF16_ATTENTION_KERNEL_IS_PRIMARY_CAUSE = NO
RAW_H2D_IS_PRIMARY_CAUSE = NOT_AVAILABLE
PRODUCTION_CODE_MODIFIED = NO
CANONICAL_OUTPUTS_MODIFIED = NO
