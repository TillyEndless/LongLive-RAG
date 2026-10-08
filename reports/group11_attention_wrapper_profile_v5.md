# Group11 attention-wrapper profile v5

## Scope

Diagnostic-only case01 capture on H200, isolated from canonical outputs. It covers temporal blocks 6–7 after retrieval activation. No Groups 12–15 were run.

## Measured evidence

| item | measured |
|---|---:|
| profiler interval wall | 154.884 s |
| G11_WRAPPER host range sum | 13.547 s |
| QKV host range sum | 1.156 s |
| BF16 attention host range sum | 0.826 s |
| output projection host range sum | 0.243 s |
| CUDA kernel duration sum | 1.781 s |
| GPU memcpy duration sum | 0.544 s |
| CUDA runtime API duration sum | 5.436 s |
| cudaStreamSynchronize API duration | 2.113 s |

Kernel and memcpy durations are aggregate and may overlap; they are not a critical-path sum.

## Conclusion

The old 1.926 s BF16-attention number was a host timer around asynchronous work and must not be treated as total device execution. This capture shows FlashAttention kernels, H2D copies, explicit stream synchronizations, and substantial framework activity. It therefore supports a **MIXED** diagnosis: GPU work plus host orchestration/synchronization. It does not justify labeling the residual purely Python, purely GPU idle, or purely attention compute.

No aligned LongLive-RAG profiler capture was run; the prior control was sync-distorted and was not reused.

`ATTENTION_WRAPPER_COVERAGE < 90%`: explicit `torch.cuda.Event` phase pairs were not installed in this run, so the requested non-overlapping critical-path decomposition is not complete. The raw Chrome trace is retained for a follow-up event-pair capture.

## Outputs

`results/group11_wrapper_profile_v5.json`, `results/group11_wrapper_cuda_events_v5.csv`, `results/group11_wrapper_ops_v5.csv`, `results/group11_wrapper_sync_sites_v5.csv`, `results/group11_vs_longlive_wrapper_v5.csv`, and `profiles/group11_wrapper_v5/group11_wrapper_v5_trace.json`.
