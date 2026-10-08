# Group11 acceleration ablation aggregation

This report is generated from raw artifacts. Missing metrics are not inferred.

| Group | Status | DINO | SSIM | PSNR | GPU KV GiB | CPU KV GiB | Draft GiB | Transient GiB | Exact IDs | Quality source |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 11.1 | REFERENCE_VALIDATED | 0.8291 | 0.3718 | 10.7762 | 3.2135 | 28.9215 | 0.2317 | NOT_AVAILABLE | NOT_AVAILABLE | validated Group11.1 reference; historical E2E |
| 11.2 | GATE_PASS_EXISTING | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | 3.2135009766 | 28.9215087891 | 0.633430481 | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE: independent frozen evaluation pending |
| 11.3 | RUNNING_OR_PENDING | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |
| 11.4 | RUNNING_OR_PENDING | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE | NOT_AVAILABLE |

## Protocol sanity

- Window = 12 for all rows.
- CPU historical K/V = BF16.
- Final attention = BF16.
- Compression = N/A because these are scheduling/retrieval variants, not KV low-bit compression.

## Coverage

| Field | Producer/status |
|---|---|
| gpu_kv_gib | 3.2135, 3.2135009766, NOT_AVAILABLE |
| cpu_kv_gib | 28.9215, 28.9215087891, NOT_AVAILABLE |
| draft_gpu_gib | 0.2317, 0.633430481, NOT_AVAILABLE |
| transient_gpu_gib | NOT_AVAILABLE |
| compression | N/A |
| fetch_strategy | current-layer chunk-streaming Flash Fetch, layer-wise current-Q fetch, layer-wise previous-Q retrieval, one-layer-ahead speculative prefetch + correction fetch |
| dino | 0.8291, NOT_AVAILABLE |
| ssim | 0.3718, NOT_AVAILABLE |
| psnr | 10.7762, NOT_AVAILABLE |
| e2e_latency_s | 218.8, NOT_AVAILABLE |
| exposed_h2d_ms_per_call | NOT_AVAILABLE |
| wrapper_latency_ms_per_call | NOT_AVAILABLE |
| selected_id_exact_rate | NOT_AVAILABLE |
| final_latent_exact | NOT_AVAILABLE |
| quality_source | NOT_AVAILABLE, NOT_AVAILABLE: independent frozen evaluation pending, validated Group11.1 reference; historical E2E |
