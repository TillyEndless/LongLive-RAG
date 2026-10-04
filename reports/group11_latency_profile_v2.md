# Group11 latency root-cause profiling v2

Diagnostic-only case01. Production Group11 semantics and canonical10 outputs were not changed; Groups12–15 were not run.

## Preflight

- Commit: `25d7e7a52a50a28e1c7fbc7d4eaf1493a7fe0098`
- Root filesystem free: `0 KB`
- Profile filesystem `/data/zxl`: approximately `1.1 TB` free
- Nsight: available, but report write failed because `/` was full

## Measured comparison

|metric|Group11|LongLive-RAG W12|delta|
|---|---:|---:|---:|
|total_wall|189.0|61.2|127.800|
|draftmap|6.938321080058813|NOT_AVAILABLE|NOT_AVAILABLE|
|fetch_host|0.7716380665078759|NOT_AVAILABLE|NOT_AVAILABLE|
|attention_host|1.8321020044386387|NOT_AVAILABLE|NOT_AVAILABLE|
|fetch_requests|30270|NOT_AVAILABLE|NOT_AVAILABLE|
|h2d_calls|60540|NOT_AVAILABLE|NOT_AVAILABLE|
|h2d_bytes|290127052800|NOT_AVAILABLE|NOT_AVAILABLE|
|unique_entries|919|NOT_AVAILABLE|NOT_AVAILABLE|
|repeated_entries|29351|NOT_AVAILABLE|NOT_AVAILABLE|
|mean_kv_tokens|17113.2|NOT_AVAILABLE|NOT_AVAILABLE|

## Findings

- Wall proxy: `189.0s`; published W12 reference: `61.2s`; delta: `127.8s`.
- DraftMap score: `6.938s`; host copy timer: `0.772s`; BF16 attention host wrapper: `1.832s`. These visible phases do not explain the majority of wall time.
- Fetch duplication: `30270` requests, `919` unique layer/chunk pairs, `29351` repeats.
- Recorded sources were pinned and transfers were non-blocking; no pinning optimization was applied.
- Exact host lookup/K/V split, CUDA-event H2D time, cat/gather allocation, explicit/implicit synchronization, high-level FFN/QKV phases, adjacent-selection Jaccard, and reuse-cache timing remain NOT_AVAILABLE.

## Coverage and conclusion

`ROOT_CAUSE_COVERAGE < 80%`. The measured evidence confirms high-frequency retrieval orchestration and nontrivial DraftMap cost, but the unmeasured model/framework/synchronization/device-idle region remains dominant. It is not valid to attribute the 189–219s latency gap primarily to H2D, DraftMap, or BF16 attention yet.

The diagnostic GPU reuse-cache ablation was not run, and no production optimization was implemented. A valid completion requires a new host-wall fetch hook, a generation-region Nsight report written to `/data/zxl`, and the one-case reuse-cache ablation.
