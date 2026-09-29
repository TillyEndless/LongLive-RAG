# Unified evaluation pipeline

Entry point: `python -m evaluation.unified_pipeline --config <json> --mode reuse`.

The pipeline has two explicit modes:

- `reuse`: reads existing output/provenance/quality/latency/memory artifacts and never launches inference.
- `full`: runs only the explicit `inference_command` in the config, then consumes the resulting artifacts; resume/skip behavior belongs to that command.

The aggregate separates `quality`, `memory`, `latency`, `rag_strategy`,
`provenance`, and `validity`. Quality, latency, and memory may have different
sources, for example canonical10 quality plus matched case_01 profiling.

`NON_TRANSFORMER_E2E_S` is always the derived residual
`E2E_INFERENCE_S - TRANSFORMER_S`. It must not be labeled as
`SELF_ATTN_WRAPPER_S`. Missing fields remain `NOT_AVAILABLE`; RAG-only fields
are `NOT_APPLICABLE` when the method has no retrieval strategy.

Memory fields distinguish persistent KV, Draft GPU memory, transient
materialization/fetch memory, and peak allocator measurements. Compression is
reported from the explicit KV compression field, never from peak total memory.

Validity requires the relevant sections and provenance gates. `FINAL_ROW_VALID`
is true only when quality, latency, memory, provenance, and inference validity
are all true.

To add a method, create a JSON config that points to its existing artifact
paths, records manifest/checkpoint/commit provenance, and declares the mixed
quality/latency/memory sources. Do not copy videos into evaluation output.

Aggregate multiple summaries with:

```bash
python -m evaluation.aggregate_table results/evaluation/*/evaluation_summary.json \
  --output results/evaluation/baseline_table.csv
```


## H2D semantic contract

The pipeline keeps three distinct timing quantities:

- H2D_CUDA_WORK_S: device-side copy work measured by compatible CUDA events.
- H2D_EXPOSED_WAIT_S: host critical-path wait attributable to H2D.
- H2D_HOST_ENQUEUE_S: host time spent issuing/enqueuing H2D operations.

H2D_HIDDEN_S is emitted only when work and exposed-wait measurements have compatible timing boundaries. Otherwise it is NOT_AVAILABLE; it is not inferred by subtraction.

H2D provenance is preserved with source classes:
DEMAND_FETCH, PREFETCH, FLASH_FETCH, PROMOTION, CACHE_INIT, and LEGACY_HISTORICAL_FETCH.

Group 4.1/4.2 legacy h2d_latency_s is mapped to LEGACY_HISTORICAL_FETCH_H2D_CUDA_INTERVAL_S semantics. It remains discoverable, but is not silently compared as modern exposed wait.

Group11.3 reports FLASH_FETCH_H2D_CUDA_WORK_S, not exposed wait.
Group12/13 report CACHE_INIT_H2D_* for construction of the persistent low-bit owner, not steady-state retrieval.
Group14/15 report promotion CUDA work, host enqueue, bytes, and calls; promotion exposed wait remains NOT_AVAILABLE unless directly timed.

Zero is a measured zero. NOT_AVAILABLE means the field was not measured with compatible semantics. The aggregate table therefore has explicit H2D work, exposed wait, enqueue, bytes, calls, source-class, and timing-class columns and does not use one generic Exposed H2D column.
