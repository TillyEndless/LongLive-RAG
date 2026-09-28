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
