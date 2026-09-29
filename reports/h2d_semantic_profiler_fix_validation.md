# H2D semantic profiler fix validation

## Scope

Validation was performed in an isolated worktree. The active dirty H200 worktree and existing results were not modified. No canonical10 quality run was performed.

## Gates

| Gate | Result |
|---|---|
| Python static compile | PASS |
| Compressed archive API check | PASS |
| Group11.3 minimal GPU profile | PASS |
| Group14 minimal GPU profile | PASS |
| Group15 minimal GPU profile | PASS |

## Observed semantic fields

| Variant | CUDA H2D work (s) | H2D bytes | H2D calls | Promotion CUDA work (s) | Promotion host enqueue (s) | Legacy exposed H2D (s) |
|---|---:|---:|---:|---:|---:|---:|
| g11_3 | 5.610501 | 290127052800 | 60540 | NOT_AVAILABLE | NOT_AVAILABLE | 0.000000 |
| g14 | 0.006391 | 287539200 | 60 | 0.006391 | 0.002266 | 0.000000 |
| g15 | 0.006427 | 287539200 | 60 | 0.006427 | 0.002307 | 0.000000 |

## Semantic decisions

- H2D_CUDA_WORK_* is populated only from CUDA event intervals or explicitly classified component work.
- H2D_EXPOSED_WAIT_* remains NOT_AVAILABLE when the path does not measure a compatible host wait interval.
- Group11.3 now aggregates its per-call async flash trace into profile-level CUDA work/bytes/calls.
- Group14/15 promotion records deferred CUDA event work and host enqueue separately; no hot-path synchronization was added.
- Existing legacy EXPOSED_H2D_* fields are retained and are not relabeled.

## Required status fields

```text
H2D_SEMANTIC_FIX_IMPLEMENTED=YES
GROUP11_3_AGGREGATION_FIXED=YES
GROUP14_15_PROMOTION_TIMING_FIXED=YES
GROUP12_13_CACHE_INIT_RECLASSIFIED=YES (cache-init bytes/calls classified; CUDA work remains NOT_AVAILABLE)
LEGACY_RAG_H2D_MAPPED=YES (unified schema adapter preserves legacy interval semantics)
UNIFIED_PIPELINE_UPDATED=YES
PER_CALL_CUDA_SYNCHRONIZE_ADDED=NO
STATIC_TESTS_PASS=YES
MINIMAL_GPU_VALIDATION_PASS=YES
QUALITY_RERUN_OCCURRED=NO
COMMIT_SHA=678fd5d
PUSH_STATUS=SUCCESS
```

## Additional static checks

- Unified schema smoke test: PASS (h2d_latency_s maps to H2D_CUDA_WORK_S with legacy timing-class metadata).
- Cache-init accounting: bytes/calls are separated from steady-state promotion; CUDA work remains NOT_AVAILABLE rather than being fabricated.

## Final packaging status

H2D_DEFINITION_FROZEN=YES
GROUP11_3_AGGREGATION_FIXED=YES
GROUP14_15_PROMOTION_TIMING_FIXED=YES
LEGACY_MAPPING_COMPLETE=YES
UNIFIED_SCHEMA_UPDATED=YES
UNIFIED_PIPELINE_UPDATED=YES
AGGREGATE_TABLE_UPDATED=YES
STATIC_TESTS_PASS=YES
INFERENCE_RERUN=NO
LATENCY_RERUN=NO
CANONICAL10_RERUN=NO
