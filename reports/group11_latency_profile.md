# Group11 case01 latency profile

This is a profiling-only rerun of case01. Group11 semantics are unchanged: DraftMap/Draft-RAG active, GPU-only persistent Draft-K, CPU Draft-K=0, BF16 historical K/V archive, BF16 attention, no quantization or sparse routing. Existing canonical10 outputs were not overwritten.

- Approximate total wall time: **189.0 s** (artifact start-to-runtime metadata; includes startup and generation).
- Existing canonical Group11 case latency reference: ~218.8 s; this run is not a quality rerun.
- Nsight Systems: launch completed the generation region but could not write the `.nsys-rep`: the remote filesystem reported zero free space (`WriteStreamException: Is the disk full?`). No `.nsys-rep` was produced. The phase traces below remain valid low-overhead CPU wall timers.

## Measured decomposition

|phase|total ms|% wall|calls|mean ms|p50 ms|p95 ms|max ms|
|---|---:|---:|---:|---:|---:|---:|---:|
|Draft-Q pool|303.3|0.16|5220|0.0581|0.0571|0.1078|0.1777|
|DraftMap score|6938.3|3.67|5220|1.3292|1.3595|1.9033|28.9392|
|DraftMap aggregate|0.0|0.00|5220|0.0000|0.0000|0.0000|0.0000|
|DraftMap top-k|146.9|0.08|5220|0.0281|0.0248|0.0350|5.3627|
|H2D copy|771.6|0.41|30270|0.0255|0.0238|0.0332|0.1012|
|BF16 attention|1832.1|0.97|6000|0.3054|0.2815|0.4122|3.5499|
|Measured instrumented phases|9992.3|5.29|-|-|-|-|-|

The residual is OTHER_CPU_OVERHEAD + synchronization/device-idle + uninstrumented model work; it is not assigned to a phase without a controlled CUDA timeline.

## Counts and H2D

- DraftMap calls: `5220`; attention calls: `6000`.
- Full-KV H2D bytes: `290127052800`; H2D calls: `60540`; mean per K/V pair: `9584640` bytes.
- Source pinned: `30270/30270`; non-blocking: `30270/30270`.
- Effective measured copy bandwidth: `375.989 GB/s`; this is host-side copy wall time, not CUDA-engine bandwidth.
- Unique `(layer, chunk)` pairs: `919`; total fetch requests: `30270`; repeated requests beyond first: `29351`.

## Working set and DraftMap scaling

- KV tokens/call: mean `17113.2`, p50 `18720.0`, p95 `18720.0`, max `18720`. Histogram: `{4680: 150, 9360: 630, 10920: 150, 15600: 150, 18720: 4920}`.
- Exact local-vs-retrieved token split was not emitted by the original cache path, so retrieved-token statistics are NOT_AVAILABLE rather than inferred.
- DraftMap score time: `6.938 s`; top-k: `0.147 s`; Q pool: `0.303 s`.

## Root-cause ranking from measured data

1. **DraftMap score**: `6.938 s` (3.67% of total).
1. **BF16 attention**: `1.832 s` (0.97% of total).
1. **H2D copy**: `0.772 s` (0.41% of total).

The largest unexplained portion remains CPU/framework/device-idle or uninstrumented model work. Repeated fetches support investigating retrieval-result reuse, batched/pinned asynchronous fetch, and DraftMap reuse across denoising steps; no optimization was implemented.

## Baseline comparison

The existing LongLive-RAG W12 reference is ~61.2 s, but no aligned phase trace for its case01 was available. Phase-by-phase deltas are NOT_AVAILABLE; the value is a wall-time reference only.

## Operation audit

The instrumented path uses per-fetch `.to(device, non_blocking=True)`, per-layer tensor construction/stack/cat, top-k, and Python list bookkeeping. No new `torch.cuda.synchronize()` was inserted. Exact GPU-active versus device-idle attribution remains NOT_AVAILABLE because Nsight Systems could not write its report on the full filesystem. The failed capture did not modify canonical outputs.
