import json, math, statistics, sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1])
runtime = root / "rank0-0-0_lora_runtime.json"
memory = root / "rank0-0-0_lora_memory_measurement.json"
d = json.loads(runtime.read_text())
p = d["group11_profile"]

def stats(values):
    xs = sorted(float(x) for x in values)
    if not xs:
        return {"count": 0, "sum": 0.0, "mean": None, "p50": None, "p95": None, "max": None}
    def q(frac):
        return xs[min(len(xs) - 1, max(0, math.ceil(frac * len(xs)) - 1))]
    return {"count": len(xs), "sum": sum(xs), "mean": statistics.mean(xs),
            "p50": q(.5), "p95": q(.95), "max": max(xs)}

draft = p["draftmap_rows"]
h2d = p["h2d_rows"]
attn = p["attention_rows"]
phase = {
    "Draft-Q pool": stats(x["time_draft_q_pool_ms"] for x in draft),
    "DraftMap score": stats(x["time_score_ms"] for x in draft),
    "DraftMap aggregate": stats(x["time_aggregate_ms"] for x in draft),
    "DraftMap top-k": stats(x["time_topk_ms"] for x in draft),
    "H2D copy": stats(x["copy_time_ms"] for x in h2d),
    "BF16 attention": stats(x["cpu_wall_ms"] for x in attn),
}
fetch_pairs = [(x["layer"], x["source_chunk_id"]) for x in h2d]
pair_counts = Counter(fetch_pairs)
total_bytes = sum(x["total_bytes"] for x in h2d)
total_copy_ms = sum(x["copy_time_ms"] for x in h2d)
kv_values = [x["kv_tokens"] for x in attn]
unique = len(pair_counts)
requests = len(fetch_pairs)
repeated = sum(max(0, n - 1) for n in pair_counts.values())

# Artifact creation time spans startup + generation + output writing. It is a
# wall-time proxy, not a CUDA-only timer.
total_latency_s = 1790490739 - 1790490550
result = {
    "case": "case_01",
    "protocol": "instrumented single-case rerun; algorithm and outputs unchanged",
    "total_group11_latency_s": total_latency_s,
    "phases_ms": phase,
    "counts": {k: p[k] for k in ["NUM_ATTENTION_CALLS", "NUM_DRAFTMAP_CALLS",
        "NUM_DRAFT_Q_POOL_CALLS", "NUM_DRAFT_K_SCORE_CALLS", "NUM_TOPK_CALLS",
        "NUM_CPU_KV_FETCH_CALLS", "NUM_H2D_COPY_CALLS"]},
    "total_full_kv_h2d_bytes": total_bytes,
    "total_full_kv_h2d_calls": p["NUM_H2D_COPY_CALLS"],
    "mean_h2d_bytes": total_bytes / max(len(h2d), 1),
    "effective_h2d_gbps": total_bytes / max(total_copy_ms / 1000, 1e-9) / 1e9,
    "unique_fetched_chunk_layer_pairs": unique,
    "total_fetch_requests": requests,
    "repeated_fetch_requests": repeated,
    "kv_token_stats": stats(kv_values),
    "kv_token_histogram": dict(Counter(kv_values)),
    "top_repeated_fetches": [
        {"layer": key[0], "chunk_id": key[1], "fetch_count": n,
         "total_bytes": sum(x["total_bytes"] for x in h2d
                             if (x["layer"], x["source_chunk_id"]) == key)}
        for key, n in pair_counts.most_common(20)
    ],
    "memory": json.loads(memory.read_text()),
}
(root / "group11_case01_profile.json").write_text(json.dumps(result, indent=2))

total_ms = total_latency_s * 1000
report = root / "group11_latency_profile.md"
with report.open("w") as f:
    f.write("# Group11 case01 latency profile\n\n")
    f.write("This is a profiling-only rerun of case01. Group11 semantics are unchanged: DraftMap/Draft-RAG active, GPU-only persistent Draft-K, CPU Draft-K=0, BF16 historical K/V archive, BF16 attention, no quantization or sparse routing. Existing canonical10 outputs were not overwritten.\n\n")
    f.write(f"- Approximate total wall time: **{total_latency_s:.1f} s** (artifact start-to-runtime metadata; includes startup and generation).\n")
    f.write("- Existing canonical Group11 case latency reference: ~218.8 s; this run is not a quality rerun.\n")
    f.write("- Nsight Systems: not run in this pass; phase traces are low-overhead CPU wall timers.\n\n")
    f.write("## Measured decomposition\n\n|phase|total ms|% wall|calls|mean ms|p50 ms|p95 ms|max ms|\n|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for name, value in phase.items():
        f.write(f"|{name}|{value['sum']:.1f}|{100*value['sum']/total_ms:.2f}|{value['count']}|{value['mean']:.4f}|{value['p50']:.4f}|{value['p95']:.4f}|{value['max']:.4f}|\n")
    measured = sum(v["sum"] for v in phase.values())
    f.write(f"|Measured instrumented phases|{measured:.1f}|{100*measured/total_ms:.2f}|-|-|-|-|-|\n\n")
    f.write("The residual is OTHER_CPU_OVERHEAD + synchronization/device-idle + uninstrumented model work; it is not assigned to a phase without a controlled CUDA timeline.\n\n")
    f.write("## Counts and H2D\n\n")
    f.write(f"- DraftMap calls: `{p['NUM_DRAFTMAP_CALLS']}`; attention calls: `{p['NUM_ATTENTION_CALLS']}`.\n")
    f.write(f"- Full-KV H2D bytes: `{total_bytes}`; H2D calls: `{p['NUM_H2D_COPY_CALLS']}`; mean per K/V pair: `{total_bytes/len(h2d):.0f}` bytes.\n")
    f.write(f"- Source pinned: `{sum(bool(x['source_pinned_memory']) for x in h2d)}/{len(h2d)}`; non-blocking: `{sum(bool(x['non_blocking']) for x in h2d)}/{len(h2d)}`.\n")
    f.write(f"- Effective measured copy bandwidth: `{result['effective_h2d_gbps']:.3f} GB/s`; this is host-side copy wall time, not CUDA-engine bandwidth.\n")
    f.write(f"- Unique `(layer, chunk)` pairs: `{unique}`; total fetch requests: `{requests}`; repeated requests beyond first: `{repeated}`.\n\n")
    f.write("## Working set and DraftMap scaling\n\n")
    f.write(f"- KV tokens/call: mean `{statistics.mean(kv_values):.1f}`, p50 `{result['kv_token_stats']['p50']}`, p95 `{result['kv_token_stats']['p95']}`, max `{max(kv_values)}`. Histogram: `{dict(Counter(kv_values))}`.\n")
    f.write("- Exact local-vs-retrieved token split was not emitted by the original cache path, so retrieved-token statistics are NOT_AVAILABLE rather than inferred.\n")
    f.write(f"- DraftMap score time: `{phase['DraftMap score']['sum']/1000:.3f} s`; top-k: `{phase['DraftMap top-k']['sum']/1000:.3f} s`; Q pool: `{phase['Draft-Q pool']['sum']/1000:.3f} s`.\n\n")
    f.write("## Root-cause ranking from measured data\n\n")
    for seconds, name in sorted((v["sum"] for v in phase.values()), reverse=True)[:0]:
        f.write(f"- {name}: {seconds/1000:.3f} s\n")
    for value, name in sorted(((v["sum"], k) for k, v in phase.items()), reverse=True)[:3]:
        f.write(f"1. **{name}**: `{value/1000:.3f} s` ({100*value/total_ms:.2f}% of total).\n")
    f.write("\nThe largest unexplained portion remains CPU/framework/device-idle or uninstrumented model work. Repeated fetches support investigating retrieval-result reuse, batched/pinned asynchronous fetch, and DraftMap reuse across denoising steps; no optimization was implemented.\n\n")
    f.write("## Baseline comparison\n\n")
    f.write("The existing LongLive-RAG W12 reference is ~61.2 s, but no aligned phase trace for its case01 was available. Phase-by-phase deltas are NOT_AVAILABLE; the value is a wall-time reference only.\n\n")
    f.write("## Operation audit\n\n")
    f.write("The instrumented path uses per-fetch `.to(device, non_blocking=True)`, per-layer tensor construction/stack/cat, top-k, and Python list bookkeeping. No new `torch.cuda.synchronize()` was inserted. Exact synchronization attribution requires Nsight Systems capture.\n")
print(root / "group11_case01_profile.json")
print(report)
