import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
report = Path(sys.argv[3])
runtime = json.loads((root / "rank0-0-0_lora_runtime.json").read_text())
p = runtime["group11_profile"]
h2d = p["h2d_rows"]
draft = p["draftmap_rows"]
attn = p["attention_rows"]
groups = defaultdict(list)
for i, row in enumerate(h2d):
    groups[(row.get("layer"), row.get("source_chunk_id"))].append((i, row))

with (out_dir / "group11_fetch_trace.csv").open("w", newline="") as f:
    fields = ["global_fetch_call", "layer_id", "history_chunk_id", "bytes_K", "bytes_V", "total_bytes", "host_copy_ms", "cuda_copy_ms", "host_minus_cuda_ms", "pinned_cpu", "non_blocking", "stream", "note"]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for i, x in enumerate(h2d):
        w.writerow({"global_fetch_call": i, "layer_id": x.get("layer", "NA"), "history_chunk_id": x.get("source_chunk_id", "NA"), "bytes_K": x["bytes_K"], "bytes_V": x["bytes_V"], "total_bytes": x["total_bytes"], "host_copy_ms": x["copy_time_ms"], "cuda_copy_ms": "NOT_AVAILABLE", "host_minus_cuda_ms": "NOT_AVAILABLE", "pinned_cpu": x["source_pinned_memory"], "non_blocking": x["non_blocking"], "stream": x.get("copy_stream", "default"), "note": "CUDA event was not attached by first hook"})

with (out_dir / "group11_fetch_reuse.csv").open("w", newline="") as f:
    fields = ["layer_id", "history_chunk_id", "first_fetch_global_call", "last_fetch_global_call", "fetch_count", "total_bytes", "cumulative_host_ms", "consecutive_reuses", "denoising_step", "note"]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for (layer, chunk), rows in sorted(groups.items()):
        w.writerow({"layer_id": layer, "history_chunk_id": chunk, "first_fetch_global_call": rows[0][0], "last_fetch_global_call": rows[-1][0], "fetch_count": len(rows), "total_bytes": sum(x["total_bytes"] for _, x in rows), "cumulative_host_ms": sum(x["copy_time_ms"] for _, x in rows), "consecutive_reuses": sum(1 for (a, _), (b, _) in zip(rows, rows[1:]) if b == a + 1), "denoising_step": "NOT_AVAILABLE", "note": "step IDs not attached"})

with (out_dir / "group11_step_breakdown.csv").open("w", newline="") as f:
    fields = ["denoising_step", "current_chunk_id", "draftmap_calls", "history_chunks_mean", "candidate_chunks_mean", "selected_chunks_mean", "draft_q_pool_ms", "draft_score_ms", "draft_aggregate_ms", "draft_topk_ms", "attention_wall_ms", "unattributed_step_wall_ms", "note"]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    by = defaultdict(list)
    for x in draft: by[(x.get("denoising_step"), x.get("current_chunk_id"))].append(x)
    for (step, chunk), rows in sorted(by.items(), key=lambda z: str(z[0])):
        avg = lambda k: statistics.mean(float(x.get(k, 0) or 0) for x in rows)
        total = lambda k: sum(float(x.get(k, 0) or 0) for x in rows)
        w.writerow({"denoising_step": step, "current_chunk_id": chunk, "draftmap_calls": len(rows), "history_chunks_mean": avg("num_historical_chunks"), "candidate_chunks_mean": avg("num_candidate_chunks"), "selected_chunks_mean": avg("num_selected_chunks"), "draft_q_pool_ms": total("time_draft_q_pool_ms"), "draft_score_ms": total("time_score_ms"), "draft_aggregate_ms": total("time_aggregate_ms"), "draft_topk_ms": total("time_topk_ms"), "attention_wall_ms": "NOT_AVAILABLE", "unattributed_step_wall_ms": "NOT_AVAILABLE", "note": "step wall timer not attached"})

v2 = {"status": "PARTIAL_DIAGNOSTIC_COMPLETE", "source_commit": "25d7e7a52a50a28e1c7fbc7d4eaf1493a7fe0098", "case": "case01", "production_semantics_unchanged": True, "root_fs_free": "0 KB", "profile_output_fs": "/data/zxl", "profile_output_free": "~1.1 TB", "draftmap_calls": p["NUM_DRAFTMAP_CALLS"], "attention_calls": p["NUM_ATTENTION_CALLS"], "fetch_requests": len(h2d), "unique_layer_chunk_pairs": len(groups), "repeated_fetch_requests": len(h2d) - len(groups), "h2d_bytes": sum(x["total_bytes"] for x in h2d), "mean_kv_tokens": statistics.mean(x["kv_tokens"] for x in attn), "reuse_ablation": "NOT_RUN", "nsys": "FAILED_DISK_FULL", "root_cause_coverage": "<80%"}
(out_dir / "group11_case01_profile_v2.json").write_text(json.dumps(v2, indent=2))
(out_dir / "group11_reuse_ablation.json").write_text(json.dumps({"status": "NOT_RUN", "reason": "No temporary GPU reuse cache was introduced."}, indent=2))

group11 = {"total_wall": 189.0, "draftmap": sum(x["time_score_ms"] for x in draft) / 1000, "fetch_host": sum(x["copy_time_ms"] for x in h2d) / 1000, "attention_host": sum(x["cpu_wall_ms"] for x in attn) / 1000, "fetch_requests": len(h2d), "h2d_calls": p["NUM_H2D_COPY_CALLS"], "h2d_bytes": sum(x["total_bytes"] for x in h2d), "unique_entries": len(groups), "repeated_entries": len(h2d) - len(groups), "mean_kv_tokens": statistics.mean(x["kv_tokens"] for x in attn)}
control = {"total_wall": 61.2, "draftmap": "NOT_AVAILABLE", "fetch_host": "NOT_AVAILABLE", "attention_host": "NOT_AVAILABLE", "fetch_requests": "NOT_AVAILABLE", "h2d_calls": "NOT_AVAILABLE", "h2d_bytes": "NOT_AVAILABLE", "unique_entries": "NOT_AVAILABLE", "repeated_entries": "NOT_AVAILABLE", "mean_kv_tokens": "NOT_AVAILABLE"}
with (out_dir / "group11_vs_longlive_profile.csv").open("w", newline="") as f:
    fields = ["metric", "group11", "longlive_rag_w12", "delta", "note"]; w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for key in group11:
        a, b = group11[key], control[key]
        delta = "NOT_AVAILABLE" if isinstance(a, str) or isinstance(b, str) else a - b
        w.writerow({"metric": key, "group11": a, "longlive_rag_w12": b, "delta": delta, "note": "LongLive value is published wall-only reference; no aligned phase trace"})

with report.open("w") as f:
    f.write("# Group11 latency root-cause profiling v2\n\n")
    f.write("Diagnostic-only case01. Production Group11 semantics and canonical10 outputs were not changed; Groups12–15 were not run.\n\n")
    f.write("## Preflight\n\n- Commit: `25d7e7a52a50a28e1c7fbc7d4eaf1493a7fe0098`\n- Root filesystem free: `0 KB`\n- Profile filesystem `/data/zxl`: approximately `1.1 TB` free\n- Nsight: available, but report write failed because `/` was full\n\n")
    f.write("## Measured comparison\n\n|metric|Group11|LongLive-RAG W12|delta|\n|---|---:|---:|---:|\n")
    for key in group11:
        a, b = group11[key], control[key]
        delta = "NOT_AVAILABLE" if isinstance(a, str) or isinstance(b, str) else f"{a-b:.3f}"
        f.write(f"|{key}|{a}|{b}|{delta}|\n")
    f.write("\n## Findings\n\n")
    f.write(f"- Wall proxy: `{group11['total_wall']:.1f}s`; published W12 reference: `{control['total_wall']:.1f}s`; delta: `{group11['total_wall']-control['total_wall']:.1f}s`.\n")
    f.write(f"- DraftMap score: `{group11['draftmap']:.3f}s`; host copy timer: `{group11['fetch_host']:.3f}s`; BF16 attention host wrapper: `{group11['attention_host']:.3f}s`. These visible phases do not explain the majority of wall time.\n")
    f.write(f"- Fetch duplication: `{group11['fetch_requests']}` requests, `{group11['unique_entries']}` unique layer/chunk pairs, `{group11['repeated_entries']}` repeats.\n")
    f.write("- Recorded sources were pinned and transfers were non-blocking; no pinning optimization was applied.\n")
    f.write("- Exact host lookup/K/V split, CUDA-event H2D time, cat/gather allocation, explicit/implicit synchronization, high-level FFN/QKV phases, adjacent-selection Jaccard, and reuse-cache timing remain NOT_AVAILABLE.\n\n")
    f.write("## Coverage and conclusion\n\n")
    f.write("`ROOT_CAUSE_COVERAGE < 80%`. The measured evidence confirms high-frequency retrieval orchestration and nontrivial DraftMap cost, but the unmeasured model/framework/synchronization/device-idle region remains dominant. It is not valid to attribute the 189–219s latency gap primarily to H2D, DraftMap, or BF16 attention yet.\n\n")
    f.write("The diagnostic GPU reuse-cache ablation was not run, and no production optimization was implemented. A valid completion requires a new host-wall fetch hook, a generation-region Nsight report written to `/data/zxl`, and the one-case reuse-cache ablation.\n")
