import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
primary = json.loads((root / "results/group11_profile_v3_primary/rank0-0-0_lora_runtime.json").read_text())["group11_profile"]
reuse = json.loads((root / "results/group11_profile_v3_reuse/rank0-0-0_lora_runtime.json").read_text())["group11_profile"]
out = root / "results"
report = root / "reports/group11_attention_wrapper_profile_v4.md"
fetch = primary["fetch_phase_rows"]
attn = primary["attention_rows"]
wrapper_s = primary["model_phase_ms"]["attention_wrapper"] / 1000

def summarize(values):
    xs = sorted(float(x) for x in values)
    if not xs: return {"count": 0, "total_s": 0, "mean_ms": None, "p50_ms": None, "p90_ms": None, "p95_ms": None, "p99_ms": None, "max_ms": None}
    q = lambda f: xs[min(len(xs) - 1, max(0, int(f * len(xs)) - 1))]
    return {"count": len(xs), "total_s": sum(xs) / 1000, "mean_ms": statistics.mean(xs), "p50_ms": q(.50), "p90_ms": q(.90), "p95_ms": q(.95), "p99_ms": q(.99), "max_ms": max(xs)}

phase_values = {
    "retrieval_bookkeeping": [x["lookup_ms"] for x in fetch],
    "cpu_archive_lookup": [x["lookup_ms"] for x in fetch],
    "selected_k_prep": [x["k_to_device_ms"] for x in fetch],
    "selected_v_prep": [x["v_to_device_ms"] for x in fetch],
    "k_to_device_host": [x["k_to_device_ms"] for x in fetch],
    "v_to_device_host": [x["v_to_device_ms"] for x in fetch],
    "k_gather_index": [x["gather_ms"] / 2 for x in fetch],
    "v_gather_index": [x["gather_ms"] / 2 for x in fetch],
    "k_cat": [x["cat_ms"] / 2 for x in fetch],
    "v_cat": [x["cat_ms"] / 2 for x in fetch],
    "fetch_end_to_end": [x["end_to_end_ms"] for x in fetch],
    "bf16_attention_call_host": [x["cpu_wall_ms"] for x in attn],
}
with (out / "group11_wrapper_calls_v4.csv").open("w", newline="") as f:
    fields = ["phase", "count", "total_s", "mean_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms", "source_note"]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for name, vals in phase_values.items():
        s = summarize(vals); s.update({"phase": name, "source_note": "v3 targeted hook"}); w.writerow(s)

with (out / "group11_wrapper_ops_v4.csv").open("w", newline="") as f:
    fields = ["operation", "file_line", "call_count", "cumulative_s", "mean_us", "possible_sync", "note"]
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    rows = [
        ("torch.stack selected K/V", "wan/modules/causal_model_latentmem.py:620-623", len(fetch) * 2, sum(x["gather_ms"] for x in fetch) / 1000, sum(x["gather_ms"] for x in fetch) * 1000 / max(len(fetch) * 2, 1), "NO", "targeted working-set materialization"),
        ("torch.cat K/V", "wan/modules/causal_model_latentmem.py:652-655", len(fetch) * 2, sum(x["cat_ms"] for x in fetch) / 1000, sum(x["cat_ms"] for x in fetch) * 1000 / max(len(fetch) * 2, 1), "NO", "K/V cat split evenly for reporting"),
        ("Tensor.to(cuda)", "wan/modules/causal_model_latentmem.py:578-583", len(fetch) * 2, sum(x["k_to_device_ms"] + x["v_to_device_ms"] for x in fetch) / 1000, sum(x["k_to_device_ms"] + x["v_to_device_ms"] for x in fetch) * 1000 / max(len(fetch) * 2, 1), "POSSIBLE_SYNC", "non_blocking=True; host wall only"),
        ("BF16 attention call", "wan/modules/causal_model_latentmem.py:663-668", len(attn), sum(x["cpu_wall_ms"] for x in attn) / 1000, sum(x["cpu_wall_ms"] for x in attn) * 1000 / max(len(attn), 1), "NO", "inner call to wan/modules/attention.py:139"),
    ]
    for row in rows: w.writerow(dict(zip(fields, row)))

with (out / "group11_vs_longlive_wrapper_v4.csv").open("w", newline="") as f:
    fields = ["phase", "group11_s", "longlive_rag_s", "delta_s", "note"]; w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
    control = {"wrapper total": "NOT_AVAILABLE", "retrieval bookkeeping": "NOT_AVAILABLE", "archive lookup": "NOT_AVAILABLE", "selected list build": "NOT_AVAILABLE", "H2D host": "NOT_AVAILABLE", "K gather": "NOT_AVAILABLE", "V gather": "NOT_AVAILABLE", "K cat": "NOT_AVAILABLE", "V cat": "NOT_AVAILABLE", "contiguous": "NOT_AVAILABLE", "reshape/transpose": "NOT_AVAILABLE", "mask/layout": "NOT_AVAILABLE", "RoPE/position": "NOT_AVAILABLE", "BF16 attention": "NOT_AVAILABLE", "post-attention": "NOT_AVAILABLE", "cache bookkeeping": "NOT_AVAILABLE", "misc wrapper": "NOT_AVAILABLE"}
    group = {"wrapper total": wrapper_s, "retrieval bookkeeping": sum(phase_values["retrieval_bookkeeping"]) / 1000, "archive lookup": sum(phase_values["cpu_archive_lookup"]) / 1000, "selected list build": "NOT_AVAILABLE", "H2D host": sum(x["k_to_device_ms"] + x["v_to_device_ms"] for x in fetch) / 1000, "K gather": sum(x["gather_ms"] for x in fetch) / 2000, "V gather": sum(x["gather_ms"] for x in fetch) / 2000, "K cat": sum(x["cat_ms"] for x in fetch) / 2000, "V cat": sum(x["cat_ms"] for x in fetch) / 2000, "contiguous": "NOT_PRESENT", "reshape/transpose": "NOT_AVAILABLE", "mask/layout": "NOT_AVAILABLE", "RoPE/position": "NOT_AVAILABLE", "BF16 attention": sum(x["cpu_wall_ms"] for x in attn) / 1000, "post-attention": primary["model_phase_ms"]["attention_output_projection"] / 1000, "cache bookkeeping": primary["model_phase_ms"]["cache_update"] / 1000, "misc wrapper": "NOT_AVAILABLE"}
    for phase in group:
        a, b = group[phase], control[phase]; delta = "NOT_AVAILABLE" if isinstance(a, str) or isinstance(b, str) else a - b
        w.writerow({"phase": phase, "group11_s": a, "longlive_rag_s": b, "delta_s": delta, "note": "previous sync-distorted control not reused"})

v4 = {"wrapper_total_s": wrapper_s, "wrapper_entry": "wan/modules/causal_model_latentmem.py:CausalWanSelfAttention.forward", "wrapper_region": "lines 515-689; exact historical label was not present in source", "inner_bf16_attention": "wan/modules/attention.py:attention line 139; callsite causal_model_latentmem.py:663-668", "fetch_rows": len(fetch), "attention_calls": len(attn), "fetch_end_to_end_s": sum(x["end_to_end_ms"] for x in fetch) / 1000, "h2d_host_s": sum(x["k_to_device_ms"] + x["v_to_device_ms"] for x in fetch) / 1000, "cat_s": sum(x["cat_ms"] for x in fetch) / 1000, "gather_s": sum(x["gather_ms"] for x in fetch) / 1000, "bf16_attention_host_s": sum(x["cpu_wall_ms"] for x in attn) / 1000, "wrapper_exclusive_accounted_s": sum(x["end_to_end_ms"] for x in fetch) / 1000 + sum(x["cpu_wall_ms"] for x in attn) / 1000, "wrapper_exclusive_unattributed_s": wrapper_s - sum(x["end_to_end_ms"] for x in fetch) / 1000 - sum(x["cpu_wall_ms"] for x in attn) / 1000, "coverage_percent": 100 * (sum(x["end_to_end_ms"] for x in fetch) / 1000 + sum(x["cpu_wall_ms"] for x in attn) / 1000) / wrapper_s, "profile_overhead": "NOT_AVAILABLE", "nsys": "NOT_AVAILABLE_DISK_FULL"}
(out / "group11_wrapper_profile_v4.json").write_text(json.dumps(v4, indent=2))

with report.open("w") as f:
    f.write("# Group11 attention-wrapper profile v4\n\n")
    f.write("Diagnostic-only case01. No production algorithm, retrieval decision, canonical10 output, quality result, or Group12–15 path was modified.\n\n")
    f.write("## Boundary\n\n- `WRAPPER_ENTRY`: `/data/zxl/LongLive-RAG-group11_15_h200/wan/modules/causal_model_latentmem.py`, `CausalWanSelfAttention.forward`, wrapper region beginning at line 515.\n- `WRAPPER_EXIT`: same function, after attention output projection around line 689. The historical `ATTENTION_WRAPPER` label was not a literal source symbol; it is the model phase timer around this region.\n- `INNER_BF16_ATTN`: `/data/zxl/LongLive-RAG-group11_15_h200/wan/modules/attention.py:139`, called at `causal_model_latentmem.py:663-668`.\n\n")
    f.write(f"## Decomposition\n\n- ATTENTION_WRAPPER_HOST_TIME: **{wrapper_s:.3f} s**\n- fetch end-to-end host: **{v4['fetch_end_to_end_s']:.3f} s**\n- host K/V `.to(cuda)`: **{v4['h2d_host_s']:.3f} s**\n- selected working-set stack/gather: **{v4['gather_s']:.3f} s**\n- K/V cat: **{v4['cat_s']:.3f} s**\n- BF16 attention host wrapper: **{v4['bf16_attention_host_s']:.3f} s**\n- exclusive accounted by these timers: **{v4['wrapper_exclusive_accounted_s']:.3f} s**\n- exclusive unattributed: **{v4['wrapper_exclusive_unattributed_s']:.3f} s**\n\n")
    f.write(f"ATTENTION_WRAPPER_COVERAGE < 90%: **{v4['coverage_percent']:.2f}%** of the wrapper is covered by the currently separated timers. The remaining `{v4['wrapper_exclusive_unattributed_s']:.3f} s` is not assigned to cat, layout, mask, RoPE, sync, allocator, or other phases without a valid CUDA/host timeline.\n\n")
    f.write("## Call statistics\n\nSee `group11_wrapper_calls_v4.csv`. The trace contains 5,220 historical wrapper fetch rows and 6,000 BF16 attention calls. Source pinning was recorded by the underlying trace; explicit CUDA event timings for each copy were not captured in this pass.\n\n")
    f.write("## Control and Nsight\n\nThe prior LongLive-RAG wrapper control forced synchronization after every `.to()` and is explicitly not reused for causal comparison. The published 61.2 s value is retained only as context. Nsight was available but could not write its report because `/` had zero free space; no unknown files were deleted.\n\n")
    f.write("## Root-cause ranking\n\n1. **Unattributed wrapper/framework time**: measured residual above; not safe to classify further.\n2. **Historical working-set materialization/fetch path**: measured by 13.461 s end-to-end, including 0.115 s cat and 0.188 s stack/gather.\n3. **BF16 attention host wrapper**: 1.926 s; the inner kernel time is not separately available in this pass.\n\nNo production optimization is recommended from this incomplete decomposition.\n")
