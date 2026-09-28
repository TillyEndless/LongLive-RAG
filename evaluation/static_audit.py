"""Read-only artifact discovery and minimum-action planning for coverage audits."""
import json
from pathlib import Path

from .experiment_registry import iter_experiments

KNOWN_ROOTS = [Path(p) for p in (
    "/home/zju/work/zxl/LongLive-RAG-common-group16_19-v2/results",
    "/home/zju/work/zxl/anemoi_native_draftattention_5090/results",
    "/data/zxl/LongLive-RAG-group11_15_h200/results",
)]

def _hardware(path):
    text = str(path)
    if "group11_15_h200" in text:
        return "H200"
    if "5090" in text or "common-group16_19" in text:
        return "RTX5090"
    return "UNKNOWN"

def _format(path):
    return "csv" if path.suffix.lower() == ".csv" else "json" if path.suffix.lower() == ".json" else path.suffix.lstrip(".") or "unknown"

def _read_json(path):
    try:
        if path.stat().st_size > 10 * 1024 * 1024:
            return {}
        return json.loads(path.read_text())
    except (OSError, ValueError, TypeError):
        return {}

def _files(root, patterns):
    found = []
    if not root.exists():
        return found
    for pattern in patterns:
        found.extend(root.rglob(pattern))
    return sorted({p for p in found if p.is_file()})

def _relevant(path, experiment, explicit_root):
    if explicit_root:
        return True
    text = str(path).lower()
    group = f"group{experiment['group']}"
    if group not in text:
        return False
    if experiment["group"] in (14, 15):
        suffix = experiment["experiment_id"].split(".")[-1]
        ratio = {"1": "sparse05", "2": "sparse10", "3": "sparse20", "4": "sparse30"}[suffix]
        return ratio in text or "group14" not in text and "group15" not in text
    return True

def _record(paths, category, reusable=False, reason="not found", valid=None, normalized=None):
    if not paths:
        return {"found": False, "paths": [], "format": "NOT_AVAILABLE", "provenance_status": "NOT_FOUND", "valid": False, "reusable": False, "reason": reason, "normalized": normalized or {}}
    hardware = sorted({_hardware(p) for p in paths})
    return {"found": True, "paths": [str(p) for p in paths], "format": sorted({_format(p) for p in paths}),
            "provenance_status": "MIXED_HARDWARE" if len(hardware) > 1 else hardware[0],
            "valid": bool(reusable if valid is None else valid), "reusable": bool(reusable), "reason": reason, "normalized": normalized or {}}

def _data(paths):
    for path in paths:
        value = _read_json(path)
        if value:
            return value
    return {}

def normalize_quality(data):
    return {key: data.get(key, "NOT_AVAILABLE") for key in ("DINO", "SSIM", "PSNR", "LPIPS")}

def normalize_memory(data):
    return {key: data.get(key, "NOT_AVAILABLE") for key in ("GPU_KV_GIB", "CPU_KV_GIB", "DRAFT_GPU_GIB", "TRANSIENT_GPU_GIB", "PEAK_GPU_ALLOCATED_GIB", "PEAK_GPU_RESERVED_GIB", "KV_COMPRESSION_RATIO")}

def normalize_latency(data):
    return {key: data.get(key, "NOT_AVAILABLE") for key in ("E2E_INFERENCE_S", "TRANSFORMER_S", "SELF_ATTN_WRAPPER_S", "ATTENTION_KERNEL_S", "NON_TRANSFORMER_E2E_S", "FETCH_WORK_S", "FETCH_EXPOSED_S", "FETCH_HIDDEN_S", "VAE_DECODE_S", "VIDEO_ENCODE_S", "PROCESS_WALL_S")}

def normalize_rag_strategy(data):
    keys = ("RETRIEVAL_WORK_S", "RETRIEVAL_EXPOSED_S", "FETCH_EVENT_COUNT", "FETCH_BYTES", "FETCH_REUSE_RATE", "EXACT_SET_RATE", "PREFETCH_PRECISION", "PREFETCH_RECALL")
    return {key: data.get(key, "NOT_APPLICABLE") for key in keys}

def _latency(paths):
    if not paths:
        return _record([], "latency")
    compatible = []
    legacy = []
    for path in paths:
        data = _read_json(path) if path.suffix == ".json" else {}
        keys = set(data)
        versioned = bool(data.get("timer_boundary_version"))
        repeated_sync = bool(data.get("repeated_cuda_synchronize") or data.get("profile") is True)
        if path.name == "runtime.csv" or repeated_sync or not versioned:
            legacy.append(path)
        else:
            compatible.append(path)
    if compatible:
        data = _data(compatible)
        return _record(paths, "latency", True, "explicit timer boundary version", valid=True, normalized=normalize_latency(data))
    return _record(paths, "latency", False, "legacy or unversioned timer boundary; discoverable but not reusable", valid=False)

def _quality(paths):
    valid, reusable = [], []
    for path in paths:
        data = _read_json(path)
        if all(isinstance(data.get(k), (int, float)) for k in ("DINO", "SSIM", "PSNR")) and (data.get("case_count", 10) >= 10):
            valid.append(path)
            if any(data.get(k) for k in ("manifest_sha256", "manifest_path")) and any(data.get(k) for k in ("evaluator_version", "reference_evaluator")) and data.get("frame_protocol"):
                reusable.append(path)
    return _record(paths, "quality", bool(reusable), "canonical metrics and evaluator provenance" if reusable else "quality metrics/canonical case or evaluator provenance incomplete", valid=bool(valid), normalized=normalize_quality(_data(paths)))

def _memory(paths):
    valid = []
    for path in paths:
        data = _read_json(path)
        if any(k in data for k in ("persistent_gpu_kv_bytes", "GPU_KV_MEASURED_GiB", "GPU_KV_GiB")) and any(k in data for k in ("persistent_cpu_kv_bytes", "CPU_KV_MEASURED_GiB", "CPU_KV_GiB")):
            valid.append(path)
    return _record(paths, "memory", bool(valid), "explicit GPU/CPU KV fields" if valid else "explicit persistent GPU and CPU KV fields missing", valid=bool(valid), normalized=normalize_memory(_data(paths)))

def discover_artifacts(experiment, roots=None):
    roots = [Path(p) for p in (roots or experiment.get("expected_artifact_roots", KNOWN_ROOTS))]
    names = [f"*{experiment['experiment_id'].replace('.', '*')}*"]
    all_files = []
    for root in roots:
        explicit = roots != KNOWN_ROOTS
        all_files.extend(p for p in _files(root, ("runtime.csv", "*runtime*.json", "*profile*.json", "*memory*.json", "memory.csv", "*quality*.json", "*metrics*.json", "summary.csv", "routing.json", "manifest.json", "provenance.json")) if _relevant(p, experiment, explicit))
    # ponytail: bounded filename scan; artifact roots are local campaign trees, and upgrade to a manifest index if scale requires it.
    all_files = sorted(set(all_files))
    quality = [p for p in all_files if any(x in p.name.lower() for x in ("quality", "metric", "summary"))]
    memory = [p for p in all_files if "memory" in p.name.lower() or (p.suffix == ".json" and "profile" in p.name.lower())]
    latency = [p for p in all_files if p.name == "runtime.csv" or "runtime" in p.name.lower() or "profile" in p.name.lower()]
    rag = [p for p in all_files if "routing" in p.name.lower()]
    provenance = [p for p in all_files if p.name in ("manifest.json", "provenance.json")]
    return {"quality": _quality(quality), "memory": _memory(memory), "latency": _latency(latency),
            "rag_strategy": _record(rag, "rag_strategy", bool(rag), "routing artifact found" if rag else "no routing artifact", normalized=normalize_rag_strategy(_data(rag))),
            "correctness": _record([], "correctness", False, "no correctness artifact discovered"),
            "provenance": _record(provenance, "provenance", bool(provenance), "manifest/provenance artifact found" if provenance else "no manifest/provenance artifact")}

def action_plan(experiment, artifacts):
    if experiment["group"] == 19 or experiment["implementation_semantics_confirmed"] != "YES":
        return "PROVENANCE_AUDIT_REQUIRED"
    q, l, m = (artifacts[x]["reusable"] for x in ("quality", "latency", "memory"))
    if not any(artifacts[x]["found"] for x in ("quality", "latency", "memory")):
        return "FULL_INFERENCE_REQUIRED"
    if not q and not l:
        return "QUALITY_AND_PROFILE_REQUIRED"
    if not q:
        return "QUALITY_ONLY_REQUIRED"
    if not l:
        return "PROFILE_ONLY_REQUIRED"
    if not m:
        return "MEMORY_ONLY_REQUIRED"
    if not artifacts["provenance"]["reusable"]:
        return "PROVENANCE_AUDIT_REQUIRED"
    return "COMPLETE"

def audit_all(group_filter=None, hardware=None):
    rows = []
    groups = set(group_filter or range(11, 20))
    for experiment in iter_experiments():
        if experiment["group"] not in groups:
            continue
        artifacts = discover_artifacts(experiment)
        if hardware:
            artifacts = {k: v for k, v in artifacts.items()}
            for item in artifacts.values():
                if item["found"]:
                    keep = [p for p in item["paths"] if _hardware(Path(p)) == hardware]
                    item.update(found=bool(keep), paths=keep, reusable=item["reusable"] and bool(keep))
        rows.append({"experiment": experiment, "artifacts": artifacts, "action": action_plan(experiment, artifacts)})
    return rows

def parse_group_filter(value):
    groups = set()
    for part in value.split(","):
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1)); groups.update(range(start, end + 1))
        else:
            groups.add(int(part))
    return groups
