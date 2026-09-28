"""Runtime storage accounting for persistent KV/Draft owners."""

from __future__ import annotations

import torch


def _storage_bytes(tensor, seen):
    if tensor is None or not hasattr(tensor, "untyped_storage"):
        return 0, 0
    storage = tensor.untyped_storage()
    key = (tensor.device.type, int(storage.data_ptr()), int(storage.nbytes()))
    logical = int(tensor.numel() * tensor.element_size())
    if key in seen:
        return 0, logical
    seen.add(key)
    return int(storage.nbytes()), logical


def _add(tensor, measured, logical, seen):
    m, l = _storage_bytes(tensor, seen)
    return measured + m, logical + l


def _tensor_attr(obj, name):
    """Resolve tensor-like quantizer fields across old/new FourOverSix APIs."""
    if isinstance(obj, torch.Tensor):
        return obj if name == "values" else None
    value = getattr(obj, name, None)
    if callable(value) and not isinstance(value, torch.Tensor):
        value = value()
    return value


def _object_bytes(obj):
    total = 0
    if obj is None:
        return 0
    if hasattr(obj, "untyped_storage"):
        return int(obj.untyped_storage().nbytes())
    for name in ("values", "scale_factors", "amax"):
        value = _tensor_attr(obj, name)
        if value is not None and hasattr(value, "untyped_storage"):
            total += int(value.untyped_storage().nbytes())
    return total


def _payload_value_bytes(obj):
    value = _tensor_attr(obj, "values")
    return _object_bytes(value if value is not None else obj)


def persistent_cache_memory(caches):
    """Return measured/logical KV and Draft bytes, split by device.

    Only tensors owned by the persistent cache are counted. Storage pointers
    are deduplicated because rolling views and metadata can alias owners.
    """
    out = {
        "gpu_kv_measured": 0, "gpu_kv_logical": 0,
        "cpu_kv_measured": 0, "cpu_kv_logical": 0,
        "draft_measured": 0, "draft_logical": 0,
    }
    seen = set()
    for cache in caches:
        owner = None
        if isinstance(cache, dict):
            owner = cache.get("anemoi_8bit_cache") or cache.get("anemoi_4bit_cache")
        if owner is not None:
            for chunk in owner.chunks:
                kv = ((chunk.k8, chunk.k_scale, chunk.v8, chunk.v_scale)
                      if hasattr(chunk, "k8") else
                      (chunk.k4, chunk.k_scale, chunk.v4, chunk.v_scale)) + (
                      chunk.valid_counts,)
                draft = (chunk.draft_k, chunk.draft_valid)
                for tensor in kv:
                    m, l = _storage_bytes(tensor, seen)
                    key = "gpu" if tensor is not None and tensor.device.type == "cuda" else "cpu"
                    out[f"{key}_kv_measured"] += m
                    out[f"{key}_kv_logical"] += l
                for tensor in draft:
                    m, l = _storage_bytes(tensor, seen)
                    out["draft_measured"] += m
                    out["draft_logical"] += l
            for tensor in (cache.get("global_end_index"), cache.get("local_end_index")):
                m, l = _storage_bytes(tensor, seen)
                key = "gpu" if tensor is not None and tensor.device.type == "cuda" else "cpu"
                out[f"{key}_kv_measured"] += m
                out[f"{key}_kv_logical"] += l
            continue

        # Persistent NVFP4 owner: count the actual backing tensors in slots.
        if isinstance(cache, dict) and cache.get("quantized"):
            for prefix in ("k", "v"):
                for slot in cache.get(prefix, []):
                    for tensor in (getattr(slot, "values", None),
                                   getattr(slot, "scale_factors", None),
                                   getattr(slot, "amax", None)):
                        m, l = _storage_bytes(tensor, seen)
                        key = "gpu" if tensor is not None and tensor.device.type == "cuda" else "cpu"
                        out[f"{key}_kv_measured"] += m
                        out[f"{key}_kv_logical"] += l
            for tensor in (cache.get("global_end_index"), cache.get("local_end_index")):
                m, l = _storage_bytes(tensor, seen)
                key = "gpu" if tensor is not None and tensor.device.type == "cuda" else "cpu"
                out[f"{key}_kv_measured"] += m
                out[f"{key}_kv_logical"] += l
            continue

        # Dense BF16 owner.
        if isinstance(cache, dict):
            for tensor in (cache.get("k"), cache.get("v")):
                m, l = _storage_bytes(tensor, seen)
                key = "gpu" if tensor is not None and tensor.device.type == "cuda" else "cpu"
                out[f"{key}_kv_measured"] += m
                out[f"{key}_kv_logical"] += l
    return out


__all__ = ["persistent_cache_memory"]


def runtime_inference_memory(caches, model, storage_mode="BF16_FAKE_QUANT"):
    """Measure the actual cache owners used by the H200 inference path."""
    seen = set()
    out = {
        "GPU_KV_MEASURED_BYTES": 0, "CPU_KV_MEASURED_BYTES": 0,
        "GPU_LOCAL_BF16_KV_BYTES": 0,
        "CPU_COMPRESSED_K_PAYLOAD_BYTES": 0,
        "CPU_COMPRESSED_V_PAYLOAD_BYTES": 0,
        "CPU_COMPRESSED_SCALE_META_BYTES": 0,
        "CPU_NVFP4_K_PAYLOAD_BYTES": 0,
        "CPU_NVFP4_V_PAYLOAD_BYTES": 0,
        "CPU_NVFP4_SCALE_META_BYTES": 0,
        "CPU_KV_TOTAL_BYTES": 0,
        "GPU_DRAFT_PERSISTENT_BYTES": 0, "CPU_DRAFT_PERSISTENT_BYTES": 0,
        "TRANSIENT_DEQUANT_GPU_BYTES": 0,
        "TRANSIENT_DEQUANT_GPU_PEAK_BYTES": 0,
        "FULL_HISTORY_BF16_SHADOW_BYTES": 0,
        "EVICTED_COMPRESSED_ENTRIES": 0,
        "RETRIEVED_ARCHIVED_ENTRIES": 0,
        "DRAFT_H2D_BYTES": 0, "DRAFT_H2D_CALLS": 0,
        "GPU_PACKED_K_BYTES": 0, "GPU_PACKED_V_BYTES": 0,
        "GPU_PACKED_SCALE_BYTES": 0, "GPU_PACKED_METADATA_BYTES": 0,
        "GPU_KV_BF16_EQUIVALENT_BYTES": 0,
        "GPU_KV_ACTUAL_PERSISTENT_BYTES": 0,
        "PERSISTENT_STORAGE_MODE": storage_mode,
    }
    def add(name, tensor):
        if tensor is None or not hasattr(tensor, "untyped_storage"):
            return
        s = tensor.untyped_storage()
        key = (tensor.device.type, int(s.data_ptr()), int(s.nbytes()))
        if key in seen:
            return
        seen.add(key)
        out[name] += int(s.nbytes())
    for cache in caches or []:
        if not isinstance(cache, dict):
            continue
        add("GPU_KV_MEASURED_BYTES", cache.get("k"))
        add("GPU_KV_MEASURED_BYTES", cache.get("v"))
        out["GPU_LOCAL_BF16_KV_BYTES"] += sum(int(t.untyped_storage().nbytes()) for t in (cache.get("k"), cache.get("v")) if t is not None)
        archive = cache.get("compressed_history_archive")
        entries = cache.get("compressed_history_entries", [])
        if archive is not None and entries:
            out["PERSISTENT_STORAGE_MODE"] = "LOWBIT_STORAGE_BF16_COMPUTE"
            out["EVICTED_COMPRESSED_ENTRIES"] += int(cache.get("evicted_compressed_entries", len(entries)))
            for rec in entries:
                cpu_bytes = int(rec.cpu_persistent_bytes()) if hasattr(rec, "cpu_persistent_bytes") else int(rec.persistent_bytes())
                out["CPU_KV_MEASURED_BYTES"] += cpu_bytes
                out["CPU_KV_TOTAL_BYTES"] += cpu_bytes
                shape = getattr(rec, "tensor_shape", None)
                if shape is not None:
                    out["GPU_KV_BF16_EQUIVALENT_BYTES"] += int(torch.tensor(shape).prod().item()) * 2 * 2
                kp, vp = getattr(rec, "gpu_k_payload", None), getattr(rec, "gpu_v_payload", None)
                if kp is not None:
                    out["GPU_PACKED_K_BYTES"] += _payload_value_bytes(kp)
                    out["GPU_PACKED_V_BYTES"] += _payload_value_bytes(vp)
                    out["GPU_PACKED_SCALE_BYTES"] += sum(_object_bytes(_tensor_attr(x, n)) for x in (kp, vp) for n in ("scale_factors", "amax"))
                    out["GPU_PACKED_SCALE_BYTES"] += _object_bytes(getattr(rec, "gpu_k_meta", None)) + _object_bytes(getattr(rec, "gpu_v_meta", None))
                    out["GPU_PACKED_METADATA_BYTES"] += 0
            out["CPU_COMPRESSED_K_PAYLOAD_BYTES"] = 0
            out["CPU_COMPRESSED_V_PAYLOAD_BYTES"] = 0
            out["CPU_COMPRESSED_SCALE_META_BYTES"] = 0
        for tensor in cache.get("cpu_k_frames", []):
            add("CPU_KV_MEASURED_BYTES", tensor)
        for tensor in cache.get("cpu_v_frames", []):
            add("CPU_KV_MEASURED_BYTES", tensor)
        for tensor in cache.get("gpu_draft_k_frames", []):
            add("GPU_DRAFT_PERSISTENT_BYTES", tensor)
    # Cache-local counters are per attention cache; do not accidentally read
    # the last loop variable from the archive accounting loop.
    for cache in caches or []:
        if isinstance(cache, dict):
            out["RETRIEVED_ARCHIVED_ENTRIES"] += int(cache.get("retrieved_archived_entries", 0))
            out["TRANSIENT_DEQUANT_GPU_BYTES"] += int(cache.get("transient_dequant_gpu_bytes", 0))
            out["TRANSIENT_DEQUANT_GPU_PEAK_BYTES"] = max(
                out["TRANSIENT_DEQUANT_GPU_PEAK_BYTES"],
                int(cache.get("transient_dequant_gpu_peak_bytes", 0)),
            )
    for block in getattr(model, "blocks", []):
        counters = getattr(getattr(block, "self_attn", None), "runtime_counters", {})
        out["DRAFT_H2D_BYTES"] += int(counters.get("draft_h2d_bytes", 0))
        out["DRAFT_H2D_CALLS"] += int(counters.get("draft_h2d_calls", 0))
        out["RETRIEVED_ARCHIVED_ENTRIES"] += int(counters.get("retrieved_archived_entries", 0))
        out["TRANSIENT_DEQUANT_GPU_BYTES"] += int(counters.get("transient_dequant_gpu_bytes", 0))
        out["TRANSIENT_DEQUANT_GPU_PEAK_BYTES"] = max(out["TRANSIENT_DEQUANT_GPU_PEAK_BYTES"], int(counters.get("transient_dequant_gpu_peak_bytes", 0)))
    out["GPU_KV_MEASURED_GiB"] = out["GPU_KV_MEASURED_BYTES"] / 2**30
    out["CPU_KV_MEASURED_GiB"] = out["CPU_KV_MEASURED_BYTES"] / 2**30
    out["GPU_DRAFT_PERSISTENT_GiB"] = out["GPU_DRAFT_PERSISTENT_BYTES"] / 2**30
    out["CPU_DRAFT_PERSISTENT_GiB"] = 0.0
    out["GPU_KV_ACTUAL_PERSISTENT_BYTES"] = out["GPU_KV_MEASURED_BYTES"] + out["GPU_PACKED_K_BYTES"] + out["GPU_PACKED_V_BYTES"] + out["GPU_PACKED_SCALE_BYTES"] + out["GPU_PACKED_METADATA_BYTES"]
    out["GPU_KV_BF16_EQUIVALENT_BYTES"] += out["GPU_LOCAL_BF16_KV_BYTES"]
    out["GPU_KV_COMPRESSION_RATIO"] = (out["GPU_KV_BF16_EQUIVALENT_BYTES"] / out["GPU_KV_ACTUAL_PERSISTENT_BYTES"]) if out["GPU_KV_ACTUAL_PERSISTENT_BYTES"] else 1.0
    out["KV_COMPRESSION_RATIO"] = out["GPU_KV_COMPRESSION_RATIO"]
    return out
