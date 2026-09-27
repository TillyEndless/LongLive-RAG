"""Runtime storage accounting for persistent KV/Draft owners."""

from __future__ import annotations


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
        "GPU_DRAFT_PERSISTENT_BYTES": 0, "CPU_DRAFT_PERSISTENT_BYTES": 0,
        "DRAFT_H2D_BYTES": 0, "DRAFT_H2D_CALLS": 0,
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
        for tensor in cache.get("cpu_k_frames", []):
            add("CPU_KV_MEASURED_BYTES", tensor)
        for tensor in cache.get("cpu_v_frames", []):
            add("CPU_KV_MEASURED_BYTES", tensor)
        for tensor in cache.get("gpu_draft_k_frames", []):
            add("GPU_DRAFT_PERSISTENT_BYTES", tensor)
    for block in getattr(model, "blocks", []):
        counters = getattr(getattr(block, "self_attn", None), "runtime_counters", {})
        out["DRAFT_H2D_BYTES"] += int(counters.get("draft_h2d_bytes", 0))
        out["DRAFT_H2D_CALLS"] += int(counters.get("draft_h2d_calls", 0))
    out["GPU_KV_MEASURED_GiB"] = out["GPU_KV_MEASURED_BYTES"] / 2**30
    out["CPU_KV_MEASURED_GiB"] = out["CPU_KV_MEASURED_BYTES"] / 2**30
    out["GPU_DRAFT_PERSISTENT_GiB"] = out["GPU_DRAFT_PERSISTENT_BYTES"] / 2**30
    out["CPU_DRAFT_PERSISTENT_GiB"] = 0.0
    out["KV_COMPRESSION_RATIO"] = 1.0
    return out
