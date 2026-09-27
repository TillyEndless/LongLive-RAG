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
