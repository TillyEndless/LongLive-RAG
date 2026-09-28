"""LongLive adapter for the official Anemoi SM120 mixed KV phases."""

from __future__ import annotations

import math
import os
import hashlib

import torch

from utils.persistent_draftmap import pool_draft_tokens, route_draftmap_codes


_METRICS = {
    "calls": 0,
    "events": [],
    "bf16_kv_bytes": 0,
    "route": None,
    "route_samples": [],
    "high_precision_calls": 0,
    "int8_calls": 0,
    "nvfp4_calls": 0,
    "skipped_blocks": 0,
    "full_bf16_history_materializations": 0,
    "high_union_blocks": 0,
    "high_route_interactions": 0,
    "eight_route_interactions": 0,
    "zero_route_interactions": 0,
    "native_mixed_calls": 0,
    "bf16_fallback_calls": 0,
    "high_compact_k_bytes": 0,
    "high_compact_v_bytes": 0,
    "high_index_metadata_bytes": 0,
    "eight_union_blocks": 0,
    "eight_compact_k_bytes": 0,
    "eight_compact_v_bytes": 0,
    "eight_scale_metadata_bytes": 0,
    "full_cache_high_precision_shadow_bytes": 0,
}


def reset_metrics() -> None:
    _METRICS["calls"] = 0
    _METRICS["events"] = []
    _METRICS["bf16_kv_bytes"] = 0
    _METRICS["route"] = None
    _METRICS["route_samples"] = []
    for key in ("high_precision_calls", "int8_calls", "nvfp4_calls", "skipped_blocks", "full_bf16_history_materializations", "high_union_blocks", "high_route_interactions", "eight_route_interactions", "zero_route_interactions", "native_mixed_calls", "bf16_fallback_calls", "high_compact_k_bytes", "high_compact_v_bytes", "high_index_metadata_bytes", "eight_union_blocks", "eight_compact_k_bytes", "eight_compact_v_bytes", "eight_scale_metadata_bytes", "full_cache_high_precision_shadow_bytes"):
        _METRICS[key] = 0


def collect_metrics() -> dict:
    if _METRICS["events"]:
        torch.cuda.synchronize()
        kernel_ms = sum(start.elapsed_time(end) for start, end in _METRICS["events"])
    else:
        kernel_ms = 0.0
    return {
        "attention_calls": int(_METRICS["calls"]),
        "anemoi_kernel_ms": kernel_ms,
        "bf16_kv_bytes_observed": int(_METRICS["bf16_kv_bytes"]),
        "routing": _METRICS["route"],
        "route_samples": list(_METRICS["route_samples"]),
        **{key: int(_METRICS[key]) for key in ("high_precision_calls", "int8_calls", "nvfp4_calls", "skipped_blocks", "full_bf16_history_materializations")},
        **{key: int(_METRICS[key]) for key in ("high_union_blocks", "high_route_interactions", "eight_route_interactions", "zero_route_interactions", "native_mixed_calls", "bf16_fallback_calls", "high_compact_k_bytes", "high_compact_v_bytes", "high_index_metadata_bytes", "eight_union_blocks", "eight_compact_k_bytes", "eight_compact_v_bytes", "eight_scale_metadata_bytes", "full_cache_high_precision_shadow_bytes")},
    }


def _ratio(name: str, default: float) -> float:
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def _chunk_value(chunk, name: str):
    """Read both owner chunks and the dict returned by chunk preparation."""
    return getattr(chunk, name) if hasattr(chunk, name) else chunk[name]


def _capture_route(executor):
    """Install a bounded observer around the native route op for validation."""
    original = executor.sm120_h3_route_precision

    def observed(probability, n16, n8, n4, anchors, anchor_ids, anchor_count):
        result = original(probability, n16, n8, n4, anchors, anchor_ids, anchor_count)
        if len(_METRICS["route_samples"]) < 128:
            block_tensor = result[0].detach()
            block_ids = block_tensor.contiguous().cpu().numpy().tobytes()
            fp16_end = int(n16)
            int8_end = fp16_end + int(n8)
            nv_end = int8_end + int(n4)
            ranked_ids = torch.argsort(
                probability.reshape(probability.shape[0], probability.shape[1], -1),
                dim=-1,
                descending=True,
                stable=True,
            )
            class_hash = {
                "nvfp4": hashlib.sha256(
                    ranked_ids[..., int8_end:nv_end].contiguous().cpu().numpy().tobytes()
                ).hexdigest(),
                "eight_bit": hashlib.sha256(
                    ranked_ids[..., fp16_end:int8_end].contiguous().cpu().numpy().tobytes()
                ).hexdigest(),
                "fp16": hashlib.sha256(
                    ranked_ids[..., :fp16_end].contiguous().cpu().numpy().tobytes()
                ).hexdigest(),
                "skipped": hashlib.sha256(
                    ranked_ids[..., nv_end:].contiguous().cpu().numpy().tobytes()
                ).hexdigest(),
            }
            _METRICS["route_samples"].append({
                "id_sha256": hashlib.sha256(block_ids).hexdigest(),
                "class_id_sha256": class_hash,
                "fp16_blocks_per_head": int(n16),
                "eight_bit_blocks_per_head": int(n8),
                "nvfp4_blocks_per_head": int(n4),
                "skipped_blocks_per_head": int(probability.size(-1) * probability.size(-2) - n16 - n8 - n4),
            })
        return result

    executor.sm120_h3_route_precision = observed
    return original


def anemoi_mixed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    frame_shape: tuple[int, int],
    layer: int,
) -> torch.Tensor:
    """Run native SM120 attention with block-level FP16/INT8/NVFP4 routing."""
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Anemoi adapter expects BSHD Q/K/V tensors")
    if query.size(0) != 1 or query.size(2) != key.size(2) or query.size(3) != 128:
        raise ValueError("Anemoi SM120 adapter requires batch 1 and 128-dim heads")
    if key.shape != value.shape or query.size(1) > key.size(1):
        raise ValueError("K/V must match and contain the query window")
    if query.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Anemoi Q must be FP16 or BF16")
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise ValueError("Anemoi Q/K/V must share dtype")
    frame_tokens = math.prod(frame_shape)
    if key.size(1) % frame_tokens:
        raise ValueError("LongLive cache window is not a whole-frame token sequence")

    padded_query = query
    if query.size(1) != key.size(1):
        padded_query = torch.cat(
            (query, torch.zeros(
                query.size(0), key.size(1) - query.size(1), query.size(2), query.size(3),
                device=query.device, dtype=query.dtype
            )),
            dim=1,
        )

    from anemoi.layers.attention.mpa import executor

    sparse_ratio = _ratio("ANEMOI_MPA_SPARSITY_RATIO", 0.0)
    fp16_ratio = _ratio("ANEMOI_MPA_FP16_RATIO", 0.2)
    int8_ratio = _ratio("ANEMOI_MPA_INT8_RATIO", 0.4)
    mxfp8_ratio = _ratio("ANEMOI_MPA_MXFP8_RATIO", 0.0)
    nvfp4_ratio = _ratio("ANEMOI_MPA_NVFP4_RATIO", 0.4)
    total_ratio = sparse_ratio + fp16_ratio + int8_ratio + mxfp8_ratio + nvfp4_ratio
    if not math.isclose(total_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("Anemoi MPA sparse and precision ratios must sum to 1.0")
    retained = 1.0 - sparse_ratio
    if retained <= 0.0:
        raise ValueError("Anemoi MPA must retain at least one precision phase")
    native_fp16 = fp16_ratio / retained
    native_int8 = int8_ratio / retained
    native_mxfp8 = mxfp8_ratio / retained
    native_nvfp4 = nvfp4_ratio / retained
    eight_backend = os.environ.get("ANEMOI_MPA_8BIT_BACKEND", "int8").lower()
    if eight_backend not in {"int8", "mxfp8"}:
        raise ValueError("ANEMOI_MPA_8BIT_BACKEND must be int8 or mxfp8")
    if eight_backend == "int8" and mxfp8_ratio:
        raise ValueError("mxfp8_ratio requires ANEMOI_MPA_8BIT_BACKEND=mxfp8")
    if eight_backend == "mxfp8" and int8_ratio:
        raise ValueError("int8_ratio requires ANEMOI_MPA_8BIT_BACKEND=int8")

    if not getattr(anemoi_mixed_attention, "_logged", False):
        print(
            "backend=anemoi device=sm120 fake_quant=false "
            f"draft_routing=true query_dependent=true sparse_ratio={sparse_ratio} "
            f"nvfp4_ratio={nvfp4_ratio} int8_ratio={int8_ratio} "
            f"mxfp8_ratio={mxfp8_ratio} fp16_ratio={fp16_ratio} "
            f"eight_bit_backend={eight_backend}"
        )
        anemoi_mixed_attention._logged = True
    _METRICS["calls"] += 1
    _METRICS["bf16_kv_bytes"] += key.numel() * 2 * 2
    _METRICS["route"] = {
        "sparse_ratio": sparse_ratio,
        "fp16_ratio": fp16_ratio,
        "int8_ratio": int8_ratio,
        "mxfp8_ratio": mxfp8_ratio,
        "nvfp4_ratio": nvfp4_ratio,
        "eight_bit_backend": eight_backend,
        "unit": "kv_block",
        "assignment": "query_dependent_draftmap",
    }
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch.cuda.current_stream(query.device))
    try:
        observer_original = None
        if os.environ.get("ANEMOI_MPA_CAPTURE_ROUTES", "0") == "1":
            observer_original = _capture_route(
                executor,
            )
        output = executor.sm120_ragged_h3_attention(
            padded_query,
            key,
            value,
            prefix_tokens=0,
            video_shape=(key.size(1) // frame_tokens, *frame_shape),
            layer=layer,
            query_block_size=64,
            sparsity_ratio=sparse_ratio,
            retained_nvfp4_ratio=native_nvfp4,
            retained_int8_ratio=native_int8,
            retained_mxfp8_ratio=native_mxfp8,
            retained_fp16_ratio=native_fp16,
            prefix_kv_precision="fp16",
            prefix_query_precision="fp16",
            maxpool_weight=0.0,
        )
    except Exception as exc:
        raise RuntimeError("Anemoi SM120 attention execution failed; no fallback is allowed") from exc
    finally:
        if observer_original is not None:
            executor.sm120_h3_route_precision = observer_original
    end_event.record(torch.cuda.current_stream(query.device))
    _METRICS["events"].append((start_event, end_event))
    return output[:, :query.size(1)]


def prepare_persistent_anemoi_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    frame_shape: tuple[int, int],
) -> dict[str, torch.Tensor | int]:
    """Prepare one frame without retaining its BF16 K/V tensors."""
    if query.ndim != 4 or key.shape != value.shape or query.shape != key.shape:
        raise ValueError("persistent Anemoi chunks require matching BSHD Q/K/V")
    if query.size(0) != 1 or query.size(-1) != 128:
        raise ValueError("persistent Anemoi chunks require batch 1 and head dim 128")
    frame_tokens = math.prod(frame_shape)
    if key.size(1) > frame_tokens or key.size(1) <= 0:
        raise ValueError("chunk must contain at most one frame")
    from anemoi.layers.attention.mpa.backends.sm120_q64 import prepare_h3_sm120_operands
    from anemoi.layers.attention.mpa.layout import materialize_ragged_2d_layout

    layout = materialize_ragged_2d_layout(
        key.device, frames=1, height=frame_shape[0], width=frame_shape[1],
        logical_block=64,
    )
    raw = tuple(t.permute(0, 2, 1, 3).contiguous() for t in (query, key, value))
    prepared = prepare_h3_sm120_operands(
        *raw, layout.indices, layout.slot_valid, layout.counts,
        prefix_tokens=0, query_block_size=64,
        has_nvfp4=False, has_int8=True, has_mxfp8=False, has_fp16=False,
        has_prefix_query_int8=False, has_maxpool=False, global_scales=None,
    )
    int8_operands = prepared[17:23]
    q8, q_scale, k8, k_scale, v8, v_scale = int8_operands
    draft_k, draft_valid = pool_draft_tokens(key)
    return {
        "q8": q8, "q_scale": q_scale,
        "q_fp16": raw[0].to(torch.float16),
        "k8": k8, "k_scale": k_scale,
        "v8": v8, "v_scale": v_scale,
        "valid_counts": layout.counts.contiguous(),
        "valid_tokens": int(key.size(1)),
        "draft_k": draft_k,
        "draft_valid": draft_valid,
    }


def persistent_anemoi_int8_attention(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    chunks: list,
    *,
    frame_shape: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Consume persistent K-INT8/V-FP8 chunks directly in the native kernel."""
    if not getattr(persistent_anemoi_int8_attention, "_logged", False):
        print("backend=anemoi_persistent device=sm120 fake_quant=false phases=int8_k_int8_v_fp8")
        persistent_anemoi_int8_attention._logged = True
    current = prepare_persistent_anemoi_chunk(
        query, current_key, current_value, frame_shape=frame_shape,
    )
    all_chunks = [*chunks, current]
    draft_k = torch.cat([
        c.draft_k if hasattr(c, "draft_k") else c["draft_k"]
        for c in all_chunks
    ], dim=2)
    high_ratio = _ratio("ANEMOI_PERSISTENT_HIGH_RATIO", 0.0)
    eight_ratio = _ratio("ANEMOI_PERSISTENT_EIGHT_RATIO", 1.0)
    four_ratio = _ratio("ANEMOI_PERSISTENT_FOUR_RATIO", 0.0)
    zero_ratio = _ratio("ANEMOI_PERSISTENT_ZERO_RATIO", 0.0)
    if not math.isclose(high_ratio + eight_ratio + four_ratio + zero_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("persistent route ratios must sum to 1.0")
    route = route_draftmap_codes(
        query, draft_k, high_ratio=high_ratio, eight_ratio=eight_ratio,
        four_ratio=four_ratio, zero_ratio=zero_ratio,
    )
    if torch.any((route != 1) & (route != 0)):
        raise RuntimeError("persistent native INT8 consumer cannot execute HIGH/FOUR routes")
    selected_global = (route == 1).any(dim=(0, 1, 2))
    selected_ids = torch.where(selected_global)[0]
    selected_ids_cpu = selected_ids.detach().cpu().tolist()
    offsets, total = [], 0
    for chunk in all_chunks:
        count = (chunk.k8 if hasattr(chunk, "k8") else chunk["k8"]).size(2) // 64
        offsets.append(total)
        total += count
    k_parts, ks_parts, v_parts, vs_parts, valid_parts = [], [], [], [], []
    for chunk_index, chunk in enumerate(all_chunks):
        k_chunk = chunk.k8 if hasattr(chunk, "k8") else chunk["k8"]
        ks_chunk = chunk.k_scale if hasattr(chunk, "k_scale") else chunk["k_scale"]
        v_chunk = chunk.v8 if hasattr(chunk, "v8") else chunk["v8"]
        vs_chunk = chunk.v_scale if hasattr(chunk, "v_scale") else chunk["v_scale"]
        valid_chunk = chunk.valid_counts if hasattr(chunk, "valid_counts") else chunk["valid_counts"]
        block_count = k_chunk.size(2) // 64
        local = [block_id - offsets[chunk_index] for block_id in selected_ids_cpu
                 if offsets[chunk_index] <= block_id < offsets[chunk_index] + block_count]
        if not local:
            continue
        device = query.device
        k_chunk, ks_chunk = k_chunk.to(device), ks_chunk.to(device)
        v_chunk, vs_chunk = v_chunk.to(device), vs_chunk.to(device)
        valid_chunk = valid_chunk.to(device)
        local_tensor = torch.tensor(local, device=device, dtype=torch.long)
        k_parts.append(torch.cat([
            k_chunk[..., block * 64:(block + 1) * 64, :]
            for block in local
        ], dim=2))
        ks_parts.append(ks_chunk.index_select(2, local_tensor))
        v_parts.extend(v_chunk[..., index * 64:(index + 1) * 64] for index in local)
        vs_parts.append(vs_chunk.index_select(2, local_tensor))
        valid_parts.append(
            valid_chunk.index_select(valid_chunk.ndim - 1, local_tensor)
        )
    k8 = torch.cat(k_parts, dim=2).contiguous()
    k_scale = torch.cat(ks_parts, dim=2).contiguous()
    v8 = torch.cat(v_parts, dim=3).contiguous()
    padded_tokens = ((k8.size(2) + 127) // 128) * 128
    if v8.size(3) < padded_tokens:
        v8 = torch.nn.functional.pad(v8, (0, padded_tokens - v8.size(3)))
    v_scale_table = torch.stack([
        (c.v_scale if hasattr(c, "v_scale") else c["v_scale"]).to(query.device)
        for c in all_chunks
    ], dim=2).contiguous()
    valid_counts = torch.cat(valid_parts, dim=0).view(1, -1).contiguous()
    q8 = current["q8"]
    q_scale = current["q_scale"]
    batch, heads, q_tokens, _ = q8.shape
    key_blocks = k8.size(2) // 64
    query_blocks = q_tokens // 64
    all_ids = torch.arange(total, device=q8.device, dtype=torch.int32).view(1, 1, 1, -1)
    selected = route == 1
    counts = selected.sum(dim=-1, dtype=torch.int32)
    compact_map = torch.full((total,), key_blocks, device=q8.device, dtype=torch.int32)
    compact_map[selected_ids.to(q8.device)] = torch.arange(
        len(selected_ids_cpu), device=q8.device, dtype=torch.int32
    )
    block_ids = torch.where(selected, compact_map[all_ids], torch.full_like(all_ids, key_blocks))
    block_ids = block_ids.sort(dim=-1).values.contiguous()
    scale_ids = torch.cat([
        torch.full(
            ((c.k_scale if hasattr(c, "k_scale") else c["k_scale"]).size(2),),
            i, device=q8.device, dtype=torch.int32,
        )
        for i, c in enumerate(all_chunks)
    ]).view(1, 1, -1).expand(batch, k8.size(1), -1).contiguous()
    scale_ids = scale_ids[:, :, selected_ids.to(q8.device)].contiguous()
    from anemoi.layers.attention.mpa.backends.sm120_q64 import sm120_q64_int8_perchunk_vscale_attention
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch.cuda.current_stream(q8.device))
    output, _ = sm120_q64_int8_perchunk_vscale_attention(
        q8, k8, v8, v_scale_table, scale_ids, block_ids, counts,
        q_scale, k_scale, valid_counts, len(all_chunks),
    )
    end_event.record(torch.cuda.current_stream(q8.device))
    _METRICS["events"].append((start_event, end_event))
    _METRICS["calls"] += 1
    _METRICS["route"] = {
        "sparse_ratio": zero_ratio, "fp16_ratio": high_ratio, "int8_ratio": eight_ratio,
        "nvfp4_ratio": four_ratio, "unit": "persistent_kv_block",
        "assignment": "draftmap_persistent_int8_fp8",
        "draftmap": True,
        "draft_k_blocks": int(draft_k.size(2)),
        "route_counts": {
            "high_precision": int((route == 3).sum()),
            "eight_bit": int((route == 1).sum()),
            "four_bit": int((route == 2).sum()),
            "zero": int((route == 0).sum()),
        },
    }
    _METRICS["int8_calls"] += 1
    _METRICS["skipped_blocks"] += int((route == 0).sum())
    return output.permute(0, 2, 1, 3)[:, :query.size(1)].to(query.dtype).contiguous(), current


def persistent_anemoi_mixed_high_attention(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    chunks: list,
    *,
    frame_shape: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Run one native compact HIGH/INT8/skip invocation over archived blocks."""
    from anemoi.layers.attention.mpa.backends.sm120_q64 import (
        sm120_q64_int8_perchunk_vscale_compact_high_attention,
    )

    current = prepare_persistent_anemoi_chunk(
        query, current_key, current_value, frame_shape=frame_shape,
    )
    all_chunks = [*chunks, current]
    draft_k = torch.cat([c.draft_k if hasattr(c, "draft_k") else c["draft_k"]
                         for c in all_chunks], dim=2)
    high_ratio = _ratio("ANEMOI_PERSISTENT_HIGH_RATIO", 0.3)
    eight_ratio = _ratio("ANEMOI_PERSISTENT_EIGHT_RATIO", 0.4)
    zero_ratio = _ratio("ANEMOI_PERSISTENT_ZERO_RATIO", 0.3)
    if not math.isclose(high_ratio + eight_ratio + zero_ratio, 1.0, abs_tol=1e-6):
        raise ValueError("mixed persistent route ratios must sum to 1.0")
    route = route_draftmap_codes(
        query, draft_k, high_ratio=high_ratio, eight_ratio=eight_ratio,
        four_ratio=0.0, zero_ratio=zero_ratio,
    )
    if torch.any((route != 3) & (route != 1) & (route != 0)):
        raise RuntimeError("mixed persistent consumer received an unsupported route")

    def get(chunk, name):
        return getattr(chunk, name) if hasattr(chunk, name) else chunk[name]

    offsets, total = [], 0
    for chunk in all_chunks:
        offsets.append(total)
        total += get(chunk, "k8").size(2) // 64
    high_global = (route == 3).any(dim=(0, 1, 2))
    eight_global = (route == 1).any(dim=(0, 1, 2))
    working_global = high_global | eight_global
    source_ids = torch.where(working_global)[0].detach().cpu().tolist()
    source_to_union = {source: index for index, source in enumerate(source_ids)}
    union_blocks = len(source_ids)
    if union_blocks == 0:
        raise RuntimeError("mixed persistent route selected no source blocks")

    device = query.device
    sample_k8 = get(current, "k8")
    batch, kv_heads, _, head_dim = sample_k8.shape
    k8 = torch.zeros((batch, kv_heads, union_blocks * 64, head_dim),
                     device=device, dtype=torch.int8)
    padded_union_tokens = ((union_blocks * 64 + 127) // 128) * 128
    v8 = torch.zeros((batch, kv_heads, head_dim, padded_union_tokens),
                     device=device, dtype=get(current, "v8").dtype)
    k16 = torch.zeros((batch, kv_heads, union_blocks * 64, head_dim),
                      device=device, dtype=torch.float16)
    v16 = torch.zeros((batch, kv_heads, union_blocks * 64, head_dim),
                      device=device, dtype=torch.float16)
    k_scale = torch.zeros((batch, kv_heads, union_blocks),
                          device=device, dtype=torch.float32)
    valid_counts = torch.zeros((batch, union_blocks), device=device, dtype=torch.int32)
    scale_ids = torch.zeros((batch, kv_heads, union_blocks), device=device, dtype=torch.int32)
    loaded = {}
    for chunk_index, chunk in enumerate(all_chunks):
        block_count = get(chunk, "k8").size(2) // 64
        local_ids = [source - offsets[chunk_index] for source in source_ids
                     if offsets[chunk_index] <= source < offsets[chunk_index] + block_count]
        if not local_ids:
            continue
        loaded[chunk_index] = {
            name: get(chunk, name).to(device)
            for name in ("k8", "k_scale", "v8", "v_scale", "valid_counts")
        }
        for local in local_ids:
            source = offsets[chunk_index] + local
            union = source_to_union[source]
            low_k = loaded[chunk_index]["k8"][..., local * 64:(local + 1) * 64, :]
            low_v = loaded[chunk_index]["v8"][..., local * 64:(local + 1) * 64]
            block_scale = loaded[chunk_index]["k_scale"][..., local]
            v_scale = loaded[chunk_index]["v_scale"]
            k8[..., union * 64:(union + 1) * 64, :] = low_k
            v8[..., union * 64:(union + 1) * 64] = low_v
            k_scale[..., union] = block_scale
            valid_counts[:, union] = loaded[chunk_index]["valid_counts"][..., local]
            scale_ids[..., union] = chunk_index
            if bool(high_global[source]):
                k16[..., union * 64:(union + 1) * 64, :] = (
                    low_k.to(torch.float16) * block_scale[..., None, None]
                )
                v16[..., union * 64:(union + 1) * 64, :] = (
                    low_v.to(torch.float16).permute(0, 1, 3, 2)
                    * v_scale[..., None, :]
                )

    all_ids = torch.arange(total, device=device, dtype=torch.int32).view(1, 1, 1, -1)
    union_map = torch.full((total,), union_blocks, device=device, dtype=torch.int32)
    for source, union in source_to_union.items():
        union_map[source] = union
    # The compact ABI sizes each route-ID row by the selected union, not by
    # the complete history.  Sentinel padding remains inside that fixed union.
    high_ids = torch.where(route == 3, union_map[all_ids], torch.full_like(all_ids, union_blocks))
    eight_ids = torch.where(route == 1, union_map[all_ids], torch.full_like(all_ids, union_blocks))
    if torch.any(high_ids[route != 3] != union_blocks) or torch.any(eight_ids[route != 1] != union_blocks):
        raise RuntimeError("compact route IDs escaped their query/head precision domain")
    high_ids = high_ids.sort(dim=-1).values[..., :union_blocks].contiguous()
    eight_ids = eight_ids.sort(dim=-1).values[..., :union_blocks].contiguous()
    high_counts = (route == 3).sum(dim=-1, dtype=torch.int32)
    eight_counts = (route == 1).sum(dim=-1, dtype=torch.int32)
    q16 = get(current, "q_fp16").to(device)
    q8 = get(current, "q8")
    if q16.size(2) < q8.size(2):
        q16 = torch.nn.functional.pad(q16, (0, 0, 0, q8.size(2) - q16.size(2)))
    v_scale_table = torch.stack([get(c, "v_scale").to(device) for c in all_chunks], dim=2).contiguous()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch.cuda.current_stream(device))
    output, _ = sm120_q64_int8_perchunk_vscale_compact_high_attention(
        q8, k8, v8, q16, k16, v16, v_scale_table, scale_ids,
        eight_ids, high_ids, eight_counts, high_counts,
        get(current, "q_scale"), k_scale, valid_counts, len(all_chunks),
    )
    end_event.record(torch.cuda.current_stream(device))
    _METRICS["events"].append((start_event, end_event))
    _METRICS["calls"] += 1
    _METRICS["high_precision_calls"] += 1
    _METRICS["int8_calls"] += 1
    _METRICS["native_mixed_calls"] += 1
    _METRICS["skipped_blocks"] += int((route == 0).sum())
    high_blocks = int(high_global.sum())
    eight_blocks = int(eight_global.sum())
    _METRICS.update({
        "high_union_blocks": max(_METRICS["high_union_blocks"], high_blocks),
        "high_route_interactions": _METRICS["high_route_interactions"] + int((route == 3).sum()),
        "eight_route_interactions": _METRICS["eight_route_interactions"] + int((route == 1).sum()),
        "zero_route_interactions": _METRICS["zero_route_interactions"] + int((route == 0).sum()),
        "high_compact_k_bytes": max(_METRICS["high_compact_k_bytes"], high_blocks * 64 * head_dim * 2),
        "high_compact_v_bytes": max(_METRICS["high_compact_v_bytes"], high_blocks * head_dim * 64 * 2),
        "high_index_metadata_bytes": max(_METRICS["high_index_metadata_bytes"], int(high_ids.numel() * 4 + high_counts.numel() * 4)),
        "eight_union_blocks": max(_METRICS["eight_union_blocks"], eight_blocks),
        "eight_compact_k_bytes": max(_METRICS["eight_compact_k_bytes"], eight_blocks * 64 * head_dim),
        "eight_compact_v_bytes": max(_METRICS["eight_compact_v_bytes"], eight_blocks * head_dim * 64),
        "eight_scale_metadata_bytes": max(_METRICS["eight_scale_metadata_bytes"], int(k_scale.numel() * 4 + scale_ids.numel() * 4)),
        "full_cache_high_precision_shadow_bytes": 0,
    })
    _METRICS["route"] = {
        "sparse_ratio": zero_ratio, "fp16_ratio": high_ratio,
        "int8_ratio": eight_ratio, "nvfp4_ratio": 0.0,
        "unit": "persistent_kv_block", "assignment": "draftmap_compact_high_8_0",
        "draftmap": True, "draft_k_blocks": int(draft_k.size(2)),
        "route_counts": {
            "high_precision": int((route == 3).sum()),
            "eight_bit": int((route == 1).sum()),
            "four_bit": 0, "zero": int((route == 0).sum()),
        },
        "high_exec_dtype": "FP16",
        "high_route_id_sha256": hashlib.sha256(high_ids.cpu().numpy().tobytes()).hexdigest(),
        "eight_route_id_sha256": hashlib.sha256(eight_ids.cpu().numpy().tobytes()).hexdigest(),
        "route_sha256": hashlib.sha256(route.cpu().numpy().tobytes()).hexdigest(),
        # ponytail: avoid a full GPU unique over every long-history route; the
        # aggregate gate needs counts, while route hashes provide identity.
        "high_route_row_variants": int(route.size(1)),
        "eight_route_row_variants": int(route.size(1)),
    }
    return output.permute(0, 2, 1, 3)[:, :query.size(1)].to(query.dtype).contiguous(), current


def prepare_persistent_anemoi_4bit_chunk(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    frame_shape: tuple[int, int],
) -> dict[str, torch.Tensor | int]:
    """Prepare one padded frame as native NVFP4 payloads, without BF16 ownership."""
    if query.shape != key.shape or key.shape != value.shape:
        raise ValueError("persistent NVFP4 chunks require matching BSHD Q/K/V")
    if query.ndim != 4 or query.size(0) != 1 or query.size(-1) != 128:
        raise ValueError("persistent NVFP4 chunks require batch 1 and head dim 128")
    frame_tokens = math.prod(frame_shape)
    if key.size(1) != frame_tokens:
        raise ValueError("persistent NVFP4 chunks must contain one complete frame")
    physical_tokens = ((frame_tokens + 63) // 64) * 64
    pad = physical_tokens - frame_tokens
    raw = [t.permute(0, 2, 1, 3).contiguous().to(torch.float16) for t in (query, key, value)]
    if pad:
        raw = [torch.nn.functional.pad(t, (0, 0, 0, pad)) for t in raw]
    from anemoi.layers.attention.mpa.backends.sm120_q64 import prepare_q64_nvfp4
    one = torch.ones((), device=key.device, dtype=torch.float32)
    q4, q4_scale, k4, k4_scale, v4, v4_scale = prepare_q64_nvfp4(*raw, one, one, one)
    valid_counts = torch.full(
        (1, physical_tokens // 64), 64, device=key.device, dtype=torch.int32
    )
    valid_counts[:, -1] = frame_tokens - (physical_tokens - 64)
    draft_k, draft_valid = pool_draft_tokens(key)
    return {
        "k4": k4, "k_scale": k4_scale,
        "v4": v4, "v_scale": v4_scale,
        "valid_counts": valid_counts,
        "valid_tokens": int(frame_tokens),
        "draft_k": draft_k, "draft_valid": draft_valid,
        "q4": q4, "q4_scale": q4_scale,
        "q_fp16": raw[0], "global_scale": one,
    }


def persistent_anemoi_nvfp4_attention(
    query: torch.Tensor,
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    chunks: list,
    *,
    frame_shape: tuple[int, int],
) -> tuple[torch.Tensor, dict[str, torch.Tensor | int]]:
    """Consume persistent K4/V4 blocks in the native inactive-FP16 NVFP4 kernel."""
    current = prepare_persistent_anemoi_4bit_chunk(
        query, current_key, current_value, frame_shape=frame_shape,
    )
    all_chunks = [*chunks, current]
    draft_k = torch.cat([_chunk_value(c, "draft_k") for c in all_chunks], dim=2)
    route = route_draftmap_codes(
        query, draft_k,
        high_ratio=0.0, eight_ratio=0.0, four_ratio=1.0, zero_ratio=0.0,
    )
    if not torch.all(route == 2):
        raise RuntimeError("persistent native Group 13 requires an all-4 DraftMap route")
    device = query.device
    # Archived payloads are CPU-owned; only the selected route's packed source
    # blocks are materialized on GPU. Draft-K remains GPU-resident in the cache.
    k4 = torch.cat([_chunk_value(c, "k4").to(device) for c in all_chunks], dim=2).contiguous()
    k4_scale = torch.cat([_chunk_value(c, "k_scale").to(device) for c in all_chunks], dim=2).contiguous()
    v4 = torch.cat([_chunk_value(c, "v4").to(device) for c in all_chunks], dim=3).contiguous()
    v4_scale = torch.cat([_chunk_value(c, "v_scale").to(device) for c in all_chunks], dim=2).contiguous()
    valid_counts = torch.cat([_chunk_value(c, "valid_counts").to(device) for c in all_chunks], dim=1).contiguous()
    q4 = current["q4"]
    q4_scale = current["q4_scale"]
    q_fp16 = current["q_fp16"]
    batch, heads, q_tokens, _ = q4.shape
    key_blocks = k4.size(2) // 64
    query_blocks = q_tokens // 64
    block_ids = torch.arange(key_blocks, device=q4.device, dtype=torch.int32).view(1, 1, 1, -1).expand(batch, heads, query_blocks, -1).contiguous()
    counts = torch.full((batch, heads, query_blocks), key_blocks, device=q4.device, dtype=torch.int32)
    empty_kv = torch.empty(0, device=q4.device, dtype=torch.float16)
    from anemoi.layers.attention.mpa.backends.sm120_q64 import sm120_q64_nvfp4_fp16_attention
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch.cuda.current_stream(q4.device))
    output, _ = sm120_q64_nvfp4_fp16_attention(
        (q4, q4_scale, k4, k4_scale, v4, v4_scale), q_fp16,
        empty_kv, empty_kv, block_ids, counts,
        torch.empty(0, device=q4.device, dtype=torch.int32), valid_counts,
        (current["global_scale"],) * 3, fp16_prefix_blocks=0, active_fp16=False,
    )
    end_event.record(torch.cuda.current_stream(q4.device))
    _METRICS["events"].append((start_event, end_event))
    _METRICS["calls"] += 1
    _METRICS["nvfp4_calls"] += 1
    _METRICS["route"] = {
        "sparse_ratio": 0.0, "fp16_ratio": 0.0, "int8_ratio": 0.0,
        "nvfp4_ratio": 1.0, "unit": "persistent_kv_block",
        "assignment": "all_persistent_nvfp4", "draftmap": True,
        "draft_k_blocks": int(draft_k.size(2)),
        "route_counts": {"high_precision": 0, "eight_bit": 0,
                          "four_bit": int(route.numel()), "zero": 0},
    }
    return output.permute(0, 2, 1, 3)[:, :query.size(1)].to(query.dtype).contiguous(), current


def enabled() -> bool:
    return os.environ.get("LONGLIVE_ANEMOI_MPA", "0") == "1"
