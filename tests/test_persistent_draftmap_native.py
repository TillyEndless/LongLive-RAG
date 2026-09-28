import torch

from utils.persistent_draftmap import pool_draft_tokens, route_draftmap, route_draftmap_codes
from utils.persistent_anemoi_8bit_cache import PersistentAnemoi8BitCache
from utils.runtime_memory_measurement import runtime_inference_memory


def test_common_route_draftmap_api_remains_unchanged():
    query = torch.ones((1, 64, 1, 2), dtype=torch.bfloat16)
    key = torch.ones((1, 128, 1, 2), dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    retained_k, retained_v, metadata = route_draftmap(query, key, value, 0.5)
    assert retained_k.shape == (1, 64, 1, 2)
    assert retained_v.shape == retained_k.shape
    assert metadata["SPARSE_EXECUTION_STATUS"] == "REAL_SPARSE_EXECUTION"


def test_pool_and_route_are_interaction_level_and_cache_keeps_draft_k_aligned():
    key = torch.arange(1 * 130 * 1 * 2, dtype=torch.bfloat16).reshape(1, 130, 1, 2)
    pooled, valid = pool_draft_tokens(key, 64)
    assert pooled.shape == (1, 1, 3, 2)
    assert valid.tolist() == [64, 64, 2]

    query = key[:, :64]
    route = route_draftmap_codes(query, pooled, eight_ratio=1.0)
    assert route.shape == (1, 1, 1, 3)
    assert torch.all(route == 1)
    sparse_route = route_draftmap_codes(query, pooled, eight_ratio=2 / 3, zero_ratio=1 / 3)
    assert int((sparse_route == 1).sum()) == 2
    assert int((sparse_route == 0).sum()) == 1

    cache = PersistentAnemoi8BitCache(max_chunks=1)
    cache.append(
        torch.zeros((1, 1, 192, 2), dtype=torch.int8),
        torch.ones((1, 1, 3), dtype=torch.float32),
        torch.zeros((1, 1, 2, 192), dtype=torch.float8_e4m3fn),
        torch.ones((1, 1, 2), dtype=torch.float32),
        draft_k=pooled,
        draft_valid=valid,
        valid_tokens=130,
    )
    assert cache.chunks[0].draft_k.shape[2] == 3
    assert cache.persistent_draft_bytes == pooled.numel() * pooled.element_size() + valid.numel() * valid.element_size()


def test_route_zero_quota_is_reserved_from_eight_bit_quota():
    query = torch.ones((1, 64, 1, 2), dtype=torch.bfloat16)
    draft = torch.ones((1, 1, 4, 2), dtype=torch.bfloat16)
    route = route_draftmap_codes(query, draft, eight_ratio=0.75, zero_ratio=0.25)
    assert int((route == 1).sum()) == 3
    assert int((route == 0).sum()) == 1


def test_native_cache_archives_evicted_kv_but_keeps_draft_history_ids():
    cache = PersistentAnemoi8BitCache(max_chunks=1)
    draft_device = "cuda" if torch.cuda.is_available() else "cpu"
    draft = torch.zeros((1, 1, 1, 2), dtype=torch.bfloat16, device=draft_device)
    valid = torch.ones((1,), dtype=torch.int32, device=draft_device)

    def append_chunk(value):
        cache.append(
            torch.full((1, 1, 64, 2), value, dtype=torch.int8),
            torch.ones((1, 1, 1), dtype=torch.float32),
            torch.zeros((1, 1, 2, 64), dtype=torch.float8_e4m3fn),
            torch.ones((1, 1, 1), dtype=torch.float32),
            valid_tokens=64,
            draft_k=draft.clone(),
            draft_valid=valid.clone(),
        )

    append_chunk(1)
    append_chunk(2)

    assert cache.chunk_ids == [1]
    assert cache.archived_chunk_ids == [0]
    assert [c.chunk_id for c in cache.history_chunks] == [0, 1]
    assert cache.history_chunks[0].k8.device.type == "cpu"
    assert cache.history_chunks[0].draft_k.device == draft.device
    expected_draft_bytes = cache.persistent_draft_bytes if draft_device == "cpu" else cache.persistent_gpu_draft_bytes
    assert cache.persistent_gpu_draft_bytes == expected_draft_bytes
    assert cache.persistent_cpu_kv_bytes > 0


def test_runtime_memory_schema_preserves_h200_prefetch_fields_for_native_owner():
    cache = PersistentAnemoi8BitCache(max_chunks=1)
    args = dict(
        k8=torch.zeros((1, 1, 64, 2), dtype=torch.int8),
        k_scale=torch.ones((1, 1, 1), dtype=torch.float32),
        v8=torch.zeros((1, 1, 2, 64), dtype=torch.float8_e4m3fn),
        v_scale=torch.ones((1, 1, 1), dtype=torch.float32),
        valid_tokens=64,
        draft_k=torch.zeros((1, 1, 1, 2), dtype=torch.bfloat16),
        draft_valid=torch.ones((1,), dtype=torch.int32),
    )
    cache.append(**args)
    memory = runtime_inference_memory(
        [{"anemoi_8bit_cache": cache,
          "global_end_index": torch.zeros((1,), dtype=torch.long),
          "local_end_index": torch.zeros((1,), dtype=torch.long),
          "prefetch_buffer_peak_bytes": 0}],
        type("Model", (), {"blocks": []})(),
        "NATIVE_PACKED",
    )
    for key in (
        "GPU_KV_MEASURED_BYTES", "GPU_KV_ACTUAL_PERSISTENT_BYTES",
        "TRANSIENT_DEQUANT_GPU_BYTES", "TRANSIENT_PREFETCH_GPU_BYTES",
        "PREFETCH_REQUESTED_CHUNKS", "PREFETCH_HIT_CHUNKS",
        "PREFETCH_CORRECTION_CHUNKS", "PREFETCH_WASTED_CHUNKS",
        "NATIVE_HISTORY_CHUNKS", "NATIVE_GPU_PACKED_KV_BYTES",
    ):
        assert key in memory
    assert memory["NATIVE_HISTORY_CHUNKS"] == 1
