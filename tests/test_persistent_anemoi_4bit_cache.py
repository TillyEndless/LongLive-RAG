import pytest
import torch

from utils.anemoi_mpa import _chunk_value
from utils.persistent_anemoi_4bit_cache import (
    Anemoi4BitChunk,
    PersistentAnemoi4BitCache,
)


def test_four_bit_cache_owns_only_packed_operands_and_metadata():
    if not torch.cuda.is_available():
        pytest.skip("persistent cache is GPU-owned")
    device = torch.device("cuda")
    cache = PersistentAnemoi4BitCache(max_chunks=2)
    chunk = Anemoi4BitChunk(
        chunk_id=0,
        k4=torch.zeros((1, 12, 64, 64), dtype=torch.uint8, device=device),
        k_scale=torch.zeros((1, 12, 64, 8), dtype=torch.uint8, device=device),
        v4=torch.zeros((1, 12, 128, 32), dtype=torch.uint8, device=device),
        v_scale=torch.zeros((1, 12, 64, 512), dtype=torch.uint8, device=device),
        valid_counts=torch.full((1, 1), 64, dtype=torch.int32, device=device),
        valid_tokens=64,
        start_token=0,
        draft_k=torch.zeros((1, 12, 1, 128), dtype=torch.bfloat16, device=device),
        draft_valid=torch.ones((1,), dtype=torch.int32, device=device),
    )
    cache.append(chunk)
    assert cache.persistent_bf16_bytes == 0
    assert cache.chunk_count == 1
    assert cache.persistent_bytes == sum(
        t.numel() * t.element_size()
        for t in (chunk.k4, chunk.k_scale, chunk.v4, chunk.v_scale,
                  chunk.valid_counts, chunk.draft_k, chunk.draft_valid)
    )


def test_nvfp4_adapter_accepts_owner_chunk_and_prepared_dict_abi():
    if not torch.cuda.is_available():
        pytest.skip("persistent cache is GPU-owned")
    device = torch.device("cuda")
    cache = PersistentAnemoi4BitCache(max_chunks=1)
    chunk = Anemoi4BitChunk(
        chunk_id=0,
        k4=torch.zeros((1, 12, 64, 64), dtype=torch.uint8, device=device),
        k_scale=torch.zeros((1, 12, 64, 8), dtype=torch.uint8, device=device),
        v4=torch.zeros((1, 12, 128, 32), dtype=torch.uint8, device=device),
        v_scale=torch.zeros((1, 12, 64, 512), dtype=torch.uint8, device=device),
        valid_counts=torch.full((1, 1), 64, dtype=torch.int32, device=device),
        valid_tokens=64,
        start_token=0,
        draft_k=torch.zeros((1, 12, 1, 128), dtype=torch.bfloat16, device=device),
        draft_valid=torch.ones((1,), dtype=torch.int32, device=device),
    )
    cache.append(chunk)
    prepared = {name: getattr(chunk, name) for name in (
        "k4", "k_scale", "v4", "v_scale", "valid_counts", "draft_k", "draft_valid"
    )}
    for name in prepared:
        assert _chunk_value(cache.history_chunks[0], name) is getattr(chunk, name)
        assert _chunk_value(prepared, name) is prepared[name]
