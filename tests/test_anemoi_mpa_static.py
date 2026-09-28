import sys
import types

import pytest
import torch

import utils.anemoi_mpa as mpa


class _FakeEvent:
    def __init__(self, *args, **kwargs):
        pass

    def record(self, *args, **kwargs):
        pass

    def elapsed_time(self, other):
        return 0.0


def _install_backend(monkeypatch, **functions):
    backend = types.ModuleType("anemoi.layers.attention.mpa.backends.sm120_q64")
    for name, function in functions.items():
        setattr(backend, name, function)
    names = [
        "anemoi",
        "anemoi.layers",
        "anemoi.layers.attention",
        "anemoi.layers.attention.mpa",
        "anemoi.layers.attention.mpa.backends",
        "anemoi.layers.attention.mpa.backends.sm120_q64",
    ]
    modules = {name: types.ModuleType(name) for name in names[:-1]}
    modules["anemoi.layers.attention.mpa.backends"].sm120_q64 = backend
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, names[-1], backend)


def _fake_events(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args, **kwargs: None)


def _prepared_int8():
    return {
        "q8": torch.zeros((1, 1, 64, 128), dtype=torch.int8),
        "q_scale": torch.ones((1, 1, 1), dtype=torch.float32),
        "q_fp16": torch.zeros((1, 1, 64, 128), dtype=torch.float16),
        "k8": torch.zeros((1, 1, 128, 128), dtype=torch.int8),
        "k_scale": torch.ones((1, 1, 2), dtype=torch.float32),
        "v8": torch.zeros((1, 1, 128, 128), dtype=torch.float8_e4m3fn),
        "v_scale": torch.ones((1, 1, 128), dtype=torch.float32),
        "valid_counts": torch.full((1, 2), 64, dtype=torch.int32),
        "draft_k": torch.zeros((1, 1, 2, 128), dtype=torch.bfloat16),
        "draft_valid": torch.full((2,), 64, dtype=torch.int32),
    }


def test_group16_compact_high_calls_only_compact_native(monkeypatch):
    _fake_events(monkeypatch)
    calls = []

    def compact(*args):
        calls.append(args)
        return torch.zeros((1, 1, 64, 128), dtype=torch.float16), torch.empty(0)

    def forbidden(*args):
        raise AssertionError("Group16 must not call the ordinary INT8 kernel")

    _install_backend(
        monkeypatch,
        sm120_q64_int8_perchunk_vscale_compact_high_attention=compact,
        sm120_q64_int8_perchunk_vscale_attention=forbidden,
    )
    prepared = _prepared_int8()
    monkeypatch.setattr(mpa, "prepare_persistent_anemoi_chunk", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(
        mpa, "route_draftmap_codes",
        lambda *args, **kwargs: torch.tensor([[[[3, 1]]]], dtype=torch.int64),
    )
    query = torch.zeros((1, 64, 1, 128), dtype=torch.bfloat16)
    output, _ = mpa.persistent_anemoi_mixed_high_attention(
        query, query, query, [], frame_shape=(8, 8),
    )
    assert output.shape == query.shape
    assert len(calls) == 1
    assert calls[0][0] is prepared["q8"]
    assert calls[0][1].shape == prepared["k8"].shape
    assert calls[0][12] is prepared["q_scale"]


def test_group18_int8_calls_perchunk_and_excludes_zero_block(monkeypatch):
    _fake_events(monkeypatch)
    calls = []

    def perchunk(*args):
        calls.append(args)
        return torch.zeros((1, 1, 64, 128), dtype=torch.float16), torch.empty(0)

    _install_backend(monkeypatch, sm120_q64_int8_perchunk_vscale_attention=perchunk)
    prepared = _prepared_int8()
    monkeypatch.setattr(mpa, "prepare_persistent_anemoi_chunk", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(
        mpa, "route_draftmap_codes",
        lambda *args, **kwargs: torch.tensor([[[[1, 0]]]], dtype=torch.int64),
    )
    query = torch.zeros((1, 64, 1, 128), dtype=torch.bfloat16)
    output, _ = mpa.persistent_anemoi_int8_attention(
        query, query, query, [], frame_shape=(8, 8),
    )
    assert output.shape == query.shape
    assert len(calls) == 1
    block_ids = calls[0][5]
    counts = calls[0][6]
    assert block_ids.flatten().tolist() == [0, 1]
    assert counts.item() == 1


def test_missing_native_symbol_fails_loudly(monkeypatch):
    _install_backend(monkeypatch)
    prepared = _prepared_int8()
    monkeypatch.setattr(mpa, "prepare_persistent_anemoi_chunk", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(
        mpa, "route_draftmap_codes",
        lambda *args, **kwargs: torch.tensor([[[[1, 0]]]], dtype=torch.int64),
    )
    query = torch.zeros((1, 64, 1, 128), dtype=torch.bfloat16)
    with pytest.raises(ImportError):
        mpa.persistent_anemoi_int8_attention(query, query, query, [], frame_shape=(8, 8))
