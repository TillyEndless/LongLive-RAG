"""CPU-only ownership and quantization tests for Group12-15-v2."""
import torch

from utils.local_lowbit_kv import LocalLowbitKVStore
from utils.compressed_history_archive import CompressedHistoryArchive


def test_local_owner(mode):
    store = LocalLowbitKVStore(3, 4, 2, 4, mode, torch.device("cpu"))
    k = torch.randn(1, 4, 2, 4, dtype=torch.bfloat16)
    v = torch.randn_like(k)
    store.insert_frames(0, k, v)
    dk, dv = store.materialize()
    assert dk.shape == k.shape and dv.shape == v.shape
    assert dk.dtype == torch.bfloat16 and dv.dtype == torch.bfloat16
    assert store.persistent_bytes() > 0
    selected = store.select_top([1.0, 0.0, 0.0], 1.0)
    assert selected == [0]
    # Promotion is a transient CPU-BF16 overlay; the persistent owner stays
    # low-bit and therefore no promoted BF16 bytes are retained in the store.
    assert store.memory_breakdown()["promoted_k"] == 0


def test_cpu_history_only():
    k = torch.randn(1, 4, 2, 4, dtype=torch.bfloat16)
    archive = CompressedHistoryArchive("int8_fp8")
    rec = archive.append(k, k, layer_id=0, chunk_id=0, history_id=0, valid_tokens=4)
    out_k, out_v = archive.fetch(rec, torch.device("cpu"))
    assert out_k.dtype == torch.bfloat16 and out_v.dtype == torch.bfloat16
    assert not hasattr(rec, "gpu_k_payload") and not hasattr(rec, "gpu_v_payload")
    assert rec.gpu_persistent_bytes() == 0


if __name__ == "__main__":
    test_local_owner("int8_fp8")
    test_local_owner("nvfp4")
    test_cpu_history_only()
    print("GROUP12_15_V2_STATIC_TESTS=PASS")
