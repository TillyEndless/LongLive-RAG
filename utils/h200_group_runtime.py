"""Portable H200 runtime hooks for corrected Groups 12--15.

Only KV storage is transformed. Queries and the final attention call remain
BF16; this module never imports or dispatches a native low-bit kernel.
"""

from __future__ import annotations

import math
import torch

from utils.persistent_kv_storage import PersistentHistoryQuantizer
from utils.quant import quantize_kv
from fouroversix.quantize import QuantizationConfig


def fake_quantize_kv(k: torch.Tensor, v: torch.Tensor, mode: str):
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("fake-quant input must be BF16")
    if mode == "group12_fake8":
        quantizer = PersistentHistoryQuantizer()
        qk, sk = quantizer.quantize_k(k)
        qv, sv = quantizer.quantize_v(v)
        return quantizer.dequantize_k(qk, sk), quantizer.dequantize_v(qv, sv), {
            "K_FAKE_QUANT_ACTIVE": "YES",
            "V_FAKE_QUANT_ACTIVE": "YES",
            "K_STORAGE_DTYPE": "int8",
            "V_STORAGE_DTYPE": "float8_e4m3fn",
            "PERSISTENT_STORAGE_MODE": "BF16_FAKE_QUANT",
            "KV_COMPRESSION_RATIO": 1.0,
        }
    if mode == "group13_fake4":
        cfg = QuantizationConfig()
        def one(x):
            shape = x.shape
            packed = quantize_kv(x.reshape(-1, shape[-1]), cfg)
            return packed.dequantize(dtype=torch.bfloat16).reshape(shape)
        return one(k), one(v), {
            "K_FAKE_QUANT_ACTIVE": "YES",
            "V_FAKE_QUANT_ACTIVE": "YES",
            "K_STORAGE_DTYPE": "nvfp4_e2m1",
            "V_STORAGE_DTYPE": "nvfp4_e2m1",
            "PERSISTENT_STORAGE_MODE": "BF16_FAKE_QUANT",
            "KV_COMPRESSION_RATIO": 1.0,
        }
    return k, v, {
        "K_FAKE_QUANT_ACTIVE": "NO",
        "V_FAKE_QUANT_ACTIVE": "NO",
        "PERSISTENT_STORAGE_MODE": "BF16_FAKE_QUANT",
        "KV_COMPRESSION_RATIO": 1.0,
    }


def sparse_retain(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, ratio: float):
    """Keep top-scoring 64-token K/V blocks; omitted blocks are not attended."""
    if not 0.0 < ratio < 1.0:
        return k, v, {"SPARSE_ROUTING_ACTIVE": "NO"}
    block = 64
    total = k.shape[1]
    blocks = math.ceil(total / block)
    pad = blocks * block - total
    kp = torch.nn.functional.pad(k, (0, 0, 0, 0, 0, pad))
    vp = torch.nn.functional.pad(v, (0, 0, 0, 0, 0, pad))
    qp = q.float().mean(dim=1, keepdim=True)
    kb = kp.reshape(k.shape[0], blocks, block, k.shape[2], k.shape[3]).float().mean(dim=2)
    scores = torch.einsum("bqhd,bkhd->bk", qp, kb).mean(dim=1)
    keep = max(1, math.ceil(blocks * (1.0 - ratio)))
    ids = torch.topk(scores, keep, dim=-1).indices.sort(dim=-1).values
    token_ids = (ids[..., None] * block + torch.arange(block, device=k.device)).reshape(k.shape[0], -1)
    token_ids = token_ids.clamp_max(total - 1)
    gather = token_ids[:, :, None, None].expand(-1, -1, k.shape[2], k.shape[3])
    ko = torch.gather(k, 1, gather)
    vo = torch.gather(v, 1, gather)
    total_interactions = int(q.shape[1] * total)
    retained_interactions = int(q.shape[1] * ko.shape[1])
    return ko, vo, {
        "SPARSE_ROUTING_ACTIVE": "YES",
        "TOTAL_INTERACTIONS": total_interactions,
        "RETAINED_INTERACTIONS": retained_interactions,
        "SKIPPED_INTERACTIONS": total_interactions - retained_interactions,
        "ACTUAL_SPARSE_RATIO": 1.0 - retained_interactions / max(total_interactions, 1),
    }


def prepare_attention_kv(q, k, v, mode="baseline", sparse_ratio=0.0):
    k, v, meta = fake_quantize_kv(k, v, mode)
    if sparse_ratio:
        k, v, sparse = sparse_retain(q, k, v, sparse_ratio)
        meta.update(sparse)
    meta["FINAL_ATTENTION_DTYPE"] = str(k.dtype).replace("torch.", "")
    meta["NATIVE_LOWBIT_KERNEL_USED"] = "NO"
    return k, v, meta

