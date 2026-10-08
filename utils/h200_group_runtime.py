"""Portable H200 runtime hooks for corrected Groups 12--15.

Only KV storage is transformed. Queries and the final attention call remain
BF16; this module never imports or dispatches a native low-bit kernel.
"""

from __future__ import annotations

import math
import torch

from utils.persistent_kv_storage import PersistentHistoryQuantizer
from utils.quant import quantize_kv
from utils.persistent_draftmap import route_draftmap
from fouroversix.quantize import QuantizationConfig
import utils.critical_path_trace as cpt


def fake_quantize_kv(k: torch.Tensor, v: torch.Tensor, mode: str):
    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("fake-quant input must be BF16")
    if mode in {"group12_corrected", "group14_corrected"}:
        # CPU history has already crossed H2D as BF16.  Quantize only the GPU
        # working operands, then dequantize for the unchanged BF16 attention.
        quantizer = PersistentHistoryQuantizer()
        qk, sk = quantizer.quantize_k(k)
        qv, sv = quantizer.quantize_v(v)
        return quantizer.dequantize_k(qk, sk), quantizer.dequantize_v(qv, sv), {
            "K_FAKE_QUANT_ACTIVE": "YES",
            "V_FAKE_QUANT_ACTIVE": "YES",
            "K_STORAGE_DTYPE": "int8_gpu_working_set",
            "V_STORAGE_DTYPE": "fp8_e4m3_gpu_working_set",
            "CPU_HISTORY_STORAGE_DTYPE": "bf16",
            "H2D_SOURCE_DTYPE": "bf16",
            "PERSISTENT_STORAGE_MODE": "LOWBIT_STORAGE_BF16_COMPUTE",
            "ARCHIVE_DEQUANT_BEFORE_BF16_ATTENTION": "YES",
            "FINAL_ATTENTION_DTYPE": "bfloat16",
        }
    if mode in {"group13_corrected", "group15_corrected"}:
        cfg = QuantizationConfig()
        def one(x):
            shape = x.shape
            packed = quantize_kv(x.reshape(-1, shape[-1]), cfg)
            return packed.dequantize(dtype=torch.bfloat16).reshape(shape)
        return one(k), one(v), {
            "K_FAKE_QUANT_ACTIVE": "YES",
            "V_FAKE_QUANT_ACTIVE": "YES",
            "K_STORAGE_DTYPE": "nvfp4_gpu_working_set",
            "V_STORAGE_DTYPE": "nvfp4_gpu_working_set",
            "CPU_HISTORY_STORAGE_DTYPE": "bf16",
            "H2D_SOURCE_DTYPE": "bf16",
            "PERSISTENT_STORAGE_MODE": "LOWBIT_STORAGE_BF16_COMPUTE",
            "ARCHIVE_DEQUANT_BEFORE_BF16_ATTENTION": "YES",
            "FINAL_ATTENTION_DTYPE": "bfloat16",
        }
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
    scores = torch.einsum("bqhd,bkhd->bk", qp, kb)
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



def select_q_sparse_history(selected_ids, selected_scores, retained_ratio):
    """Select retained history chunks using the existing CURRENT_Q DraftMap scores.

    The score/order provenance is inherited from the current-Q retrieval call;
    this operation only removes history chunks for Group14/15 and does not
    alter the retrieval candidate IDs or their order.
    """
    ids = [int(x) for x in selected_ids]
    ratio = float(retained_ratio)
    if not 0.0 < ratio < 1.0:
        return ids, [], {
            "Q_SPARSE_ENABLED": "NO",
            "Q_SPARSE_RETAINED_IDS": ids,
            "Q_SPARSE_REMOVED_IDS": [],
            "Q_SPARSE_RETAINED_COUNT": len(ids),
            "Q_SPARSE_TOTAL_COUNT": len(ids),
        }
    score_map = {int(k): float(v) for k, v in selected_scores.items()}
    keep = min(len(ids), max(1, int(math.ceil(len(ids) * ratio))))
    ranked = sorted(ids, key=lambda x: (-score_map.get(x, 0.0), x))
    retained_set = set(ranked[:keep])
    retained = [x for x in ids if x in retained_set]
    removed = [x for x in ids if x not in retained_set]
    return retained, removed, {
        "Q_SPARSE_ENABLED": "YES",
        "Q_SPARSE_RETAINED_RATIO": float(len(retained) / max(1, len(ids))),
        "Q_SPARSE_REQUESTED_RATIO": ratio,
        "Q_SPARSE_RETAINED_IDS": retained,
        "Q_SPARSE_REMOVED_IDS": removed,
        "Q_SPARSE_RETAINED_COUNT": len(retained),
        "Q_SPARSE_TOTAL_COUNT": len(ids),
        "Q_SPARSE_REMOVED_COUNT": len(removed),
        "Q_SPARSE_SCORE_SOURCE": "CURRENT_Q_DRAFTMAP_SELECTED_SCORES",
    }

def prepare_attention_kv(q, k, v, mode="baseline", sparse_ratio=0.0,
                         persistent_owner_already_dequantized=False,
                         apply_storage_quant=True):
    # Local/sink/current tensors are transient attention operands, not
    # persistent historical KV.  The low-bit contract applies only to the
    # archive-backed GPU owner; keep these operands BF16 and do not introduce
    # an accidental quantize->dequantize round trip here.
    if not apply_storage_quant and not persistent_owner_already_dequantized:
        k, v, meta = k, v, {
            "K_FAKE_QUANT_ACTIVE": "NO",
            "V_FAKE_QUANT_ACTIVE": "NO",
            "GPU_PERSISTENT_LOWBIT_OWNER": "NO",
            "CPU_HISTORY_STORAGE_DTYPE": "bf16",
            "FINAL_ATTENTION_DTYPE": "bfloat16",
            "TRANSIENT_BF16_DEQUANT": "NO",
            "PERSISTENT_STORAGE_MODE": "LOWBIT_STORAGE_BF16_COMPUTE"
            if mode in {"group12_corrected", "group13_corrected", "group14_corrected", "group15_corrected"}
            else "BF16_FAKE_QUANT",
        }
    elif persistent_owner_already_dequantized:
        if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise TypeError("persistent low-bit owner must materialize BF16 operands")
        k, v, meta = k, v, {
            "K_FAKE_QUANT_ACTIVE": "YES",
            "V_FAKE_QUANT_ACTIVE": "YES",
            "GPU_PERSISTENT_LOWBIT_OWNER": "YES",
            "CPU_HISTORY_STORAGE_DTYPE": "bf16",
            "H2D_SOURCE_DTYPE": "bf16",
            "FINAL_ATTENTION_DTYPE": "bfloat16",
            "TRANSIENT_BF16_DEQUANT": "YES",
            "PERSISTENT_STORAGE_MODE": "LOWBIT_STORAGE_BF16_COMPUTE",
        }
    else:
        k, v, meta = fake_quantize_kv(k, v, mode)
    if sparse_ratio:
        if mode in {"group14_corrected", "group15_corrected"}:
            trace = cpt.ACTIVE_TRACE
            if trace:
                k, v, sparse = trace.measure(
                    "DRAFTMAP_ROUTE_PARENT",
                    lambda: route_draftmap(q, k, v, sparse_ratio),
                )
            else:
                k, v, sparse = route_draftmap(q, k, v, sparse_ratio)
        else:
            k, v, sparse = sparse_retain(q, k, v, sparse_ratio)
        meta.update(sparse)
    meta["FINAL_ATTENTION_DTYPE"] = str(k.dtype).replace("torch.", "")
    meta["NATIVE_LOWBIT_KERNEL_USED"] = "NO"
    return k, v, meta
