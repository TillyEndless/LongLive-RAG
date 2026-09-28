"""Authoritative full-precision CPU history archive for Groups 12--15.

The group-specific low-bit representation is a GPU working-set operation and
must happen only after a selected BF16 history block has crossed H2D.  This
archive therefore owns BF16 K/V only; the class name is retained for import
compatibility with the existing pipeline.
"""
from dataclasses import dataclass
from typing import Any
import time

import torch



@dataclass
class CompressedHistoryRecord:
    history_id: int
    layer_id: int
    chunk_id: int
    valid_tokens: int
    mode: str
    k_payload: torch.Tensor
    v_payload: torch.Tensor
    k_meta: Any = None
    v_meta: Any = None
    tensor_shape: tuple[int, int, int, int] | None = None
    # GPU-owned low-bit representation. These are populated lazily on the
    # first selected fetch and remain the long-lived owner thereafter.
    gpu_k_payload: Any = None
    gpu_v_payload: Any = None
    gpu_k_meta: Any = None
    gpu_v_meta: Any = None

    def persistent_bytes(self) -> int:
        total = 0
        values = []
        values = [self.k_payload, self.v_payload]
        for value in values:
            if value is not None and hasattr(value, "untyped_storage"):
                total += int(value.untyped_storage().nbytes())
        return total

    def cpu_persistent_bytes(self) -> int:
        return sum(int(x.untyped_storage().nbytes()) for x in (self.k_payload, self.v_payload))

    def gpu_persistent_bytes(self) -> int:
        total = 0
        for value in (self.gpu_k_payload, self.gpu_v_payload,
                      self.gpu_k_meta, self.gpu_v_meta):
            if value is None:
                continue
            if hasattr(value, "untyped_storage"):
                total += int(value.untyped_storage().nbytes())
            else:
                for name in ("values", "scale_factors", "amax"):
                    item = getattr(value, name, None)
                    if callable(item) and not isinstance(item, torch.Tensor):
                        item = item()
                    if item is not None and hasattr(item, "untyped_storage"):
                        total += int(item.untyped_storage().nbytes())
        return total


class CompressedHistoryArchive:
    """BF16 CPU history shared by Group12/13/14/15.

    ``mode`` describes the GPU-side transform to apply after H2D.  It does
    not change the CPU archive representation.
    """

    def __init__(self, mode: str):
        if mode not in {"int8_fp8", "nvfp4"}:
            raise ValueError(mode)
        self.mode = mode
        self.records: list[CompressedHistoryRecord] = []
        self.last_h2d_ms = 0.0
        self.last_h2d_calls = 0

    def append(self, k_bf16: torch.Tensor, v_bf16: torch.Tensor, *, layer_id: int,
               chunk_id: int, history_id: int, valid_tokens: int) -> CompressedHistoryRecord:
        if k_bf16.ndim != 4 or v_bf16.ndim != 4:
            raise ValueError("archive expects BTHD tensors")
        if k_bf16.dtype != torch.bfloat16 or v_bf16.dtype != torch.bfloat16:
            raise TypeError("archive input must be BF16")
        # The authoritative archive is always BF16 CPU storage.  Detach and
        # clone so no GPU tensor or transient quantized payload is retained.
        k_cpu = k_bf16.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        v_cpu = v_bf16.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        rec = CompressedHistoryRecord(history_id, layer_id, chunk_id, valid_tokens,
                                      self.mode, k_cpu, v_cpu,
                                      tensor_shape=tuple(k_cpu.shape))
        self.records.append(rec)
        return rec

    def fetch(self, rec: CompressedHistoryRecord, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_h2d_ms = 0.0
        self.last_h2d_calls = 0
        if rec.k_payload.dtype != torch.bfloat16 or rec.v_payload.dtype != torch.bfloat16:
            raise AssertionError("CPU historical archive must remain BF16")
        if rec.gpu_k_payload is None:
            from utils.persistent_kv_storage import PersistentHistoryQuantizer
            if self.mode == "int8_fp8":
                quantizer = PersistentHistoryQuantizer()
                # BF16 is transferred first; only the packed tensors and
                # scale metadata remain as persistent GPU owners.
                h2d_start = time.perf_counter()
                k_bf16 = rec.k_payload.to(device=device, dtype=torch.bfloat16)
                v_bf16 = rec.v_payload.to(device=device, dtype=torch.bfloat16)
                self.last_h2d_ms = (time.perf_counter() - h2d_start) * 1000.0
                self.last_h2d_calls = 2
                rec.gpu_k_payload, rec.gpu_k_meta = quantizer.quantize_k(k_bf16)
                rec.gpu_v_payload, rec.gpu_v_meta = quantizer.quantize_v(v_bf16)
                del k_bf16, v_bf16
            else:
                from utils.quant import quantize_kv
                from fouroversix.quantize import QuantizationConfig
                h2d_start = time.perf_counter()
                k_bf16 = rec.k_payload.to(device=device, dtype=torch.bfloat16)
                v_bf16 = rec.v_payload.to(device=device, dtype=torch.bfloat16)
                self.last_h2d_ms = (time.perf_counter() - h2d_start) * 1000.0
                self.last_h2d_calls = 2
                cfg = QuantizationConfig()
                rec.gpu_k_payload = quantize_kv(k_bf16.reshape(-1, k_bf16.shape[-1]), cfg)
                rec.gpu_v_payload = quantize_kv(v_bf16.reshape(-1, v_bf16.shape[-1]), cfg)
                del k_bf16, v_bf16
        if self.mode == "int8_fp8":
            from utils.persistent_kv_storage import PersistentHistoryQuantizer
            quantizer = PersistentHistoryQuantizer()
            return (quantizer.dequantize_k(rec.gpu_k_payload, rec.gpu_k_meta),
                    quantizer.dequantize_v(rec.gpu_v_payload, rec.gpu_v_meta))
        return (rec.gpu_k_payload.dequantize(dtype=torch.bfloat16).reshape(rec.tensor_shape),
                rec.gpu_v_payload.dequantize(dtype=torch.bfloat16).reshape(rec.tensor_shape))

    def total_bytes(self) -> int:
        return sum(r.persistent_bytes() for r in self.records)
