"""Authoritative full-precision CPU history archive for Groups 12--15.

The group-specific low-bit representation is a GPU working-set operation and
must happen only after a selected BF16 history block has crossed H2D.  This
archive therefore owns BF16 K/V only; the class name is retained for import
compatibility with the existing pipeline.
"""
from dataclasses import dataclass
from typing import Any, Iterable
import time

import torch
import utils.critical_path_trace as cpt



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

    def __init__(self, mode: str, *, promotion_ratio: float = 0.0,
                 promotion_source: str = "draftmap",
                 materialization_mode: str = "serial_reference"):
        if mode not in {"int8_fp8", "nvfp4"}:
            raise ValueError(mode)
        if not 0.0 <= float(promotion_ratio) <= 1.0:
            raise ValueError("promotion_ratio must be in [0, 1]")
        if promotion_source != "draftmap":
            raise ValueError("only DraftMap promotion is supported")
        if materialization_mode not in {"serial_reference", "async_overlap"}:
            raise ValueError("invalid materialization_mode")
        self.mode = mode
        self.promotion_ratio = float(promotion_ratio)
        self.promotion_source = promotion_source
        self.materialization_mode = materialization_mode
        self.records: list[CompressedHistoryRecord] = []
        self.last_h2d_ms = 0.0
        self.last_h2d_calls = 0
        self.last_metadata: dict[str, Any] = {}
        self.last_transient_bf16_bytes = 0
        self.last_promotion_bytes = 0
        self.last_promotion_calls = 0
        self.last_dequant_bytes = 0
        self.last_promotion_enqueue_ms = 0.0
        self.last_promotion_cuda_work_ms = 0.0
        self._promotion_event_records = []
        self.last_dequant_enqueue_ms = 0.0
        self.last_join_wait_ms = 0.0
        self.last_materialization_host_ms = 0.0

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
                trace = cpt.ACTIVE_TRACE
                k_bf16 = (trace.measure('FULL_K_H2D', lambda: rec.k_payload.to(device=device, dtype=torch.bfloat16))
                          if trace else rec.k_payload.to(device=device, dtype=torch.bfloat16))
                v_bf16 = (trace.measure('FULL_V_H2D', lambda: rec.v_payload.to(device=device, dtype=torch.bfloat16))
                          if trace else rec.v_payload.to(device=device, dtype=torch.bfloat16))
                self.last_h2d_ms = (time.perf_counter() - h2d_start) * 1000.0
                self.last_h2d_calls = 2
                rec.gpu_k_payload, rec.gpu_k_meta = quantizer.quantize_k(k_bf16)
                rec.gpu_v_payload, rec.gpu_v_meta = quantizer.quantize_v(v_bf16)
                del k_bf16, v_bf16
            else:
                from utils.quant import quantize_kv
                from fouroversix.quantize import QuantizationConfig
                h2d_start = time.perf_counter()
                trace = cpt.ACTIVE_TRACE
                k_bf16 = (trace.measure('FULL_K_H2D', lambda: rec.k_payload.to(device=device, dtype=torch.bfloat16))
                          if trace else rec.k_payload.to(device=device, dtype=torch.bfloat16))
                v_bf16 = (trace.measure('FULL_V_H2D', lambda: rec.v_payload.to(device=device, dtype=torch.bfloat16))
                          if trace else rec.v_payload.to(device=device, dtype=torch.bfloat16))
                self.last_h2d_ms = (time.perf_counter() - h2d_start) * 1000.0
                self.last_h2d_calls = 2
                cfg = QuantizationConfig()
                rec.gpu_k_payload = quantize_kv(k_bf16.reshape(-1, k_bf16.shape[-1]), cfg)
                rec.gpu_v_payload = quantize_kv(v_bf16.reshape(-1, v_bf16.shape[-1]), cfg)
                del k_bf16, v_bf16
        if self.mode == "int8_fp8":
            from utils.persistent_kv_storage import PersistentHistoryQuantizer
            quantizer = PersistentHistoryQuantizer()
            trace = cpt.ACTIVE_TRACE
            k_out = (trace.measure('FULL_LOWBIT_K_DECODE', lambda: quantizer.dequantize_k(rec.gpu_k_payload, rec.gpu_k_meta))
                     if trace else quantizer.dequantize_k(rec.gpu_k_payload, rec.gpu_k_meta))
            v_out = (trace.measure('FULL_LOWBIT_V_DECODE', lambda: quantizer.dequantize_v(rec.gpu_v_payload, rec.gpu_v_meta))
                     if trace else quantizer.dequantize_v(rec.gpu_v_payload, rec.gpu_v_meta))
            return k_out, v_out
        trace = cpt.ACTIVE_TRACE
        k_out = (trace.measure('FULL_LOWBIT_K_DECODE', lambda: rec.gpu_k_payload.dequantize(dtype=torch.bfloat16).reshape(rec.tensor_shape))
                 if trace else rec.gpu_k_payload.dequantize(dtype=torch.bfloat16).reshape(rec.tensor_shape))
        v_out = (trace.measure('FULL_LOWBIT_V_DECODE', lambda: rec.gpu_v_payload.dequantize(dtype=torch.bfloat16).reshape(rec.tensor_shape))
                 if trace else rec.gpu_v_payload.dequantize(dtype=torch.bfloat16).reshape(rec.tensor_shape))
        return k_out, v_out


    def _ensure_lowbit(self, rec: CompressedHistoryRecord, device: torch.device) -> None:
        """Create the persistent GPU low-bit owner once, never a BF16 shadow."""
        if rec.gpu_k_payload is not None:
            owner_device = rec.gpu_k_payload.device
            requested_device = torch.device(device)
            if owner_device.type != requested_device.type or (
                requested_device.index is not None and owner_device.index != requested_device.index
            ):
                raise RuntimeError("persistent low-bit payload is on the wrong device")
            return
        if self.mode == "int8_fp8":
            from utils.persistent_kv_storage import PersistentHistoryQuantizer
            quantizer = PersistentHistoryQuantizer()
            k_bf16 = rec.k_payload.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            v_bf16 = rec.v_payload.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            rec.gpu_k_payload, rec.gpu_k_meta = quantizer.quantize_k(k_bf16)
            rec.gpu_v_payload, rec.gpu_v_meta = quantizer.quantize_v(v_bf16)
            del k_bf16, v_bf16
        else:
            from utils.quant import quantize_kv
            from fouroversix.quantize import QuantizationConfig
            cfg = QuantizationConfig()
            k_bf16 = rec.k_payload.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            v_bf16 = rec.v_payload.to(device=device, dtype=torch.bfloat16, non_blocking=True)
            rec.gpu_k_payload = quantize_kv(k_bf16.reshape(-1, k_bf16.shape[-1]), cfg)
            rec.gpu_v_payload = quantize_kv(v_bf16.reshape(-1, v_bf16.shape[-1]), cfg)
            del k_bf16, v_bf16

    @staticmethod
    def _promotion_positions(ids: list[int], scores: dict[int, float], ratio: float) -> set[int]:
        if ratio <= 0.0 or not ids:
            return set()
        if ratio >= 1.0:
            return set(range(len(ids)))
        count = min(len(ids), max(1, int(torch.ceil(torch.tensor(len(ids) * ratio)).item())))
        ranked = sorted(range(len(ids)), key=lambda p: (-float(scores.get(ids[p], 0.0)), ids[p]))
        return set(ranked[:count])

    def materialize_selected(self, ids: Iterable[int], scores: dict[int, float],
                             device: torch.device, *, promotion_ratio: float | None = None,
                             materialization_mode: str | None = None):
        """Materialize exactly ids once each in the original order."""
        ids = [int(x) for x in ids]
        ratio = self.promotion_ratio if promotion_ratio is None else float(promotion_ratio)
        mode = self.materialization_mode if materialization_mode is None else materialization_mode
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("promotion_ratio must be in [0, 1]")
        if any(i < 0 or i >= len(self.records) for i in ids):
            raise IndexError("selected history id is not present in CPU archive")
        positions = self._promotion_positions(ids, scores, ratio)
        promoted_ids = [ids[p] for p in sorted(positions)]
        lowbit_ids = [ids[p] for p in range(len(ids)) if p not in positions]
        k_out: list[torch.Tensor | None] = [None] * len(ids)
        v_out: list[torch.Tensor | None] = [None] * len(ids)
        self.last_h2d_ms = 0.0
        self.last_h2d_calls = 0
        self.last_promotion_bytes = 0
        self.last_promotion_calls = 0
        self.last_dequant_bytes = 0
        self.last_promotion_enqueue_ms = 0.0
        self.last_promotion_cuda_work_ms = 0.0
        self._promotion_event_records.clear()
        self.last_dequant_enqueue_ms = 0.0
        self.last_join_wait_ms = 0.0
        materialize_start = time.perf_counter()
        use_async = mode == "async_overlap" and device.type == "cuda" and len(ids) > 1
        if use_async:
            promotion_stream = torch.cuda.Stream(device=device)
            dequant_stream = torch.cuda.Stream(device=device)
            phase_start = time.perf_counter()
            promotion_start = torch.cuda.Event(enable_timing=True)
            promotion_end = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(promotion_stream):
                promotion_start.record(promotion_stream)
                for p in sorted(positions):
                    rec = self.records[ids[p]]
                    k_out[p] = rec.k_payload.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                    v_out[p] = rec.v_payload.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                    self.last_h2d_calls += 2
                    self.last_promotion_calls += 2
                    self.last_promotion_bytes += int(rec.k_payload.numel() * rec.k_payload.element_size() + rec.v_payload.numel() * rec.v_payload.element_size())
                promotion_end.record(promotion_stream)
            self._promotion_event_records.append((promotion_start, promotion_end))
            self.last_promotion_enqueue_ms = (time.perf_counter() - phase_start) * 1000.0
            promotion_event = torch.cuda.Event(); promotion_event.record(promotion_stream)
            phase_start = time.perf_counter()
            with torch.cuda.stream(dequant_stream):
                for p in range(len(ids)):
                    if p in positions:
                        continue
                    rec = self.records[ids[p]]
                    self._ensure_lowbit(rec, device)
                    if self.mode == "int8_fp8":
                        from utils.persistent_kv_storage import PersistentHistoryQuantizer
                        quantizer = PersistentHistoryQuantizer()
                        k_out[p] = quantizer.dequantize_k(rec.gpu_k_payload, rec.gpu_k_meta)
                        v_out[p] = quantizer.dequantize_v(rec.gpu_v_payload, rec.gpu_v_meta)
                    else:
                        k_out[p] = rec.gpu_k_payload.dequantize(dtype=torch.bfloat16).reshape(rec.tensor_shape)
                        v_out[p] = rec.gpu_v_payload.dequantize(dtype=torch.bfloat16).reshape(rec.tensor_shape)
                    self.last_dequant_bytes += int(k_out[p].numel() * k_out[p].element_size() + v_out[p].numel() * v_out[p].element_size())
            self.last_dequant_enqueue_ms = (time.perf_counter() - phase_start) * 1000.0
            dequant_event = torch.cuda.Event(); dequant_event.record(dequant_stream)
            current = torch.cuda.current_stream(device)
            current.wait_event(promotion_event)
            current.wait_event(dequant_event)
        else:
            for p, history_id in enumerate(ids):
                rec = self.records[history_id]
                if p in positions:
                    phase_start = time.perf_counter()
                    promotion_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                    promotion_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                    if promotion_start is not None:
                        promotion_start.record(torch.cuda.current_stream(device))
                    k_out[p] = rec.k_payload.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                    v_out[p] = rec.v_payload.to(device=device, dtype=torch.bfloat16, non_blocking=True)
                    if promotion_end is not None:
                        promotion_end.record(torch.cuda.current_stream(device))
                        self._promotion_event_records.append((promotion_start, promotion_end))
                    self.last_h2d_calls += 2
                    self.last_promotion_calls += 2
                    self.last_promotion_bytes += int(rec.k_payload.numel() * rec.k_payload.element_size() + rec.v_payload.numel() * rec.v_payload.element_size())
                    self.last_promotion_enqueue_ms += (time.perf_counter() - phase_start) * 1000.0
                else:
                    phase_start = time.perf_counter()
                    k_out[p], v_out[p] = self.fetch(rec, device)
                    self.last_dequant_enqueue_ms += (time.perf_counter() - phase_start) * 1000.0
                    self.last_dequant_bytes += int(k_out[p].numel() * k_out[p].element_size() + v_out[p].numel() * v_out[p].element_size())
        if any(k is None or v is None for k, v in zip(k_out, v_out)):
            raise RuntimeError("materialization produced a missing selected entry")
        self.last_transient_bf16_bytes = self.last_promotion_bytes + self.last_dequant_bytes
        self.last_materialization_host_ms = (time.perf_counter() - materialize_start) * 1000.0
        self.last_metadata = {
            "SELECTED_IDS": ids,
            "PROMOTED_IDS": promoted_ids,
            "LOWBIT_IDS": lowbit_ids,
            "SELECTED_COUNT": len(ids),
            "PROMOTED_COUNT": len(promoted_ids),
            "LOWBIT_COUNT": len(lowbit_ids),
            "PROMOTION_RATIO": ratio,
            "PROMOTION_SOURCE": self.promotion_source,
            "PROMOTED_SOURCE": "CPU_BF16",
            "NONPROMOTED_SOURCE": "GPU_LOWBIT_DEQUANT",
            "SOURCE_CARDINALITY_PER_ID": 1,
            "FINAL_HISTORY_ORDER_EXACT": True,
            "NO_DUPLICATE_HISTORY_ID": True,
            "MATERIALIZATION_MODE": mode,
            "PROMOTION_COPY_EVENT_PRESENT": bool(use_async and promoted_ids),
            "DEQUANT_DONE_EVENT_PRESENT": bool(use_async and lowbit_ids),
            "NO_GLOBAL_SYNC_IN_HOT_PATH": True,
            "PROMOTION_H2D_HOST_ENQUEUE_MS": self.last_promotion_enqueue_ms,
            "LOWBIT_DEQUANT_HOST_ENQUEUE_MS": self.last_dequant_enqueue_ms,
            "JOIN_WAIT_HOST_MS": self.last_join_wait_ms,
            "MATERIALIZATION_HOST_MS": self.last_materialization_host_ms,
            "MATERIALIZATION_TIMING_SEMANTICS": "host_observed_enqueue_not_cuda_work",
            "SOURCE_BY_ID": {str(i): ("CPU_BF16" if i in promoted_ids else "GPU_LOWBIT_DEQUANT") for i in ids},
            "PROMOTION_H2D_HOST_ENQUEUE_MS": self.last_promotion_enqueue_ms,
            "LOWBIT_DEQUANT_HOST_ENQUEUE_MS": self.last_dequant_enqueue_ms,
            "JOIN_WAIT_HOST_MS": self.last_join_wait_ms,
            "MATERIALIZATION_HOST_MS": self.last_materialization_host_ms,
            "MATERIALIZATION_TIMING_SEMANTICS": "host_observed_enqueue_not_cuda_work",
            "SOURCE_BY_ID": {str(i): ("CPU_BF16" if i in promoted_ids else "GPU_LOWBIT_DEQUANT") for i in ids},
        }
        return list(k_out), list(v_out), dict(self.last_metadata)

    def finalize_h2d_timing(self):
        if self._promotion_event_records:
            torch.cuda.synchronize()
            self.last_promotion_cuda_work_ms = sum(float(start.elapsed_time(end)) for start, end in self._promotion_event_records)
            self._promotion_event_records.clear()
        return {"cuda_work_ms": float(self.last_promotion_cuda_work_ms), "host_enqueue_ms": float(self.last_promotion_enqueue_ms), "bytes": int(self.last_promotion_bytes), "calls": int(self.last_promotion_calls)}

    def total_bytes(self) -> int:
        return sum(r.persistent_bytes() for r in self.records)
