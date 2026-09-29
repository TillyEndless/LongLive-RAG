"""Storage-only historical K/V quantization for LongLive retrieval."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import utils.critical_path_trace as cpt


FP8_E4M3 = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_E4M3).max)


@dataclass(frozen=True)
class QuantizedHistoricalKVEntry:
    """Immutable CPU archive record; no BF16 shadow tensors are retained."""

    history_id: int
    k_int8: torch.Tensor
    k_scale: torch.Tensor
    v_fp8: torch.Tensor
    v_scale: torch.Tensor
    original_shape: tuple[int, ...]
    start_token: int

    @property
    def persistent_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.k_int8, self.k_scale, self.v_fp8, self.v_scale)
        )

    @property
    def bf16_equivalent_bytes(self) -> int:
        return 2 * self.k_int8.numel() + 2 * self.v_fp8.numel()


class PersistentHistoryQuantizer:
    """Quantize only archived un-RoPE BF16 historical frames.

    K scale: FP32 [B, H, ceil(T / 64)], one scalar per head and K64 block.
    V scale: FP32 [B, H, D], one immutable per-chunk per-head/channel scale.
    """

    def __init__(self, *, k_block_tokens: int = 64):
        if k_block_tokens <= 0:
            raise ValueError("k_block_tokens must be positive")
        self.k_block_tokens = int(k_block_tokens)

    def quantize_k(self, k_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if k_bf16.ndim != 4 or k_bf16.dtype != torch.bfloat16:
            raise ValueError("K archive input must be BF16 BTHD")
        b, tokens, heads, dim = k_bf16.shape
        blocks = (tokens + self.k_block_tokens - 1) // self.k_block_tokens
        padded = torch.nn.functional.pad(
            k_bf16, (0, 0, 0, 0, 0, blocks * self.k_block_tokens - tokens)
        ).view(b, blocks, self.k_block_tokens, heads, dim)
        scale = padded.float().abs().amax(dim=(2, 4)).permute(0, 2, 1).contiguous()
        scale = torch.where(scale == 0, torch.ones_like(scale), scale / 127.0)
        q = torch.round(
            padded.float() / scale.permute(0, 2, 1).unsqueeze(2).unsqueeze(-1)
        ).clamp(-127, 127).to(torch.int8)
        return q.view(b, blocks * self.k_block_tokens, heads, dim)[:, :tokens].contiguous(), scale

    def dequantize_k(self, k_int8: torch.Tensor, k_scale: torch.Tensor) -> torch.Tensor:
        if k_int8.dtype != torch.int8 or k_scale.dtype != torch.float32:
            raise ValueError("invalid K archive dtypes")
        b, tokens, heads, dim = k_int8.shape
        blocks = k_scale.shape[-1]
        padded_tokens = blocks * self.k_block_tokens
        q = torch.nn.functional.pad(k_int8, (0, 0, 0, 0, 0, padded_tokens - tokens))
        scale = k_scale.permute(0, 2, 1).unsqueeze(2).unsqueeze(-1)
        decoded = (q.float().view(b, blocks, self.k_block_tokens, heads, dim) * scale).view(
            b, padded_tokens, heads, dim
        )[:, :tokens]
        trace = cpt.ACTIVE_TRACE
        return (trace.measure('BF16_K_MATERIALIZATION_CONTIGUOUS',
                              lambda: decoded.to(torch.bfloat16).contiguous())
                if trace else decoded.to(torch.bfloat16).contiguous())

    def quantize_v(self, v_bf16: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if v_bf16.ndim != 4 or v_bf16.dtype != torch.bfloat16:
            raise ValueError("V archive input must be BF16 BTHD")
        # Per archived chunk, head, and channel; no cross-chunk rescaling.
        scale = v_bf16.float().abs().amax(dim=1).permute(0, 1, 2).contiguous() / FP8_MAX
        scale = torch.where(scale == 0, torch.ones_like(scale), scale)
        q = (v_bf16.float() / scale.unsqueeze(1)).clamp(-FP8_MAX, FP8_MAX).to(FP8_E4M3)
        return q.contiguous(), scale

    def dequantize_v(self, v_fp8: torch.Tensor, v_scale: torch.Tensor) -> torch.Tensor:
        if v_fp8.dtype != FP8_E4M3 or v_scale.dtype != torch.float32:
            raise ValueError("invalid V archive dtypes")
        decoded = v_fp8.float() * v_scale.unsqueeze(1)
        trace = cpt.ACTIVE_TRACE
        return (trace.measure('BF16_V_MATERIALIZATION_CONTIGUOUS',
                              lambda: decoded.to(torch.bfloat16).contiguous())
                if trace else decoded.to(torch.bfloat16).contiguous())

    def archive(
        self, history_id: int, k_bf16: torch.Tensor, v_bf16: torch.Tensor, *, start_token: int
    ) -> QuantizedHistoricalKVEntry:
        if k_bf16.shape != v_bf16.shape:
            raise ValueError("historical K/V shapes must match")
        k8, ks = self.quantize_k(k_bf16)
        v8, vs = self.quantize_v(v_bf16)
        return QuantizedHistoricalKVEntry(
            int(history_id), k8.cpu(), ks.cpu(), v8.cpu(), vs.cpu(),
            tuple(k_bf16.shape), int(start_token),
        )

    def fetch(self, entry: QuantizedHistoricalKVEntry, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.dequantize_k(entry.k_int8.to(device), entry.k_scale.to(device))
        v = self.dequantize_v(entry.v_fp8.to(device), entry.v_scale.to(device))
        return k, v


__all__ = ["FP8_E4M3", "QuantizedHistoricalKVEntry", "PersistentHistoryQuantizer"]
