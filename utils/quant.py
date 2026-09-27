"""FourOverSix NVFP4 helpers for persistent LongLive KV storage."""

from dataclasses import dataclass

import torch

from fouroversix.quantize import QuantizationConfig, QuantizedTensor, quantize_to_fp4
from fouroversix.utils import ScaleRule


@dataclass
class LongLiveQuantizationConfig(QuantizationConfig):
    type: str = "kv"


def quantize_kv(x: torch.Tensor, config: QuantizationConfig) -> QuantizedTensor:
    """Quantize flattened ``[tokens * heads, head_dim]`` KV data."""
    if x.ndim != 2:
        raise ValueError(f"expected 2-D KV input, got {tuple(x.shape)}")
    return quantize_to_fp4(x, config)


def dequantize_kv_cache(
    slots: list[QuantizedTensor | None],
    block_tokens: int,
    num_heads: int,
    dtype: torch.dtype,
    device: torch.device,
    max_blocks: int | None = None,
) -> torch.Tensor:
    """Materialize active packed blocks into contiguous ``[1,T,H,D]`` BF16/FP16."""
    count = len(slots) if max_blocks is None else min(max_blocks, len(slots))
    blocks = [slot for slot in slots[:count] if slot is not None]
    if not blocks:
        head_dim = 128
        return torch.empty((1, 0, num_heads, head_dim), dtype=dtype, device=device)
    tensors = [slot.dequantize(dtype=dtype).reshape(-1, num_heads, slot.original_shape[1]) for slot in blocks]
    return torch.cat(tensors, dim=0).unsqueeze(0).to(device=device)


def clone_quantized_tensor(slot: QuantizedTensor) -> QuantizedTensor:
    return QuantizedTensor(
        slot.values.clone(),
        slot.scale_factors.clone(),
        None if slot.amax is None else slot.amax.clone(),
        slot.dtype,
        slot.original_shape,
        slot.scale_rule,
        slot.padded_shape,
    )


def copy_quantized_into(dst: QuantizedTensor, src: QuantizedTensor) -> None:
    if dst.values.shape != src.values.shape or dst.scale_factors.shape != src.scale_factors.shape:
        raise ValueError("incompatible QuantizedTensor layouts")
    dst.values.copy_(src.values)
    dst.scale_factors.copy_(src.scale_factors)
    if dst.amax is not None and src.amax is not None:
        dst.amax.copy_(src.amax)
