"""Persistent packed NVFP4 KV slots with temporary dense materialization."""

from __future__ import annotations

import torch

from fouroversix.quantize import QuantizedTensor
from utils.quant import clone_quantized_tensor, dequantize_kv_cache, quantize_kv


class PersistentNVFP4Cache:
    def __init__(self, max_blocks, block_tokens, num_heads, head_dim, dtype, device, config=None):
        if head_dim <= 0 or block_tokens <= 0 or num_heads <= 0:
            raise ValueError("cache dimensions must be positive")
        self.max_blocks = int(max_blocks)
        self.block_tokens = int(block_tokens)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.dtype = dtype
        self.device = torch.device(device)
        self.config = config
        self.k: list[QuantizedTensor | None] = [None] * self.max_blocks
        self.v: list[QuantizedTensor | None] = [None] * self.max_blocks
        self.valid_tokens = torch.zeros(self.max_blocks, dtype=torch.long, device=self.device)

    def _quantize(self, x: torch.Tensor) -> QuantizedTensor:
        flat = x.reshape(-1, self.head_dim).to(device=self.device, dtype=self.dtype)
        return quantize_kv(flat, self.config)

    def insert(self, block_id: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if not 0 <= block_id < self.max_blocks:
            raise IndexError(block_id)
        expected = (1, self.block_tokens, self.num_heads, self.head_dim)
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(f"expected KV shape {expected}, got {tuple(k.shape)} and {tuple(v.shape)}")
        self.k[block_id] = self._quantize(k)
        self.v[block_id] = self._quantize(v)
        self.valid_tokens[block_id] = self.block_tokens

    def roll(self, dst_block: int, src_block: int) -> None:
        self.k[dst_block] = None if self.k[src_block] is None else clone_quantized_tensor(self.k[src_block])
        self.v[dst_block] = None if self.v[src_block] is None else clone_quantized_tensor(self.v[src_block])
        self.valid_tokens[dst_block] = self.valid_tokens[src_block]

    def materialize(self, max_blocks: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            dequantize_kv_cache(self.k, self.block_tokens, self.num_heads, self.dtype, self.device, max_blocks),
            dequantize_kv_cache(self.v, self.block_tokens, self.num_heads, self.dtype, self.device, max_blocks),
        )

    def persistent_bytes(self) -> int:
        total = int(self.valid_tokens.numel() * self.valid_tokens.element_size())
        for slot in (*self.k, *self.v):
            if slot is not None:
                total += slot.values.numel() * slot.values.element_size()
                total += slot.scale_factors.numel() * slot.scale_factors.element_size()
                if slot.amax is not None:
                    total += slot.amax.numel() * slot.amax.element_size()
        return total
