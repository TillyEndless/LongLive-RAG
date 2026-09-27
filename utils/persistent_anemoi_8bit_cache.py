"""Persistent native Anemoi INT8/FP8 historical KV storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Anemoi8BitChunk:
    chunk_id: int
    k8: torch.Tensor
    k_scale: torch.Tensor
    v8: torch.Tensor
    v_scale: torch.Tensor
    valid_tokens: int
    start_token: int
    valid_counts: torch.Tensor | None = None


class PersistentAnemoi8BitCache:
    """Own immutable packed historical chunks; never retain BF16 K/V."""

    def __init__(self, max_chunks: int):
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        self.max_chunks = int(max_chunks)
        self.chunks: list[Anemoi8BitChunk] = []
        self._next_chunk_id = 0

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def chunk_ids(self) -> list[int]:
        return [chunk.chunk_id for chunk in self.chunks]

    @property
    def persistent_bf16_bytes(self) -> int:
        return 0

    @property
    def persistent_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for chunk in self.chunks
            for tensor in (chunk.k8, chunk.k_scale, chunk.v8, chunk.v_scale, chunk.valid_counts)
            if tensor is not None
        )

    def append(
        self,
        k8: torch.Tensor,
        k_scale: torch.Tensor,
        v8: torch.Tensor,
        v_scale: torch.Tensor,
        *,
        valid_tokens: int,
        start_token: int | None = None,
        valid_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if k8.dtype != torch.int8:
            raise TypeError("k8 must be signed INT8")
        if v8.dtype != torch.float8_e4m3fn:
            raise TypeError("v8 must be FP8 E4M3")
        if k_scale.dtype != torch.float32 or v_scale.dtype != torch.float32:
            raise TypeError("scales must be FP32")
        if valid_tokens <= 0:
            raise ValueError("valid_tokens must be positive")
        if not all(t.is_contiguous() for t in (k8, k_scale, v8, v_scale)):
            raise ValueError("packed operands must be contiguous")
        chunk = Anemoi8BitChunk(
            self._next_chunk_id,
            k8,
            k_scale,
            v8,
            v_scale,
            int(valid_tokens),
            int(self.chunk_count * valid_tokens if start_token is None else start_token),
            valid_counts,
        )
        self._next_chunk_id += 1
        self.chunks.append(chunk)
        del self.chunks[:-self.max_chunks]
        return k8, k_scale, v8, v_scale

    def replace_last(
        self,
        k8: torch.Tensor,
        k_scale: torch.Tensor,
        v8: torch.Tensor,
        v_scale: torch.Tensor,
        *,
        valid_tokens: int,
        start_token: int,
        valid_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if not self.chunks or self.chunks[-1].start_token != int(start_token):
            raise ValueError("replace_last requires the matching final chunk")
        self.chunks.pop()
        self._next_chunk_id -= 1
        return self.append(
            k8, k_scale, v8, v_scale,
            valid_tokens=valid_tokens, start_token=start_token,
            valid_counts=valid_counts,
        )

    def clear(self) -> None:
        self.chunks.clear()


__all__ = ["Anemoi8BitChunk", "PersistentAnemoi8BitCache"]
