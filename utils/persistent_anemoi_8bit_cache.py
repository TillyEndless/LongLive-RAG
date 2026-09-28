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
    draft_k: torch.Tensor | None = None
    draft_valid: torch.Tensor | None = None


class PersistentAnemoi8BitCache:
    """Own immutable packed historical chunks; never retain BF16 K/V."""

    def __init__(self, max_chunks: int):
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        self.max_chunks = int(max_chunks)
        self.chunks: list[Anemoi8BitChunk] = []
        self._archived_chunks: list[Anemoi8BitChunk] = []
        self._next_chunk_id = 0

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def chunk_ids(self) -> list[int]:
        return [chunk.chunk_id for chunk in self.chunks]

    @property
    def archived_chunk_ids(self) -> list[int]:
        return [chunk.chunk_id for chunk in self._archived_chunks]

    @property
    def history_chunks(self) -> list[Anemoi8BitChunk]:
        return [*self._archived_chunks, *self.chunks]

    @property
    def persistent_bf16_bytes(self) -> int:
        return 0

    @property
    def persistent_draft_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for chunk in self.history_chunks
            for tensor in (chunk.draft_k, chunk.draft_valid)
            if tensor is not None
        )

    @property
    def persistent_gpu_draft_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for chunk in self.history_chunks
            for tensor in (chunk.draft_k, chunk.draft_valid)
            if tensor is not None and tensor.device.type == "cuda"
        )

    @property
    def persistent_cpu_kv_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for chunk in self._archived_chunks
            for tensor in (chunk.k8, chunk.k_scale, chunk.v8, chunk.v_scale, chunk.valid_counts)
            if tensor is not None and tensor.device.type == "cpu"
        )

    @property
    def persistent_bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for chunk in self.history_chunks
            for tensor in (chunk.k8, chunk.k_scale, chunk.v8, chunk.v_scale, chunk.valid_counts,
                           chunk.draft_k, chunk.draft_valid)
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
        draft_k: torch.Tensor | None = None,
        draft_valid: torch.Tensor | None = None,
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
        if (draft_k is None) != (draft_valid is None):
            raise ValueError("draft_k and draft_valid must be supplied together")
        if draft_k is not None:
            if draft_k.dtype != torch.bfloat16 or draft_k.ndim != 4:
                raise TypeError("draft_k must be contiguous BF16 BHKD")
            if draft_valid.dtype != torch.int32 or draft_valid.ndim != 1:
                raise TypeError("draft_valid must be contiguous INT32")
            if draft_k.size(2) != k8.size(2) // 64 or draft_valid.numel() != draft_k.size(2):
                raise ValueError("Draft-K blocks must align with packed K blocks")
            if not draft_k.is_contiguous() or not draft_valid.is_contiguous():
                raise ValueError("Draft-K metadata must be contiguous")
        chunk = Anemoi8BitChunk(
            self._next_chunk_id,
            k8,
            k_scale,
            v8,
            v_scale,
            int(valid_tokens),
            int(self.chunk_count * valid_tokens if start_token is None else start_token),
            valid_counts,
            draft_k,
            draft_valid,
        )
        self._next_chunk_id += 1
        self.chunks.append(chunk)
        if len(self.chunks) > self.max_chunks:
            evicted = self.chunks[:-self.max_chunks]
            self._archived_chunks.extend(
                Anemoi8BitChunk(
                    c.chunk_id, c.k8.cpu(), c.k_scale.cpu(), c.v8.cpu(), c.v_scale.cpu(),
                    c.valid_tokens, c.start_token,
                    c.valid_counts.cpu() if c.valid_counts is not None else None,
                    c.draft_k, c.draft_valid,
                )
                for c in evicted
            )
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
        draft_k: torch.Tensor | None = None,
        draft_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        if not self.chunks or self.chunks[-1].start_token != int(start_token):
            raise ValueError("replace_last requires the matching final chunk")
        self.chunks.pop()
        self._next_chunk_id -= 1
        return self.append(
            k8, k_scale, v8, v_scale,
            valid_tokens=valid_tokens, start_token=start_token,
            valid_counts=valid_counts, draft_k=draft_k, draft_valid=draft_valid,
        )

    def clear(self) -> None:
        self.chunks.clear()
        self._archived_chunks.clear()


__all__ = ["Anemoi8BitChunk", "PersistentAnemoi8BitCache"]
