"""Persistent native Anemoi NVFP4 historical KV storage."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Anemoi4BitChunk:
    chunk_id: int
    k4: torch.Tensor
    k_scale: torch.Tensor
    v4: torch.Tensor
    v_scale: torch.Tensor
    valid_counts: torch.Tensor
    valid_tokens: int
    start_token: int
    draft_k: torch.Tensor | None = None
    draft_valid: torch.Tensor | None = None


class PersistentAnemoi4BitCache:
    def __init__(self, max_chunks: int):
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")
        self.max_chunks = int(max_chunks)
        self.chunks: list[Anemoi4BitChunk] = []
        self._archived_chunks: list[Anemoi4BitChunk] = []
        self._next_chunk_id = 0

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)

    @property
    def persistent_bf16_bytes(self) -> int:
        return 0

    @property
    def archived_chunk_ids(self) -> list[int]:
        return [chunk.chunk_id for chunk in self._archived_chunks]

    @property
    def chunk_ids(self) -> list[int]:
        return [chunk.chunk_id for chunk in self.chunks]

    @property
    def next_chunk_id(self) -> int:
        return self._next_chunk_id

    @property
    def history_chunks(self) -> list[Anemoi4BitChunk]:
        return [*self._archived_chunks, *self.chunks]

    @property
    def persistent_draft_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for c in self.history_chunks for t in (c.draft_k, c.draft_valid)
            if t is not None
        )

    @property
    def persistent_gpu_draft_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for c in self.history_chunks for t in (c.draft_k, c.draft_valid)
            if t is not None and t.device.type == "cuda"
        )

    @property
    def persistent_cpu_kv_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for c in self._archived_chunks
            for t in (c.k4, c.k_scale, c.v4, c.v_scale, c.valid_counts)
            if t is not None and t.device.type == "cpu"
        )

    @property
    def persistent_gpu_kv_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for c in self.history_chunks
            for t in (c.k4, c.k_scale, c.v4, c.v_scale, c.valid_counts)
            if t is not None and t.device.type == "cuda"
        )

    @property
    def persistent_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for c in self.history_chunks
            for t in (c.k4, c.k_scale, c.v4, c.v_scale, c.valid_counts,
                      c.draft_k, c.draft_valid)
            if t is not None
        )

    def append(self, chunk: Anemoi4BitChunk) -> None:
        if chunk.chunk_id != self._next_chunk_id:
            raise ValueError("chunk ids must be append-only")
        if any(t.dtype != torch.uint8 for t in (chunk.k4, chunk.k_scale, chunk.v4, chunk.v_scale)):
            raise TypeError("NVFP4 payloads and scales must be uint8")
        if chunk.valid_counts.dtype != torch.int32:
            raise TypeError("valid_counts must be int32")
        if (chunk.draft_k is None) != (chunk.draft_valid is None):
            raise ValueError("draft_k and draft_valid must be supplied together")
        if chunk.draft_k is not None and (
            chunk.draft_k.dtype != torch.bfloat16 or
            chunk.draft_valid.dtype != torch.int32
        ):
            raise TypeError("Draft metadata must be BF16/INT32")
        if not all(t.is_cuda and t.is_contiguous() for t in (
            chunk.k4, chunk.k_scale, chunk.v4, chunk.v_scale,
            chunk.valid_counts,
        )):
            raise ValueError("persistent NVFP4 tensors must be contiguous CUDA tensors")
        if chunk.start_token < 0 or chunk.valid_tokens <= 0:
            raise ValueError("chunk metadata must describe a positive history span")
        if chunk.valid_counts.numel() != chunk.k4.size(2) // 64:
            raise ValueError("valid_counts must align with packed K64 blocks")
        self.chunks.append(chunk)
        self._next_chunk_id += 1
        if len(self.chunks) > self.max_chunks:
            evicted = self.chunks[:-self.max_chunks]
            self._archived_chunks.extend(
                Anemoi4BitChunk(
                    c.chunk_id, c.k4.cpu(), c.k_scale.cpu(), c.v4.cpu(), c.v_scale.cpu(),
                    c.valid_counts.cpu(), c.valid_tokens, c.start_token,
                    c.draft_k, c.draft_valid,
                )
                for c in evicted
            )
            del self.chunks[:-self.max_chunks]

    def replace_last(self, chunk: Anemoi4BitChunk) -> None:
        if not self.chunks or self.chunks[-1].start_token != chunk.start_token:
            raise ValueError("replace_last requires the matching final chunk")
        self.chunks.pop()
        self._next_chunk_id -= 1
        self.append(
            Anemoi4BitChunk(
                chunk_id=self._next_chunk_id,
                k4=chunk.k4,
                k_scale=chunk.k_scale,
                v4=chunk.v4,
                v_scale=chunk.v_scale,
                valid_counts=chunk.valid_counts,
                valid_tokens=chunk.valid_tokens,
                start_token=chunk.start_token,
                draft_k=chunk.draft_k,
                draft_valid=chunk.draft_valid,
            )
        )

    def clear(self) -> None:
        self.chunks.clear()
        self._archived_chunks.clear()


__all__ = ["Anemoi4BitChunk", "PersistentAnemoi4BitCache"]
