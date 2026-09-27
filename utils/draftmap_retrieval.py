"""BF16 DraftMap descriptors for the Group 11 retriever experiment.

Only historical Draft-K is retained. Current Draft-Q is a transient scoring
input and is never stored in a history record.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class DraftChunkRecord:
    chunk_id: str
    draft_k: torch.Tensor
    block_ids: tuple[int, ...]


class DraftMapChunkIndex:
    """Persist BF16 historical Draft-K beside canonical history identities."""

    def __init__(self, block_tokens: int = 64):
        if type(block_tokens) is not int or block_tokens <= 0:
            raise ValueError("block_tokens must be a positive integer")
        self.block_tokens = block_tokens
        self.records: list[DraftChunkRecord] = []

    def register_chunk(self, chunk_id: str, draft_k: torch.Tensor) -> None:
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("chunk_id must be a non-empty string")
        if draft_k.dtype != torch.bfloat16:
            raise TypeError("Group 11 historical Draft-K storage must be BF16")
        if draft_k.ndim != 4:
            raise ValueError("draft K must have [B,T,H,D] shape")
        if draft_k.size(1) == 0 or draft_k.size(1) % self.block_tokens:
            raise ValueError("chunk token count must be block aligned")
        blocks = draft_k.size(1) // self.block_tokens
        pooled_k = self.pooled_blocks(draft_k).permute(0, 2, 1, 3).contiguous()
        self.records.append(DraftChunkRecord(
            chunk_id=chunk_id,
            draft_k=pooled_k.detach(),
            block_ids=tuple(range(blocks)),
        ))

    def pooled_blocks(self, tensor: torch.Tensor) -> torch.Tensor:
        """Use Anemoi's mean-pool geometry without low-bit preparation."""
        if tensor.dtype != torch.bfloat16 or tensor.ndim != 4:
            raise ValueError("DraftMap tensors must be BF16 [B,T,H,D]")
        blocks_count = math.ceil(tensor.size(1) / self.block_tokens)
        padded = blocks_count * self.block_tokens - tensor.size(1)
        if padded:
            tensor = torch.cat(
                (tensor, tensor.new_zeros(tensor.size(0), padded, tensor.size(2), tensor.size(3))),
                dim=1,
            )
        blocks = tensor.view(
            tensor.size(0), blocks_count,
            self.block_tokens, tensor.size(2), tensor.size(3)
        )
        counts = torch.full((blocks_count,), self.block_tokens, device=tensor.device, dtype=torch.float32)
        if padded:
            counts[-1] = self.block_tokens - padded
        return blocks.float().sum(dim=2).div_(counts.view(1, -1, 1, 1)).to(torch.bfloat16).contiguous()

    def score_blocks(
        self,
        current_draft_q: torch.Tensor,
        records: list[DraftChunkRecord] | None = None,
    ) -> torch.Tensor:
        """Return Anemoi-style row-softmax scores over historical K blocks."""
        if records is None:
            records = self.records
        if not records:
            return torch.empty((*current_draft_q.shape[:1], current_draft_q.shape[2], 0, 0), device=current_draft_q.device)
        q_pool = (
            self.pooled_blocks(current_draft_q).permute(0, 2, 1, 3)
            if current_draft_q.ndim == 4 and current_draft_q.size(1) != current_draft_q.size(2)
            else current_draft_q
        )
        k_pool = torch.cat([r.draft_k for r in records], dim=2)
        if q_pool.size(1) != k_pool.size(1):
            if q_pool.size(1) % k_pool.size(1):
                raise ValueError("query heads must be divisible by key heads")
            k_pool = k_pool.repeat_interleave(q_pool.size(1) // k_pool.size(1), dim=1)
        logits = torch.matmul(q_pool.float(), k_pool.float().transpose(-2, -1))
        logits.mul_(1.0 / math.sqrt(q_pool.size(-1)))
        return torch.softmax(logits, dim=-1).to(torch.float16)

    def aggregate_chunks(self, block_scores: torch.Tensor, records=None):
        if records is None:
            records = self.records
        if block_scores.ndim != 4:
            raise ValueError("block_scores must be [B,H,Q_blocks,K_blocks]")
        if sum(len(r.block_ids) for r in records) != block_scores.size(-1):
            raise ValueError("block_scores and history block mapping disagree")
        chunks = []
        offset = 0
        for record in records:
            width = len(record.block_ids)
            chunks.append(block_scores[..., offset:offset + width].sum(dim=-1))
            offset += width
        return torch.stack(chunks, dim=-1).mean(dim=(1, 2))

    def score_history(self, current_draft_q, records=None):
        records = self.records if records is None else records
        return self.aggregate_chunks(self.score_blocks(current_draft_q, records), records)

    @staticmethod
    def select_topk(scores: torch.Tensor, k: int):
        if scores.ndim != 2 or k <= 0:
            raise ValueError("scores must be [B, candidates] and k positive")
        return torch.topk(scores, k=min(k, scores.size(-1)), dim=-1)

    @property
    def metadata_bytes(self) -> int:
        return sum(len(r.chunk_id.encode()) + len(r.block_ids) * 8 for r in self.records)


__all__ = ["DraftChunkRecord", "DraftMapChunkIndex"]
