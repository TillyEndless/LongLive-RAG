"""Small, query-dependent DraftMap utilities for persistent KV attention."""

from __future__ import annotations

import math

import torch


def pool_draft_tokens(tokens: torch.Tensor, block_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool BTHD tokens into B H blocks D, retaining valid lengths."""
    if tokens.ndim != 4 or block_size <= 0:
        raise ValueError("tokens must be BTHD and block_size must be positive")
    batch, length, heads, dim = tokens.shape
    blocks = math.ceil(length / block_size)
    padded = blocks * block_size - length
    if padded:
        tokens = torch.nn.functional.pad(tokens, (0, 0, 0, 0, 0, padded))
    tokens = tokens.reshape(batch, blocks, block_size, heads, dim)
    valid = torch.clamp(
        torch.arange(blocks, device=tokens.device) * block_size + block_size,
        max=length,
    ) - torch.arange(blocks, device=tokens.device) * block_size
    pooled = tokens.sum(dim=2) / valid.to(tokens.dtype).view(1, blocks, 1, 1)
    return pooled.permute(0, 2, 1, 3).contiguous(), valid.to(torch.int32)


def route_draftmap(
    query: torch.Tensor,
    draft_k: torch.Tensor,
    *,
    high_ratio: float = 0.0,
    eight_ratio: float = 1.0,
    four_ratio: float = 0.0,
    zero_ratio: float = 0.0,
) -> torch.Tensor:
    """Return per-Q-block route codes: 0=skip, 1=eight, 2=four, 3=high."""
    if query.ndim != 4 or draft_k.ndim != 4:
        raise ValueError("query and draft_k must be BTHD and BHKD")
    ratios = (high_ratio, eight_ratio, four_ratio, zero_ratio)
    if any(r < 0 or r > 1 for r in ratios) or sum(ratios) > 1.000001:
        raise ValueError("routing ratios must be non-negative and sum to at most one")
    q_pool, _ = pool_draft_tokens(query)
    scores = torch.einsum("bhqd,bhkd->bhqk", q_pool.float(), draft_k.float()) / math.sqrt(query.size(-1))
    order = scores.argsort(dim=-1, descending=True, stable=True)
    blocks = draft_k.size(2)
    ratios_with_zero = (high_ratio, eight_ratio, four_ratio, zero_ratio)
    quotas = [int(math.floor(r * blocks)) for r in ratios_with_zero]
    remainder = blocks - sum(quotas)
    for index, ratio in sorted(enumerate(ratios_with_zero), key=lambda item: -item[1]):
        if remainder == 0:
            break
        quotas[index] += 1
        remainder -= 1
    route = torch.zeros_like(order, dtype=torch.int8)
    cursor = 0
    for code, quota in ((3, quotas[0]), (1, quotas[1]), (2, quotas[2])):
        if quota:
            route.scatter_(-1, order[..., cursor:cursor + quota], torch.full_like(order[..., cursor:cursor + quota], code, dtype=torch.int8))
            cursor += quota
    return route


__all__ = ["pool_draft_tokens", "route_draftmap"]
