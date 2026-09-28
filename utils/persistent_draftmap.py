"""Portable DraftMap routing adapter for corrected H200 Group15.
Uses BF16 pooled Q/K scores and deterministic global block top-k. It gathers
retained BF16 K/V blocks before the unchanged BF16 attention call.
"""
from __future__ import annotations
import math
import torch

ROUTING_IMPLEMENTATION_ID = "anemoi.routing.draft_probability+stable_global_topk_h200_adapter_v1"
DRAFT_BLOCK_SIZE = 64
Q_POOLING = "mean over 64-token blocks"
K_POOLING = "mean over 64-token blocks"
DRAFTMAP_SCORE_FORMULA = "softmax((mean(Q) dot mean(K)) / sqrt(D), over historical blocks), mean over query blocks"
NORMALIZATION = "row softmax then query-block mean"
ROUTE_QUOTA_POLICY = "retain ceil(retained_ratio*K) global historical blocks"
TIE_BREAK_POLICY = "stable descending score, lower block id first"
BLOCK_INDEXING_POLICY = "zero-based historical KV block order"

def _pool(x):
    b,t,h,d=x.shape
    n=math.ceil(t/DRAFT_BLOCK_SIZE)
    pad=n*DRAFT_BLOCK_SIZE-t
    if pad:
        x=torch.nn.functional.pad(x,(0,0,0,0,0,pad))
    return x.reshape(b,n,DRAFT_BLOCK_SIZE,h,d).float().mean(2)

def route_draftmap(q,k,v,retained_ratio):
    """Route while retaining the configured fraction of historical blocks."""
    if not 0.0 < float(retained_ratio) < 1.0:
        return k,v,{"SPARSE_ROUTING_ACTIVE":"NO","SPARSE_EXECUTION_STATUS":"NOT_REQUESTED"}
    b,kt,h,d=k.shape
    kp=_pool(k)
    qp=_pool(q)
    logits=torch.einsum("bqhd,bkhd->bqhk",qp,kp)/math.sqrt(d)
    prob=torch.softmax(logits,dim=-1).mean(dim=(1,2))
    blocks=kp.shape[1]
    keep=max(1,math.ceil(float(retained_ratio)*blocks))
    ids=torch.argsort(prob,dim=-1,descending=True,stable=True)[:,:keep].sort(dim=-1).values
    selected_scores = torch.gather(prob, 1, ids)
    token_ids=(ids[:,:,None]*DRAFT_BLOCK_SIZE+torch.arange(DRAFT_BLOCK_SIZE,device=k.device)[None,None,:]).reshape(b,-1)
    token_ids=token_ids.clamp_max(kt-1)
    gather=token_ids[:,:,None,None].expand(-1,-1,h,d)
    ko=torch.gather(k,1,gather); vo=torch.gather(v,1,gather)
    total=int(q.shape[1]*kt); retained=int(q.shape[1]*ko.shape[1])
    return ko,vo,{
      "SPARSE_ROUTING_ACTIVE":"YES",
      "SPARSE_EXECUTION_STATUS":"REAL_SPARSE_EXECUTION",
      "ROUTING_IMPLEMENTATION_ID":ROUTING_IMPLEMENTATION_ID,
      "DRAFT_BLOCK_SIZE":DRAFT_BLOCK_SIZE,
      "Q_POOLING":Q_POOLING,"K_POOLING":K_POOLING,
      "DRAFTMAP_SCORE_FORMULA":DRAFTMAP_SCORE_FORMULA,
      "NORMALIZATION":NORMALIZATION,"ROUTE_QUOTA_POLICY":ROUTE_QUOTA_POLICY,
      "TIE_BREAK_POLICY":TIE_BREAK_POLICY,"BLOCK_INDEXING_POLICY":BLOCK_INDEXING_POLICY,
      "SPARSITY_TARGET":float(retained_ratio),
      "ROUTE_BLOCKS_TOTAL":blocks,"ROUTE_BLOCKS_RETAINED":keep,
      # Lightweight provenance trace: block IDs and their DraftMap scores.
      # This is not an attention matrix and does not affect routing.
      "ROUTE_SELECTED_BLOCK_IDS": ids.detach().cpu().tolist(),
      "ROUTE_SELECTED_BLOCK_SCORES": selected_scores.detach().float().cpu().tolist(),
      "TOTAL_INTERACTIONS":total,"RETAINED_INTERACTIONS":retained,
      "SKIPPED_INTERACTIONS":total-retained,
      "ACTUAL_ZERO_FRACTION":1.0-retained/max(total,1),
      "ACTUAL_BF16_FRACTION":retained/max(total,1),
    }


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
    offsets = torch.arange(blocks, device=tokens.device) * block_size
    valid = torch.clamp(offsets + block_size, max=length) - offsets
    pooled = tokens.sum(dim=2) / valid.to(tokens.dtype).view(1, blocks, 1, 1)
    return pooled.permute(0, 2, 1, 3).contiguous(), valid.to(torch.int32)


def route_draftmap_codes(
    query: torch.Tensor,
    draft_k: torch.Tensor,
    *,
    high_ratio: float = 0.0,
    eight_ratio: float = 1.0,
    four_ratio: float = 0.0,
    zero_ratio: float = 0.0,
) -> torch.Tensor:
    """Return native route codes: 0=skip, 1=eight-bit, 2=four-bit, 3=high."""
    if query.ndim != 4 or draft_k.ndim != 4:
        raise ValueError("query and draft_k must be BTHD and BHKD")
    ratios = (float(high_ratio), float(eight_ratio), float(four_ratio), float(zero_ratio))
    if any(r < 0.0 or r > 1.0 for r in ratios) or sum(ratios) > 1.000001:
        raise ValueError("routing ratios must be non-negative and sum to at most one")
    q_pool, _ = pool_draft_tokens(query)
    scores = torch.einsum("bhqd,bhkd->bhqk", q_pool.float(), draft_k.float()) / math.sqrt(query.size(-1))
    order = scores.argsort(dim=-1, descending=True, stable=True)
    blocks = draft_k.size(2)
    quotas = [int(math.floor(r * blocks)) for r in ratios]
    remainder = blocks - sum(quotas)
    for index, ratio in sorted(enumerate(ratios), key=lambda item: -item[1]):
        if remainder == 0:
            break
        quotas[index] += 1
        remainder -= 1
    route = torch.zeros_like(order, dtype=torch.int8)
    cursor = 0
    for code, quota in ((3, quotas[0]), (1, quotas[1]), (2, quotas[2])):
        if quota:
            selected = order[..., cursor:cursor + quota]
            route.scatter_(-1, selected, torch.full_like(selected, code, dtype=torch.int8))
            cursor += quota
    return route


__all__ = ["pool_draft_tokens", "route_draftmap", "route_draftmap_codes"]
