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



def route_candidate_chunks(q, k, v, drop_ratio, chunk_tokens,
                           mandatory_chunk_ids=()):
    """Route the complete local+retrieved candidate set at chunk granularity.

    drop_ratio is the fraction removed. Returned tensors preserve chunk order;
    mandatory chunks (normally current) always consume quota. Promotion is
    intentionally applied by the caller afterwards.
    """
    if not 0.0 < float(drop_ratio) < 1.0:
        n = int(math.ceil(k.shape[1] / max(1, int(chunk_tokens))))
        return k, v, {"SPARSE_ROUTING_ACTIVE": "NO", "ROUTE_BLOCKS_TOTAL": n,
                       "ROUTE_BLOCKS_RETAINED": n,
                       "ROUTE_SELECTED_CHUNK_IDS": list(range(n)),
                       "ROUTE_DROPPED_CHUNK_IDS": [],
                       "ROUTE_CHUNK_SCORES": [],
                       "SPARSE_SCOPE": "local_plus_retrieved_history"}
    chunk_tokens = int(chunk_tokens)
    if chunk_tokens <= 0 or k.shape[1] % chunk_tokens != 0:
        raise ValueError("candidate KV must contain complete chunks")
    b, tokens, h, d = k.shape
    blocks = tokens // chunk_tokens
    if blocks <= 0:
        return k, v, {"SPARSE_ROUTING_ACTIVE": "NO", "ROUTE_BLOCKS_TOTAL": 0,
                       "ROUTE_BLOCKS_RETAINED": 0,
                       "ROUTE_SELECTED_CHUNK_IDS": [],
                       "ROUTE_DROPPED_CHUNK_IDS": [],
                       "ROUTE_CHUNK_SCORES": [],
                       "SPARSE_SCOPE": "local_plus_retrieved_history"}
    qp = q.float().mean(dim=1)
    kp = k.view(b, blocks, chunk_tokens, h, d).float().mean(dim=2)
    scores = torch.einsum("bhd,bkhd->bk", qp, kp).mean(dim=0)
    keep = max(1, int(math.ceil((1.0 - float(drop_ratio)) * blocks)))
    mandatory = {int(i) for i in mandatory_chunk_ids if 0 <= int(i) < blocks}
    keep = max(keep, len(mandatory))
    order = torch.argsort(scores, descending=True, stable=True)
    selected = list(sorted(mandatory))
    for idx in order.detach().cpu().tolist():
        if int(idx) not in mandatory:
            selected.append(int(idx))
        if len(selected) >= keep:
            break
    selected = sorted(selected[:keep])
    ids = torch.tensor(selected, device=k.device, dtype=torch.long)
    starts = ids[:, None] * chunk_tokens + torch.arange(chunk_tokens, device=k.device)[None, :]
    token_ids = starts.reshape(1, -1).expand(b, -1)
    gather = token_ids[:, :, None, None].expand(-1, -1, h, d)
    ko = torch.gather(k, 1, gather)
    vo = torch.gather(v, 1, gather)
    selected_scores = scores[ids]
    return ko, vo, {
        "SPARSE_ROUTING_ACTIVE": "YES",
        "SPARSE_EXECUTION_STATUS": "REAL_SPARSE_EXECUTION",
        "ROUTING_IMPLEMENTATION_ID": ROUTING_IMPLEMENTATION_ID,
        "DRAFT_BLOCK_SIZE": chunk_tokens,
        "SPARSE_SCOPE": "local_plus_retrieved_history",
        "ROUTE_QUOTA_POLICY": "ceil((1-q_sparse_ratio)*candidate_chunks)",
        "CURRENT_CHUNK_MANDATORY": True,
        "ROUTE_BLOCKS_TOTAL": blocks,
        "ROUTE_BLOCKS_RETAINED": len(selected),
        "ROUTE_SELECTED_CHUNK_IDS": selected,
        "ROUTE_DROPPED_CHUNK_IDS": [i for i in range(blocks) if i not in selected],
        "ROUTE_SELECTED_CHUNK_SCORES": selected_scores.detach().float().cpu().tolist(),
        "ROUTE_CHUNK_SCORES": [float(x) for x in scores.detach().cpu().tolist()],
        "TOTAL_INTERACTIONS": int(q.shape[1] * tokens),
        "RETAINED_INTERACTIONS": int(q.shape[1] * ko.shape[1]),
        "SKIPPED_INTERACTIONS": int(q.shape[1] * (tokens - ko.shape[1])),
        "ACTUAL_ZERO_FRACTION": 1.0 - ko.shape[1] / max(tokens, 1),
        "ACTUAL_DENSE_FRACTION": ko.shape[1] / max(tokens, 1),
    }
