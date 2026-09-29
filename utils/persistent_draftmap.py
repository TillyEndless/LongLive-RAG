"""Portable DraftMap routing adapter for corrected H200 Group15.
Uses BF16 pooled Q/K scores and deterministic global block top-k. It gathers
retained BF16 K/V blocks before the unchanged BF16 attention call.
"""
from __future__ import annotations
import math
import os
import torch
import utils.critical_path_trace as cpt

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
    trace=cpt.ACTIVE_TRACE
    kp=(trace.measure('DRAFT_K_POOL', lambda: _pool(k)) if trace else _pool(k))
    qp=(trace.measure('DRAFT_Q_POOL', lambda: _pool(q)) if trace else _pool(q))
    logits=(trace.measure('DRAFT_SCORE_MATMUL', lambda: torch.einsum("bqhd,bkhd->bqhk",qp,kp)/math.sqrt(d)) if trace else torch.einsum("bqhd,bkhd->bqhk",qp,kp)/math.sqrt(d))
    probs=(trace.measure('DRAFT_SCORE_SOFTMAX', lambda: torch.softmax(logits,dim=-1)) if trace else torch.softmax(logits,dim=-1))
    prob=(trace.measure('IMPORTANCE_REDUCTION', lambda: probs.mean(dim=(1,2))) if trace else probs.mean(dim=(1,2)))
    blocks=kp.shape[1]
    keep=max(1,math.ceil(float(retained_ratio)*blocks))
    ranked=(trace.measure('TOPK_SELECTION', lambda: torch.argsort(prob,dim=-1,descending=True,stable=True)[:,:keep]) if trace else torch.argsort(prob,dim=-1,descending=True,stable=True)[:,:keep])
    ids=(trace.measure('SELECTED_HISTORY_ID_CONSTRUCTION', lambda: ranked.sort(dim=-1).values) if trace else ranked.sort(dim=-1).values)
    selected_scores = torch.gather(prob, 1, ids)
    token_ids=(trace.measure('TOKEN_ID_EXPANSION', lambda: (ids[:,:,None]*DRAFT_BLOCK_SIZE+torch.arange(DRAFT_BLOCK_SIZE,device=k.device)[None,None,:]).reshape(b,-1)) if trace else (ids[:,:,None]*DRAFT_BLOCK_SIZE+torch.arange(DRAFT_BLOCK_SIZE,device=k.device)[None,None,:]).reshape(b,-1))
    token_ids=(trace.measure('TOKEN_ID_CLAMP', lambda: token_ids.clamp_max(kt-1)) if trace else token_ids.clamp_max(kt-1))
    gather=token_ids[:,:,None,None].expand(-1,-1,h,d)
    ko=(trace.measure('GATHER_K', lambda: torch.gather(k,1,gather)) if trace else torch.gather(k,1,gather))
    vo=(trace.measure('GATHER_V', lambda: torch.gather(v,1,gather)) if trace else torch.gather(v,1,gather))
    gpu_only_metadata = os.environ.get("GROUP14_GPU_ONLY_ROUTING", "0") == "1"
    if trace and not gpu_only_metadata:
        ids_cpu = trace.measure('CPU_DETACH_CPU_IDS', lambda: ids.detach().cpu())
        ids_list = trace.measure('CPU_TOLIST_IDS', lambda: ids_cpu.tolist())
        scores_cpu = trace.measure('CPU_DETACH_CPU_SCORES', lambda: selected_scores.detach().float().cpu())
        scores_list = trace.measure('CPU_TOLIST_SCORES', lambda: scores_cpu.tolist())
    elif gpu_only_metadata:
        ids_list = "GPU_ONLY_PROVENANCE_OMITTED"
        scores_list = "GPU_ONLY_PROVENANCE_OMITTED"
    else:
        ids_list = ids.detach().cpu().tolist()
        scores_list = selected_scores.detach().float().cpu().tolist()
    total=int(q.shape[1]*kt); retained=int(q.shape[1]*ko.shape[1])
    metadata = {
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
      "ROUTE_SELECTED_BLOCK_IDS": ids_list,
      "ROUTE_SELECTED_BLOCK_SCORES": scores_list,
      "TOTAL_INTERACTIONS":total,"RETAINED_INTERACTIONS":retained,
      "SKIPPED_INTERACTIONS":total-retained,
      "ACTUAL_ZERO_FRACTION":1.0-retained/max(total,1),
      "ACTUAL_BF16_FRACTION":retained/max(total,1),
      "GPU_ONLY_ROUTING_FLAG": "YES" if gpu_only_metadata else "NO",
    }
    if trace:
        metadata = trace.measure('ROUTING_METADATA_PYTHON', lambda: metadata)
    return ko,vo,metadata
