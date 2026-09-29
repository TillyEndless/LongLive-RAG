# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
from utils.compressed_history_archive import CompressedHistoryArchive
import math
import time
import os
import torch.distributed as dist
from contextlib import nullcontext
try:
    from torch.profiler import record_function
except Exception:
    record_function = None
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, log_gpu_memory

from utils.debug_option import DEBUG
from utils.h200_group_runtime import prepare_attention_kv, select_q_sparse_history
from utils.unified_latency_profiler import UnifiedLatencyProfiler
import utils.critical_path_trace as cpt

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


def causal_online_rope(x, grid_sizes, freqs, start_frame=0, relative_frame_indices=None):
    """
    Apply causal RoPE (Rotary Position Embedding) to input tensor.
    
    Args:
        x: Input tensor of shape [B, L, num_heads, head_dim]
        grid_sizes: Tensor of shape [B, 3] containing (F, H, W)
        freqs: RoPE frequencies
        start_frame: Starting frame index for sequential RoPE (default: 0)
        relative_frame_indices: Optional tensor of shape [F] specifying explicit frame indices
                               for causal online RoPE. If provided, overrides start_frame.
    """
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        
        # Use relative_frame_indices if provided (causal online RoPE),
        # otherwise use sequential indices starting from start_frame
        if relative_frame_indices is not None:
            # relative_frame_indices should be a tensor of shape [f] with explicit frame indices
            frame_indices = relative_frame_indices.long()
            freqs_temporal = freqs[0][frame_indices].view(f, 1, 1, -1).expand(f, h, w, -1)
        else:
            freqs_temporal = freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1)
        
        freqs_i = torch.cat([
            freqs_temporal,
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def _draft_frame_pools(tensor, frame_tokens, block_tokens=64):
    """Mean-pool BF16 Q/K into Anemoi-compatible per-frame blocks."""
    frames = tensor.size(1) // frame_tokens
    outputs = []
    for frame in range(frames):
        value = tensor[:, frame * frame_tokens:(frame + 1) * frame_tokens]
        blocks = math.ceil(frame_tokens / block_tokens)
        padded = blocks * block_tokens - frame_tokens
        if padded:
            value = torch.cat((value, value.new_zeros(value.size(0), padded, value.size(2), value.size(3))), dim=1)
        counts = torch.full((blocks,), block_tokens, device=value.device, dtype=torch.float32)
        if padded:
            counts[-1] = frame_tokens % block_tokens
        pooled = value.view(value.size(0), blocks, block_tokens, value.size(2), value.size(3)).float().sum(dim=2)
        pooled = pooled.div_(counts.view(1, blocks, 1, 1))
        outputs.append(pooled.permute(0, 2, 1, 3).to(torch.bfloat16).contiguous())
    return outputs



class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 memory_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.memory_size = memory_size
        self.retrieval_backend = "original"
        self.retrieval_query_mode = "current_q"
        self.recent_exclude = 0
        self.draftmap_trace = []
        self.runtime_counters = {
            "draft_h2d_bytes": 0,
            "draft_h2d_calls": 0,
            "full_kv_h2d_bytes": 0,
            "full_kv_h2d_calls": 0,
        }
        self.history_fetch_trace = []
        self.current_denoising_step = None
        self.group_runtime_mode = "baseline"
        self.group_sparse_ratio = 0.0
        self.q_sparse_ratio = 0.0
        self.promotion_ratio = 0.0
        self.promotion_source = "draftmap"
        self.materialization_mode = "serial_reference"
        self.group_runtime_trace = []
        self.group11_fetch_mode = "serial_full"
        self.flash_fetch_mode = "serial_reference"
        self._flash_fetch_state = {}
        self._flash_pinned_sources = {}
        self._flash_event_records = []
        self.group11_flash_trace = []
        self._rag_fetch_event_id = 0
        self._rag_cuda_pending = []
        # Group11.4-only transient prefetch pointer. Group12-15 do not use it.
        self.prefetch_kv_cache = None
        self.group11_profile = None
        self.unified_latency_profiler = None
        self._group11_reuse_cache = {}
        self.qk_norm = qk_norm
        self.eps = eps
        # Support list/tuple local_attn_size by converting to list first (handles OmegaConf ListConfig)
        if not isinstance(local_attn_size, int) and hasattr(local_attn_size, "__iter__"):
            values = list(local_attn_size)
        else:
            values = [int(local_attn_size)]
        non_neg_vals = [int(v) for v in values if int(v) != -1]
        max_local = max(non_neg_vals) if len(non_neg_vals) > 0 else -1
        self.max_attention_size = 32760 if max_local == -1 else max_local * 1560
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def _rag_profile_on(self):
        return self.group11_profile is not None and os.environ.get("RAG_STRATEGY_PROFILE", "0") == "1"

    def _finalize_rag_cuda_events(self):
        """Resolve already-recorded CUDA event pairs after the existing run barrier."""
        if not self._rag_profile_on():
            return
        rows = self.group11_profile.setdefault("rag_cuda_measurements", [])
        event_by_id = {int(e.get("fetch_event_id", -1)): e
                       for e in self.group11_profile.get("rag_fetch_events", [])}
        prefetch_by_key = {}
        for e in self.group11_profile.get("rag_fetch_events", []):
            if e.get("fetch_reason") == "prefetch":
                prefetch_by_key.setdefault((int(e.get("layer_id", -1)), int(e.get("history_id", -1))), []).append(e)
        for item in self._rag_cuda_pending:
            try:
                ms = float(item["start"].elapsed_time(item["end"]))
            except Exception:
                ms = None
            row = {k: v for k, v in item.items() if k not in {"start", "end"}}
            row["cuda_ms"] = ms
            rows.append(row)
            event_id = row.get("fetch_event_id")
            if event_id is None and row.get("measurement_kind") == "prefetch_wait":
                candidates = [e for e in prefetch_by_key.get((int(row.get("layer_id", -2)), int(row.get("history_id", -2))), [])
                              if "prefetch_exposed_wait_ms" not in e]
                if candidates:
                    event_id = candidates[-1].get("fetch_event_id")
            if event_id is not None:
                event = event_by_id.get(int(event_id))
                if event is not None:
                    kind = row.get("measurement_kind")
                    if kind == "prefetch_copy":
                        event["prefetch_cuda_work_ms"] = ms
                    elif kind == "prefetch_wait":
                        event["prefetch_exposed_wait_ms"] = ms
                        event["prefetch_hit_status"] = "hit"
                    elif kind == "correction_copy":
                        event["correction_cuda_work_ms"] = ms
                        event["correction_exposed_ms"] = ms
        self._rag_cuda_pending = []

    def _rag_record_fetch(self, *, layer_id, attention_call_id, history_id,
                          fetch_reason, fetch_bytes_k, fetch_bytes_v,
                          fetch_work_ms, fetch_exposed_ms=None,
                          is_first_fetch_in_call=False, is_duplicate=False,
                          is_cache_hit=False, is_cache_miss=False,
                          prefetch_hit_status="not_applicable", source_chunk_id=None,
                          block_id=None):
        if not self._rag_profile_on():
            return
        exposed = float(fetch_work_ms if fetch_exposed_ms is None else fetch_exposed_ms)
        work = float(fetch_work_ms)
        self._rag_fetch_event_id += 1
        self.group11_profile["rag_fetch_events"].append({
            "case_id": os.environ.get("RAG_PROFILE_CASE_ID", "case_01"),
            "step_id": self.current_denoising_step,
            "layer_id": int(layer_id), "attention_call_id": int(attention_call_id),
            "fetch_event_id": int(self._rag_fetch_event_id),
            "history_id": int(history_id),
            "source_chunk_id": int(history_id if source_chunk_id is None else source_chunk_id),
            "block_id": int(history_id if block_id is None else block_id),
            "fetch_reason": str(fetch_reason),
            "fetch_bytes_k": int(fetch_bytes_k), "fetch_bytes_v": int(fetch_bytes_v),
            "fetch_bytes_total": int(fetch_bytes_k + fetch_bytes_v),
            "fetch_work_ms": work, "fetch_exposed_ms": exposed,
            "fetch_hidden_ms": max(0.0, work - exposed),
            "is_first_fetch_in_call": bool(is_first_fetch_in_call),
            "is_duplicate_fetch": bool(is_duplicate),
            "is_cache_hit": bool(is_cache_hit), "is_cache_miss": bool(is_cache_miss),
            "prefetch_hit_status": str(prefetch_hit_status),
        })

    def _draftmap_select_for_query(self, retrieval_query, kv_cache, layer_index,
                                   frame_seqlen, current_start, query_source,
                                   target_invocation_id=None, bootstrap=False):
        """Run one DraftMap selection and attach explicit temporal provenance.

        This helper is used only by ``previous_q_prefetch``.  It deliberately
        leaves the existing ``previous_q_direct`` implementation untouched.
        """
        compressed_entries = kv_cache.get("compressed_history_entries", [])
        cpu_k = compressed_entries if compressed_entries else kv_cache.get("cpu_k_frames", [])
        gpu_draft_k = kv_cache.get("gpu_draft_k_frames", [])
        eligible = max(len(cpu_k) - self.recent_exclude, 0)
        if eligible <= 0 or len(gpu_draft_k) < eligible:
            return None
        from utils.draftmap_retrieval import DraftChunkRecord, DraftMapChunkIndex
        query_ready_host = time.perf_counter()
        score_start = time.perf_counter()
        records = [
            DraftChunkRecord(str(i), gpu_draft_k[i], tuple(range(gpu_draft_k[i].size(2))))
            for i in range(eligible)
        ]
        index = DraftMapChunkIndex(block_tokens=64)
        if any(r.draft_k.device != retrieval_query.device for r in records):
            raise RuntimeError("persistent Draft-K must remain GPU-resident")
        scores = index.score_history(retrieval_query, records)
        score_ms = (time.perf_counter() - score_start) * 1000.0
        topk_start = time.perf_counter()
        topk_scores, selected = index.select_topk(scores, self.memory_size)
        topk_ms = (time.perf_counter() - topk_start) * 1000.0
        prediction_end_host = time.perf_counter()
        kv_cache["temporal_last_prediction_timing"] = {
            "q_ready_host": query_ready_host,
            "predict_start_host": score_start,
            "predict_end_host": prediction_end_host,
            "predict_work_ms": (prediction_end_host - score_start) * 1000.0,
        }
        selected_ids = selected[0].detach().cpu().tolist()
        row = {
            "layer": int(layer_index),
            "denoising_step": self.current_denoising_step,
            "current_chunk_id": int(kv_cache.get("global_end_index", torch.zeros(1)).item() // frame_seqlen),
            "num_historical_chunks": int(len(cpu_k)),
            "num_candidate_chunks": int(eligible),
            "num_selected_chunks": int(selected.shape[-1]),
            "draft_q_shape": list(retrieval_query.shape),
            "draft_k_shape": list(gpu_draft_k[0].shape) if gpu_draft_k else [],
            "draft_score_elements": int(scores.numel()),
            "time_draft_q_pool_ms": 0.0,
            "time_score_ms": score_ms,
            "time_aggregate_ms": 0.0,
            "time_topk_ms": topk_ms,
            "query_source": str(query_source),
            "target_invocation_id": (None if target_invocation_id is None else int(target_invocation_id)),
            "bootstrap": bool(bootstrap),
        }
        if self.group11_profile is not None:
            p = self.group11_profile
            p["NUM_DRAFTMAP_CALLS"] += 1
            p["NUM_DRAFT_Q_POOL_CALLS"] += 1
            p["NUM_DRAFT_K_SCORE_CALLS"] += 1
            p["NUM_TOPK_CALLS"] += 1
            p.setdefault("draftmap_rows", []).append(row)
        self.draftmap_trace.append({
            "generation_unit": int(kv_cache.get("global_end_index", torch.zeros(1)).item() // frame_seqlen),
            "layer": int(layer_index),
            "candidate_count": int(eligible),
            "candidate_history_ids": list(range(eligible)),
            "selected_history_ids": [int(v) for v in selected_ids],
            "selected_scores": [float(v) for v in topk_scores[0].detach().cpu()],
            "retrieval_budget": int(self.memory_size),
            "retrieval_query_mode": self.retrieval_query_mode,
            "wrapper_impl": "group11_2_temporal_qprev_prefetch",
            "retrieval_query_source": str(query_source),
            "target_invocation_id": (None if target_invocation_id is None else int(target_invocation_id)),
            "bootstrap": bool(bootstrap),
            "current_q_logical_id": {"layer": int(layer_index), "chunk": int(current_start // frame_seqlen), "denoise_step": self.current_denoising_step},
            "retrieval_q_logical_id": {"layer": int(layer_index), "chunk": int(current_start // frame_seqlen), "denoise_step": self.current_denoising_step},
            "draft_score_ms": score_ms,
            "topk_ms": topk_ms,
            "denoising_step": self.current_denoising_step,
        })
        return selected

    def _schedule_temporal_qprev_prefetch(self, kv_cache, layer_index, selected,
                                          target_valid_invocation_id,
                                          target_invocation_id, device):
        """Prefetch same-layer K/V for the next invocation on a dedicated stream."""
        if selected is None or kv_cache is None:
            return
        predictions = kv_cache.setdefault("temporal_qprev_predictions", {})
        state = kv_cache.setdefault("temporal_qprev_prefetch", {})
        slots = kv_cache.setdefault("temporal_qprev_staging_slots", {})
        stream = kv_cache.get("temporal_qprev_stream")
        if stream is None or stream.device != device:
            stream = torch.cuda.Stream(device=device)
            kv_cache["temporal_qprev_stream"] = stream
        consumer_event = kv_cache.get("temporal_qprev_consumer_event")
        if consumer_event is not None:
            stream.wait_event(consumer_event)
        compressed_entries = kv_cache.get("compressed_history_entries", [])
        cpu_k = compressed_entries if compressed_entries else kv_cache.get("cpu_k_frames", [])
        cpu_v = kv_cache.get("cpu_v_frames", [])
        host_control_start = time.perf_counter()
        ids = [int(v) for v in selected[0].detach().cpu().tolist()]
        host_control_end = time.perf_counter()
        archive_lookup_start = time.perf_counter()
        target_valid = int(target_valid_invocation_id)
        target_raw = int(target_invocation_id)
        archive_lookup_end = time.perf_counter()
        predictions[target_valid] = {
            "target_valid_invocation_id": target_valid,
            "target_invocation_id": target_raw,
            "target_layer": int(layer_index),
            "selected_history_ids": ids,
            "source": "Q_T_MINUS_1",
            "source_invocation_id": target_raw - 1,
            "source_denoising_step": self.current_denoising_step,
        }
        requested = 0
        prepare_start = time.perf_counter()
        h2d_start_event = torch.cuda.Event(enable_timing=True)
        h2d_end_event = torch.cuda.Event(enable_timing=True)
        h2d_enqueue_host = time.perf_counter()
        h2d_start_event.record(stream)
        with torch.cuda.stream(stream):
            for history_id in ids:
                if history_id >= len(cpu_k) or history_id >= len(cpu_v):
                    continue
                src_k, src_v = cpu_k[history_id], cpu_v[history_id]
                if not src_k.is_pinned() or not src_v.is_pinned():
                    raise RuntimeError("temporal prefetch source archive is not pre-pinned")
                slot = slots.get(history_id)
                if slot is None or slot[0].shape != src_k.shape or slot[1].shape != src_v.shape:
                    dst_k = torch.empty_like(src_k, device=device)
                    dst_v = torch.empty_like(src_v, device=device)
                    slots[history_id] = (dst_k, dst_v)
                    kv_cache["temporal_qprev_staging_allocations"] = int(kv_cache.get("temporal_qprev_staging_allocations", 0)) + 1
                else:
                    dst_k, dst_v = slot
                dst_k.copy_(src_k, non_blocking=True)
                dst_v.copy_(src_v, non_blocking=True)
                event = torch.cuda.Event()
                event.record(stream)
                state[(target_valid, int(layer_index), history_id)] = {
                    "k": dst_k, "v": dst_v, "event": event,
                    "target_valid_invocation_id": target_valid,
                    "target_invocation_id": target_raw,
                    "target_layer": int(layer_index),
                    "history_id": history_id,
                }
                requested += 1
        h2d_end_event.record(stream)
        prepare_end = time.perf_counter()
        event_state = kv_cache.setdefault("temporal_qprev_event_state", {})
        event_state[target_valid] = {
            "target_valid_invocation_id": target_valid,
            "target_invocation_id": target_raw,
            "target_layer": int(layer_index),
            "q_ready_host": float(kv_cache.get("temporal_last_prediction_timing", {}).get("q_ready_host", h2d_enqueue_host)),
            "predict_start_host": float(kv_cache.get("temporal_last_prediction_timing", {}).get("predict_start_host", h2d_enqueue_host)),
            "predict_end_host": float(kv_cache.get("temporal_last_prediction_timing", {}).get("predict_end_host", h2d_enqueue_host)),
            "host_control_ms": (host_control_end - host_control_start) * 1000.0,
            "archive_lookup_ms": (archive_lookup_end - archive_lookup_start) * 1000.0,
            "prefetch_prepare_ms": (prepare_end - prepare_start) * 1000.0,
            "h2d_enqueue_host": h2d_enqueue_host,
            "h2d_start_event": h2d_start_event,
            "h2d_end_event": h2d_end_event,
            "requested_ids": requested,
            "selected_ids": ids,
            "source_invocation_id": target_raw - 1,
            "target_invocation_start_host": None,
            "target_consumer_need_host": None,
            "target_wait_start_event": None,
            "target_wait_end_event": None,
            "kv_assembly_ms": None,
            "fa_start_host": None,
            "fa_end_host": None,
        }
        kv_cache["temporal_qprev_requested"] = int(kv_cache.get("temporal_qprev_requested", 0)) + requested
        if self.group11_profile is not None:
            self.group11_profile["QPREFETCH_REQUESTED_IDS"] = int(self.group11_profile.get("QPREFETCH_REQUESTED_IDS", 0)) + requested
            self.group11_profile["QPREFETCH_H2D_COUNT"] = int(self.group11_profile.get("QPREFETCH_H2D_COUNT", 0)) + requested
        kv_cache.setdefault("temporal_qprev_trace", []).append({
            "source_invocation_id": target_raw - 1,
            "target_valid_invocation_id": target_valid,
            "target_invocation_id": target_raw,
            "target_layer": int(layer_index),
            "selected_history_ids": ids,
            "requested_ids": requested,
            "launch_point": "after_current_attention_before_output_projection",
        })

    def _consume_temporal_qprev_prefetch(self, kv_cache, layer_index,
                                         valid_invocation_id, history_id, batch_index, device):
        state = kv_cache.setdefault("temporal_qprev_prefetch", {})
        key = (int(valid_invocation_id), int(layer_index), int(history_id))
        entry = state.pop(key, None)
        if entry is None:
            # Bootstrap demand fetch is expected and is not a temporal
            # prefetch miss.  Only a non-bootstrap missing entry is invalid.
            if kv_cache.get("temporal_current_selection_source") != "BOOTSTRAP_CURRENT_Q":
                kv_cache["temporal_qprev_misses"] = int(kv_cache.get("temporal_qprev_misses", 0)) + 1
                if self.group11_profile is not None:
                    self.group11_profile["QPREFETCH_MISSES"] = int(self.group11_profile.get("QPREFETCH_MISSES", 0)) + 1
            return None
        if entry.get("target_valid_invocation_id") != int(valid_invocation_id) or entry.get("target_layer") != int(layer_index):
            kv_cache["temporal_qprev_stale_rejects"] = int(kv_cache.get("temporal_qprev_stale_rejects", 0)) + 1
            if self.group11_profile is not None:
                self.group11_profile["QPREFETCH_STALE_REJECTS"] = int(self.group11_profile.get("QPREFETCH_STALE_REJECTS", 0)) + 1
            return None
        current = torch.cuda.current_stream(device)
        event_state = kv_cache.get("temporal_qprev_event_state", {}).get(int(valid_invocation_id))
        if event_state is not None:
            event_state["target_consumer_need_host"] = time.perf_counter()
            event_state["target_wait_start_event"] = torch.cuda.Event(enable_timing=True)
            event_state["target_wait_end_event"] = torch.cuda.Event(enable_timing=True)
            event_state["target_wait_start_event"].record(current)
            wait_call_start_host = time.perf_counter()
        current.wait_event(entry["event"])
        wait_call_end_host = time.perf_counter()
        if event_state is not None:
            event_state["target_wait_end_event"].record(current)
            event_state["target_wait_call_host_ms"] = (wait_call_end_host - wait_call_start_host) * 1000.0
        kv_cache["temporal_qprev_hits"] = int(kv_cache.get("temporal_qprev_hits", 0)) + 1
        if self.group11_profile is not None:
            self.group11_profile["QPREFETCH_HITS"] = int(self.group11_profile.get("QPREFETCH_HITS", 0)) + 1
        kv_cache.setdefault("temporal_qprev_consumed_slots", []).append(int(history_id))
        return entry["k"][batch_index, 0], entry["v"][batch_index, 0]

    def _online_temporal_qprev_indices(self, query, kv_cache, layer_index,
                                       frame_seqlen, current_start):
        """Consume P_(t,l), or perform only the explicit bootstrap/fallback."""
        if self.retrieval_backend != "draftmap_online" or self.memory_size <= 0:
            return None
        compressed_entries = kv_cache.get("compressed_history_entries", [])
        cpu_k = compressed_entries if compressed_entries else kv_cache.get("cpu_k_frames", [])
        gpu_draft_k = kv_cache.get("gpu_draft_k_frames", [])
        eligible = max(len(cpu_k) - self.recent_exclude, 0)
        if eligible <= 0 or len(gpu_draft_k) < eligible:
            return None
        # Persist Q_t after this call so that a missing prediction can only
        # fall back to the exact preceding same-layer invocation.
        kv_cache["draft_q_retrieval_started"] = True
        invocation_id = int(kv_cache.get("qprev_current_invocation_id", 0))
        valid_invocation_id = int(kv_cache.get("temporal_valid_invocation_counter", 0))
        kv_cache["temporal_valid_invocation_counter"] = valid_invocation_id + 1
        kv_cache["temporal_current_valid_invocation_id"] = valid_invocation_id
        predictions = kv_cache.setdefault("temporal_qprev_predictions", {})
        prediction = predictions.get(valid_invocation_id)
        if prediction is not None:
            if prediction.get("target_layer") != int(layer_index):
                kv_cache["temporal_qprev_stale_rejects"] = int(kv_cache.get("temporal_qprev_stale_rejects", 0)) + 1
                prediction = None
            else:
                ids = prediction.get("selected_history_ids", [])
                event_state = kv_cache.get("temporal_qprev_event_state", {}).get(valid_invocation_id)
                if event_state is not None:
                    event_state["target_invocation_start_host"] = time.perf_counter()
                selected = torch.tensor([ids], device=query.device, dtype=torch.long)
                kv_cache["temporal_current_selection_source"] = "QPREV_PREFETCH"
                kv_cache["temporal_qprev_alignment_pass"] = int(kv_cache.get("temporal_qprev_alignment_pass", 0)) + 1
                if self.group11_profile is not None:
                    self.group11_profile.setdefault("QPREFETCH_TEMPORAL_ALIGNMENT_PASS", 0)
                    self.group11_profile["QPREFETCH_TEMPORAL_ALIGNMENT_PASS"] += 1
                return selected

        # No prediction exists only for bootstrap or a failed/stale prefetch.
        previous = kv_cache.get("draft_q_history")
        previous_meta = kv_cache.get("draft_q_history_meta")
        if not kv_cache.get("temporal_qprev_bootstrap_done", False):
            kv_cache["temporal_current_selection_source"] = "BOOTSTRAP_CURRENT_Q"
            kv_cache["temporal_qprev_bootstrap_count"] = int(kv_cache.get("temporal_qprev_bootstrap_count", 0)) + 1
            kv_cache["temporal_qprev_bootstrap_done"] = True
            if self.group11_profile is not None:
                self.group11_profile["BOOTSTRAP_CALLS"] = int(self.group11_profile.get("BOOTSTRAP_CALLS", 0)) + 1
            selected = self._draftmap_select_for_query(
                query, kv_cache, layer_index, frame_seqlen, current_start,
                "BOOTSTRAP_CURRENT_Q", invocation_id, bootstrap=True)
            if selected is not None:
                # Bootstrap has no prior attention whose tail can hide this
                # copy.  Establish P_(1,l) now; all steady-state predictions
                # are still launched after the current attention below.
                self._schedule_temporal_qprev_prefetch(
                    kv_cache, layer_index, selected, valid_invocation_id + 1,
                    invocation_id + 1, query.device)
                kv_cache["temporal_bootstrap_prefetch_scheduled"] = True
                if self.group11_profile is not None:
                    self.group11_profile["QPREFETCH_PREDICTIONS"] += 1
        else:
            kv_cache["temporal_qprev_nonbootstrap_misses"] = int(kv_cache.get("temporal_qprev_nonbootstrap_misses", 0)) + 1
            if self.group11_profile is not None:
                self.group11_profile["NON_BOOTSTRAP_PREFETCH_MISS"] = int(self.group11_profile.get("NON_BOOTSTRAP_PREFETCH_MISS", 0)) + 1
                self.group11_profile["QPREFETCH_TEMPORAL_ALIGNMENT_FAIL"] = int(self.group11_profile.get("QPREFETCH_TEMPORAL_ALIGNMENT_FAIL", 0)) + 1
            # A development-only fallback is available for debugging, but is
            # disabled for Group11.2 smoke/benchmark validation by default.
            if os.environ.get("TEMPORAL_QPREFETCH_DEBUG_FALLBACK", "0") != "1":
                raise AssertionError(
                    "non-bootstrap temporal prefetch state missing or stale: "
                    f"layer={layer_index} raw_invocation={invocation_id} "
                    f"valid_invocation={valid_invocation_id} "
                    f"prediction_keys={sorted(predictions.keys())[-8:]} "
                    f"bootstrap_done={kv_cache.get('temporal_qprev_bootstrap_done')}"
                )
            aligned = bool(
                previous_meta is not None
                and int(previous_meta.get("layer", -1)) == int(layer_index)
                and int(previous_meta.get("invocation_id", -2)) == invocation_id - 1
            )
            if not aligned:
                raise RuntimeError("debug previous_q_prefetch fallback is not exact previous invocation")
            kv_cache["temporal_current_selection_source"] = "PREVIOUS_Q_DEBUG_FALLBACK"
            kv_cache["temporal_qprev_fallbacks"] = int(kv_cache.get("temporal_qprev_fallbacks", 0)) + 1
            if self.group11_profile is not None:
                self.group11_profile["NON_BOOTSTRAP_FALLBACK_COUNT"] = int(self.group11_profile.get("NON_BOOTSTRAP_FALLBACK_COUNT", 0)) + 1
            selected = self._draftmap_select_for_query(
                previous, kv_cache, layer_index, frame_seqlen, current_start,
                "PREVIOUS_Q_DEBUG_FALLBACK", invocation_id)
        if self.group11_profile is not None:
            self.group11_profile.setdefault("QPREFETCH_TEMPORAL_ALIGNMENT_PASS", 0)
            self.group11_profile.setdefault("QPREFETCH_TEMPORAL_ALIGNMENT_FAIL", 0)
            self.group11_profile["QPREFETCH_TEMPORAL_ALIGNMENT_PASS"] += int(selected is not None)
        return selected

    def _schedule_temporal_next_from_current_q(self, query, kv_cache, layer_index,
                                               frame_seqlen, current_start,
                                               current_selected):
        """Use Q_t to create P_(t+1,l) after the current attention has run."""
        invocation_id = int(kv_cache.get("qprev_current_invocation_id", 0))
        target_valid_id = int(kv_cache.get("temporal_current_valid_invocation_id", 0)) + 1
        target_raw_id = invocation_id + 1
        source = kv_cache.get("temporal_current_selection_source")
        if source == "BOOTSTRAP_CURRENT_Q":
            if kv_cache.pop("temporal_bootstrap_prefetch_scheduled", False):
                return
            selected_next = current_selected
        else:
            selected_next = self._draftmap_select_for_query(
                query, kv_cache, layer_index, frame_seqlen, current_start,
                "CURRENT_Q_PREDICT_NEXT", target_valid_id)
        if selected_next is None:
            return
        self._schedule_temporal_qprev_prefetch(
            kv_cache, layer_index, selected_next, target_valid_id,
            target_raw_id, query.device)
        if self.group11_profile is not None:
            self.group11_profile.setdefault("QPREFETCH_PREDICTIONS", 0)
            self.group11_profile["QPREFETCH_PREDICTIONS"] += 1

    def _online_memory_indices(self, query, kv_cache, layer_index, frame_seqlen, current_start=0):
        if self.retrieval_query_mode == "previous_q_prefetch":
            return self._online_temporal_qprev_indices(query, kv_cache, layer_index, frame_seqlen, current_start)
        if self.retrieval_backend != "draftmap_online" or self.memory_size <= 0:
            return None
        compressed_entries = kv_cache.get("compressed_history_entries", [])
        cpu_k = compressed_entries if compressed_entries else kv_cache.get("cpu_k_frames", [])
        gpu_draft_k = kv_cache.get("gpu_draft_k_frames", [])
        eligible = max(len(cpu_k) - self.recent_exclude, 0)
        if eligible <= 0 or len(gpu_draft_k) < eligible:
            return None
        if self.retrieval_query_mode in {"previous_q", "previous_q_direct"}:
            kv_cache["draft_q_retrieval_started"] = True
        retrieval_query = query
        retrieval_query_source = "CURRENT_Q"
        q_prev_age_steps = None
        previous_meta = kv_cache.get("draft_q_history_meta")
        current_qprev_invocation = kv_cache.get("qprev_current_invocation_id")
        if self.retrieval_query_mode in {"previous_q", "previous_q_direct"}:
            previous = kv_cache.get("draft_q_history")
            if previous is None:
                retrieval_query_source = "BOOTSTRAP_CURRENT_Q"
                if self.group11_profile is not None:
                    self.group11_profile["BOOTSTRAP_CALLS"] += 1
                    self.group11_profile.setdefault("qprev_alignment_rows", []).append({
                        "layer": int(layer_index), "current_invocation_id": current_qprev_invocation,
                        "status": "BOOTSTRAP_CURRENT_Q", "exact_previous_invocation": False,
                    })
            else:
                layer_aligned = bool(
                    previous_meta is not None
                    and int(previous_meta.get("layer", -1)) == int(layer_index)
                )
                aligned = bool(
                    layer_aligned
                    and current_qprev_invocation is not None
                    and int(previous_meta.get("invocation_id", -2)) == int(current_qprev_invocation) - 1
                )
                if not layer_aligned:
                    raise RuntimeError("q_history layer mismatch in previous_q retrieval")
                retrieval_query = previous
                retrieval_query_source = "PREVIOUS_Q"
                if previous_meta.get("denoising_step") is not None and self.current_denoising_step is not None:
                    q_prev_age_steps = 1
                if self.group11_profile is not None:
                    self.group11_profile["PREVIOUS_Q_RETRIEVAL_CALLS"] += 1
                    self.group11_profile.setdefault("qprev_alignment_rows", []).append({
                        "layer": int(layer_index), "current_invocation_id": current_qprev_invocation,
                        "previous_invocation_id": int(previous_meta.get("invocation_id", -1)),
                        "previous_denoising_step": previous_meta.get("denoising_step"),
                        "current_denoising_step": self.current_denoising_step,
                        "previous_current_start": previous_meta.get("current_start"),
                        "current_start": int(current_start),
                        "status": "PASS" if aligned else "FAIL",
                        "exact_previous_invocation": bool(aligned),
                    })
                    if (not aligned) and os.environ.get("QPREV_ALIGNMENT_ASSERT", "0") == "1":
                        raise AssertionError("previous_q state is not the exact previous invocation")
        from utils.draftmap_retrieval import DraftChunkRecord, DraftMapChunkIndex
        score_start = time.perf_counter()
        pool_start = time.perf_counter()
        records = [
            DraftChunkRecord(str(i), gpu_draft_k[i], tuple(range(gpu_draft_k[i].size(2))))
            for i in range(eligible)
        ]
        pool_ms = (time.perf_counter() - pool_start) * 1000.0
        if len(records) != eligible:
            return None
        index = DraftMapChunkIndex(block_tokens=64)
        if any(r.draft_k.device != retrieval_query.device for r in records):
            raise RuntimeError("persistent Draft-K must remain GPU-resident")
        device_records = records
        scores = index.score_history(retrieval_query, device_records)
        draft_ms = (time.perf_counter() - score_start) * 1000.0
        topk_start = time.perf_counter()
        topk_scores, selected = index.select_topk(scores, self.memory_size)
        topk_ms = (time.perf_counter() - topk_start) * 1000.0
        trace = cpt.ACTIVE_TRACE
        if trace:
            selected_ids_cpu = trace.measure(
                "ONLINE_SELECTED_IDS_DETACH_CPU", lambda: selected[0].detach().cpu()
            )
            selected_ids_list = trace.measure(
                "ONLINE_SELECTED_IDS_TOLIST", lambda: selected_ids_cpu.tolist()
            )
        else:
            selected_ids_list = selected[0].detach().cpu().tolist()
        if self.group11_profile is not None:
            p = self.group11_profile
            p["NUM_DRAFTMAP_CALLS"] += 1
            p["NUM_DRAFT_Q_POOL_CALLS"] += 1
            p["NUM_DRAFT_K_SCORE_CALLS"] += 1
            p["NUM_TOPK_CALLS"] += 1
            p["draftmap_rows"].append({
                "layer": int(layer_index), "denoising_step": self.current_denoising_step,
                "current_chunk_id": int(kv_cache.get("global_end_index", torch.zeros(1)).item() // frame_seqlen),
                "num_historical_chunks": int(len(cpu_k)), "num_candidate_chunks": int(eligible),
                "num_selected_chunks": int(selected.shape[-1]),
                "draft_q_shape": list(query.shape), "draft_k_shape": list(gpu_draft_k[0].shape) if gpu_draft_k else [],
                "draft_score_elements": int(scores.numel()), "time_draft_q_pool_ms": pool_ms,
                "time_score_ms": draft_ms, "time_aggregate_ms": 0.0, "time_topk_ms": topk_ms,
            })
        self.draftmap_trace.append({
            "generation_unit": int(kv_cache.get("global_end_index", torch.zeros(1)).item() // frame_seqlen),
            "layer": int(layer_index),
            "candidate_count": int(eligible),
            "candidate_history_ids": list(range(eligible)),
            "selected_history_ids": selected_ids_list,
            "selected_scores": [float(v) for v in topk_scores[0].detach().cpu()],
            "retrieval_budget": int(self.memory_size),
            "retrieval_query_mode": self.retrieval_query_mode,
            "wrapper_impl": "original_group11_custom",
            "current_q_logical_id": {"layer": int(layer_index), "chunk": int(current_start // frame_seqlen), "denoise_step": self.current_denoising_step},
            "retrieval_q_logical_id": ({"layer": int(layer_index), "chunk": int(previous_meta.get("chunk", -1)), "denoise_step": previous_meta.get("denoising_step")} if retrieval_query_source == "PREVIOUS_Q" and previous_meta is not None else {"layer": int(layer_index), "chunk": int(current_start // frame_seqlen), "denoise_step": self.current_denoising_step}),
            "retrieval_query_source": retrieval_query_source,
            "q_prev_age_steps": q_prev_age_steps,
            "recent_history_ids": list(range(eligible, len(cpu_k))),
            "draft_score_ms": draft_ms,
            "topk_ms": topk_ms,
            "cpu_gather_ms": 0.0,
            "h2d_ms": 0.0,
            "denoising_step": self.current_denoising_step,
        })
        return selected

    def _prefetch_state(self, kv_cache):
        """Shared transient state; never a persistent KV owner."""
        all_cache = getattr(self, "prefetch_kv_cache", None)
        owner = all_cache[0] if isinstance(all_cache, list) and all_cache else kv_cache
        state = owner.setdefault("next_layer_prefetch", {})
        owner.setdefault("next_layer_prefetch_trace", [])
        owner.setdefault("prefetch_buffer_peak_bytes", 0)
        owner.setdefault("prefetch_staging_slots", {})
        return owner, state

    def _reconcile_prefetch(self, kv_cache, layer_index, true_ids):
        """Drop stale predictions before attention and account for waste."""
        owner, state = self._prefetch_state(kv_cache)
        true_set = {int(x) for x in true_ids}
        wasted = []
        for key in list(state):
            if key[0] == int(layer_index) and key[1] not in true_set:
                wasted.append(state.pop(key))
        owner["prefetch_wasted_bytes"] = int(owner.get("prefetch_wasted_bytes", 0)) + sum(int(v.get("bytes", 0)) for v in wasted)
        owner["prefetch_wasted_chunks"] = int(owner.get("prefetch_wasted_chunks", 0)) + len(wasted)
        return wasted

    def _schedule_next_layer_prefetch(self, kv_cache, layer_index, selected, device):
        """Schedule (layer+1, I_l) from the next layer's CPU BF16 archive."""
        all_cache = getattr(self, "prefetch_kv_cache", None)
        if selected is None or not isinstance(all_cache, list) or layer_index + 1 >= len(all_cache):
            return
        owner, state = self._prefetch_state(kv_cache)
        stream = owner.get("prefetch_stream")
        if stream is None or stream.device != device:
            stream = torch.cuda.Stream(device=device)
            owner["prefetch_stream"] = stream
        next_cache = all_cache[layer_index + 1]
        cpu_k, cpu_v = next_cache.get("cpu_k_frames", []), next_cache.get("cpu_v_frames", [])
        # The archive is a Python-indexed CPU structure.  Until it is replaced
        # by a GPU-indexable packed archive, this conversion is unavoidable for
        # correct source selection; make it explicit in provenance.
        ids = [int(v) for v in selected[0].detach().cpu().tolist()]
        owner["prefetch_host_id_sync_count"] = int(owner.get("prefetch_host_id_sync_count", 0)) + 1
        requested = 0
        copied_bytes = 0
        with torch.cuda.stream(stream):
            for history_id in ids:
                key = (int(layer_index + 1), history_id)
                if key in state or history_id >= len(cpu_k) or history_id >= len(cpu_v):
                    continue
                src_k, src_v = cpu_k[history_id], cpu_v[history_id]
                if not src_k.is_pinned() or not src_v.is_pinned():
                    raise RuntimeError("prefetch source archive is not pre-pinned")
                copy_start_event = torch.cuda.Event(enable_timing=True)
                copy_end_event = torch.cuda.Event(enable_timing=True)
                copy_start_event.record(stream)
                slots = owner["prefetch_staging_slots"]
                slot = slots.get(key)
                if slot is None or slot[0].shape != src_k.shape or slot[1].shape != src_v.shape:
                    dst_k = torch.empty_like(src_k, device=device)
                    dst_v = torch.empty_like(src_v, device=device)
                    slots[key] = (dst_k, dst_v)
                    owner["prefetch_staging_allocations"] = int(owner.get("prefetch_staging_allocations", 0)) + 1
                else:
                    dst_k, dst_v = slot
                dst_k.copy_(src_k, non_blocking=True)
                dst_v.copy_(src_v, non_blocking=True)
                copy_end_event.record(stream)
                event = torch.cuda.Event()
                event.record(stream)
                nbytes = int(src_k.numel() * src_k.element_size() + src_v.numel() * src_v.element_size())
                state[key] = {"k": dst_k, "v": dst_v, "event": event, "bytes": nbytes}
                self._rag_record_fetch(
                    layer_id=int(layer_index + 1), attention_call_id=int(self.group11_profile.get("NUM_ATTENTION_CALLS", 0)) if self.group11_profile is not None else 0,
                    history_id=int(history_id), fetch_reason="prefetch",
                    fetch_bytes_k=int(src_k.numel() * src_k.element_size()), fetch_bytes_v=int(src_v.numel() * src_v.element_size()),
                    fetch_work_ms=0.0, fetch_exposed_ms=0.0, is_first_fetch_in_call=(requested == 0),
                    prefetch_hit_status="not_applicable")
                if self._rag_profile_on():
                    fetch_id = int(self.group11_profile["rag_fetch_events"][-1]["fetch_event_id"])
                    self._rag_cuda_pending.append({"measurement_kind": "prefetch_copy", "fetch_event_id": fetch_id,
                                                   "layer_id": int(layer_index + 1), "history_id": int(history_id),
                                                   "start": copy_start_event, "end": copy_end_event})
                requested += 1
                copied_bytes += nbytes
        owner["prefetch_requested_chunks"] = int(owner.get("prefetch_requested_chunks", 0)) + requested
        owner["prefetch_scheduled_bytes"] = int(owner.get("prefetch_scheduled_bytes", 0)) + copied_bytes
        owner["prefetch_buffer_peak_bytes"] = max(int(owner.get("prefetch_buffer_peak_bytes", 0)), sum(int(v.get("bytes", 0)) for v in state.values()))
        owner["next_layer_prefetch_trace"].append({"layer_id": int(layer_index), "next_layer_id": int(layer_index + 1), "predicted_ids": ids, "requested_chunks": requested, "prefetch_bytes": copied_bytes})

    def _consume_prefetch_or_fetch(self, kv_cache, layer_index, history_id, batch_index, device):
        """Consume only exact current-Q hits; None requests a correction fetch."""
        owner, state = self._prefetch_state(kv_cache)
        entry = state.pop((int(layer_index), int(history_id)), None)
        if entry is None:
            return None
        current = torch.cuda.current_stream(device)
        wait_start_event = torch.cuda.Event(enable_timing=True)
        wait_end_event = torch.cuda.Event(enable_timing=True)
        wait_start_event.record(current)
        current.wait_event(entry["event"])
        wait_end_event.record(current)
        if self._rag_profile_on():
            # The corresponding prefetch event is identified by history/layer
            # after the event record has been created; the wait itself remains
            # on the implementation's existing current stream.
            self._rag_cuda_pending.append({"measurement_kind": "prefetch_wait",
                                           "layer_id": int(layer_index), "history_id": int(history_id),
                                           "prefetch_entry_bytes": int(entry.get("bytes", 0)),
                                           "start": wait_start_event, "end": wait_end_event})
        owner["prefetch_hit_chunks"] = int(owner.get("prefetch_hit_chunks", 0)) + 1
        owner["prefetch_hit_bytes"] = int(owner.get("prefetch_hit_bytes", 0)) + int(entry["bytes"])
        return entry["k"][batch_index, 0], entry["v"][batch_index, 0]

    def _queue_draft_q_history(self, query, kv_cache, layer_index, current_start, frame_seqlen):
        if self.retrieval_query_mode not in {"previous_q", "previous_q_direct", "previous_q_prefetch"} or kv_cache is None:
            return
        invocation_id = int(kv_cache.get("qprev_invocation_counter", 0))
        kv_cache["qprev_invocation_counter"] = invocation_id + 1
        kv_cache["qprev_current_invocation_id"] = invocation_id
        kv_cache["draft_q_pending"] = (
            query.detach().clone(),
            {"layer": int(layer_index), "chunk": int(current_start // frame_seqlen),
             "current_start": int(current_start), "denoising_step": self.current_denoising_step,
             "invocation_id": invocation_id},
        )

    def _commit_draft_q_history(self, kv_cache):
        if self.retrieval_query_mode not in {"previous_q", "previous_q_direct", "previous_q_prefetch"} or kv_cache is None:
            return
        pending = kv_cache.pop("draft_q_pending", None)
        if pending is not None and kv_cache.get("draft_q_retrieval_started", False):
            kv_cache["draft_q_history"], kv_cache["draft_q_history_meta"] = pending
            if self.group11_profile is not None:
                self.group11_profile["Q_HISTORY_PERSISTED_CALLS"] += 1

    def _flashfetch_online_attention(self, query, sink_k, sink_v, local_k, local_v,
                                     kv_cache, memory_indices, grid_sizes, freqs,
                                     layer_index):
        if self.flash_fetch_mode in {"async_double_buffer", "flashfetch_async", "async_stock_flashattn"}:
            return self._flashfetch_online_attention_async_double_buffer(
                query, sink_k, sink_v, local_k, local_v, kv_cache,
                memory_indices, grid_sizes, freqs, layer_index)
        return self._flashfetch_online_attention_serial_reference(
            query, sink_k, sink_v, local_k, local_v, kv_cache,
            memory_indices, grid_sizes, freqs, layer_index)

    def _flashfetch_online_attention_serial_reference(self, query, sink_k, sink_v, local_k, local_v,
                                     kv_cache, memory_indices, grid_sizes, freqs,
                                     layer_index):
        """Serial full-chunk Flash Fetch correctness path.

        Each selected history chunk is one 1560-token partition.  The online
        softmax merge is mathematically equivalent to one dense unmasked
        attention over resident + selected history tokens.  This first phase
        intentionally uses blocking H2D; async double buffering is enabled
        only after this path passes the numerical gate.
        """
        if query.size(0) != 1:
            raise RuntimeError("flashfetch prototype currently requires batch_size=1")
        if memory_indices is None:
            return attention(query, torch.cat([sink_k, local_k], dim=1),
                             torch.cat([sink_v, local_v], dim=1)), {"mode": "no_history"}
        cpu_k = kv_cache.get("cpu_k_frames", [])
        cpu_v = kv_cache.get("cpu_v_frames", [])
        ids = [int(v) for v in memory_indices[0].detach().cpu().tolist()]
        frame_seq = int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item())
        h, w = int(grid_sizes[0, 1].item()), int(grid_sizes[0, 2].item())
        one_frame_grid = grid_sizes.new_tensor([[1, h, w]])
        scale = query.shape[-1] ** -0.5
        M = None
        L = None
        O = None
        timeline = []

        def merge(k_part, v_part, partition_name, history_id=None):
            nonlocal M, L, O
            start = time.perf_counter()
            scores = torch.einsum("bqhd,bkhd->bqhk", query.float(), k_part.float()) * scale
            m = scores.amax(dim=-1)
            exp_scores = torch.exp(scores - m.unsqueeze(-1))
            l = exp_scores.sum(dim=-1)
            o = torch.einsum("bqhk,bkhd->bqhd", exp_scores, v_part.float())
            if M is None:
                M, L, O = m, l, o
            else:
                new_m = torch.maximum(M, m)
                alpha = torch.exp(M - new_m)
                beta = torch.exp(m - new_m)
                O = alpha.unsqueeze(-1) * O + beta.unsqueeze(-1) * o
                L = alpha * L + beta * l
                M = new_m
            end = time.perf_counter()
            timeline.append({"partition": partition_name, "history_id": history_id,
                             "compute_ms": (end - start) * 1000.0,
                             "tokens": int(k_part.shape[1])})

        resident_k = torch.cat([sink_k, local_k], dim=1)
        resident_v = torch.cat([sink_v, local_v], dim=1)
        merge(resident_k, resident_v, "resident", None)
        total_h2d = 0.0
        for rank, history_id in enumerate(ids):
            if history_id >= len(cpu_k) or history_id >= len(cpu_v):
                raise IndexError(f"history id {history_id} unavailable")
            h2d_start = time.perf_counter()
            src_k = cpu_k[history_id][0, 0].to(query.device, non_blocking=True)
            src_v = cpu_v[history_id][0, 0].to(query.device, non_blocking=True)
            h2d_ms = (time.perf_counter() - h2d_start) * 1000.0
            total_h2d += h2d_ms
            rope_start = time.perf_counter()
            k_part = causal_online_rope(src_k.unsqueeze(0), one_frame_grid, freqs,
                                        relative_frame_indices=torch.zeros(1, dtype=torch.long, device=query.device))
            k_part = k_part[0:1].type_as(query)
            v_part = src_v.unsqueeze(0).type_as(query)
            rope_ms = (time.perf_counter() - rope_start) * 1000.0
            compute_start = time.perf_counter()
            merge(k_part, v_part, "history", history_id)
            compute_ms = (time.perf_counter() - compute_start) * 1000.0
            timeline[-1].update({"chunk_rank": rank, "h2d_ms": h2d_ms,
                                 "rope_ms": rope_ms, "compute_start_ms": compute_start,
                                 "compute_end_ms": time.perf_counter(), "bytes": int(src_k.numel() * src_k.element_size() * 2)})
        output = (O / L.unsqueeze(-1)).type_as(query)
        self.group11_flash_trace.append({
            "layer_id": int(layer_index), "history_ids": ids,
            "tile_tokens": frame_seq, "h2d_ms": total_h2d,
            "timeline": timeline,
            "cpu_pinned": bool(all(x.is_pinned() for x in cpu_k[:len(ids)])) if ids else True,
            "async_overlap": False,
        })
        return output, {"mode": "flashfetch_serial", "history_ids": ids,
                        "h2d_ms": total_h2d, "tile_tokens": frame_seq}

    def _flashfetch_pinned_source(self, source, source_kind, history_id):
        """Return a reusable pinned CPU view, accounting staging separately."""
        if source.device.type != "cpu":
            raise RuntimeError("Flash Fetch source must be CPU-resident")
        # A [0, 0] view is a temporary Python object; its id() can be reused
        # for another history block.  Use the underlying storage address and
        # shape instead, so cached pinned sources cannot alias another block.
        key = (source_kind, int(source.data_ptr()), tuple(source.shape), str(source.dtype))
        cached = self._flash_pinned_sources.get(key)
        if cached is not None:
            return cached, 0.0, False
        start = time.perf_counter()
        if source.is_pinned():
            pinned = source
        else:
            pinned = source.contiguous().pin_memory()
        staging_ms = (time.perf_counter() - start) * 1000.0
        self._flash_pinned_sources[key] = pinned
        return pinned, staging_ms, True

    def _flashfetch_online_attention_async_double_buffer(
            self, query, sink_k, sink_v, local_k, local_v, kv_cache,
            memory_indices, grid_sizes, freqs, layer_index):
        """Same-layer async Flash Fetch with pinned CPU sources and two GPU buffers.

        The visible history IDs, partition order, online-softmax equations, RoPE,
        and final BF16 attention are intentionally identical to the serial path.
        Only the transfer schedule changes: the dedicated fetch stream fills the
        next reusable buffer while the current buffer is merged on the compute
        stream.  No host wait is used in this per-tile loop.
        """
        if query.size(0) != 1:
            raise RuntimeError("flashfetch prototype currently requires batch_size=1")
        if memory_indices is None:
            return attention(query, torch.cat([sink_k, local_k], dim=1),
                             torch.cat([sink_v, local_v], dim=1)), {"mode": "no_history"}
        if not query.is_cuda:
            raise RuntimeError("async_double_buffer requires CUDA")
        use_stock_flashattn = self.flash_fetch_mode == "async_stock_flashattn"
        cpu_k = kv_cache.get("cpu_k_frames", [])
        cpu_v = kv_cache.get("cpu_v_frames", [])
        ids = [int(v) for v in memory_indices[0].detach().cpu().tolist()]
        frame_seq = int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item())
        h, w = int(grid_sizes[0, 1].item()), int(grid_sizes[0, 2].item())
        one_frame_grid = grid_sizes.new_tensor([[1, h, w]])
        scale = query.shape[-1] ** -0.5
        compute_stream = torch.cuda.current_stream(query.device)
        device_key = (query.device.index, query.dtype, int(query.shape[-2]),
                      int(query.shape[-1]), frame_seq)
        state = self._flash_fetch_state.get(device_key)
        if state is None:
            fetch_stream = torch.cuda.Stream(device=query.device)
            state = {"stream": fetch_stream, "buffers": [None, None],
                     "ready_events": [None, None], "shape": None}
            self._flash_fetch_state[device_key] = state
        fetch_stream = state["stream"]

        # Stage sources once into pinned host memory. This is explicitly recorded
        # as host staging and is never included in the H2D overlap claim.
        staged_k, staged_v = [], []
        host_stage_ms = 0.0
        staged_new = 0
        for history_id in ids:
            if history_id >= len(cpu_k) or history_id >= len(cpu_v):
                raise IndexError(f"history id {history_id} unavailable")
            sk, ms, new = self._flashfetch_pinned_source(
                cpu_k[history_id][0, 0], "k", history_id)
            sv, ms_v, new_v = self._flashfetch_pinned_source(
                cpu_v[history_id][0, 0], "v", history_id)
            staged_k.append(sk)
            staged_v.append(sv)
            host_stage_ms += ms + ms_v
            staged_new += int(new) + int(new_v)

        if not staged_k:
            resident_k = torch.cat([sink_k, local_k], dim=1)
            resident_v = torch.cat([sink_v, local_v], dim=1)
            return attention(query, resident_k, resident_v), {
                "mode": "flashfetch_async", "history_ids": ids,
                "h2d_ms": 0.0, "tile_tokens": frame_seq,
                "host_stage_ms": host_stage_ms, "cpu_pinned": True,
                "async_overlap": True,
            }

        shape = tuple(staged_k[0].shape)
        if state["shape"] != (shape, staged_k[0].dtype, staged_v[0].dtype):
            state["buffers"] = [
                torch.empty(shape, device=query.device, dtype=staged_k[0].dtype),
                torch.empty(shape, device=query.device, dtype=staged_k[0].dtype),
            ], [
                torch.empty(tuple(staged_v[0].shape), device=query.device, dtype=staged_v[0].dtype),
                torch.empty(tuple(staged_v[0].shape), device=query.device, dtype=staged_v[0].dtype),
            ]
            state["shape"] = (shape, staged_k[0].dtype, staged_v[0].dtype)
            state["ready_events"] = [None, None]
        k_buffers, v_buffers = state["buffers"]
        # Make the fetch stream wait for prior compute use before reusing buffers.
        for event in state["ready_events"]:
            if event is not None:
                fetch_stream.wait_event(event)
        state["ready_events"] = [None, None]

        copy_records = {}
        compute_records = []
        timeline = []

        def enqueue_fetch(rank, buf):
            start = time.perf_counter()
            copy_start = torch.cuda.Event(enable_timing=True)
            copy_done = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(fetch_stream):
                copy_start.record(fetch_stream)
                k_buffers[buf].copy_(staged_k[rank], non_blocking=True)
                v_buffers[buf].copy_(staged_v[rank], non_blocking=True)
                copy_done.record(fetch_stream)
            copy_records[rank] = (copy_start, copy_done, (time.perf_counter() - start) * 1000.0)

        # First fetch is launched before resident attention, enabling overlap.
        enqueue_fetch(0, 0)
        M = L = O = None
        stock_lse = None

        def stock_flashattn_with_lse(k_part, v_part):
            """Call the exact FA2 varlen API used by wan.modules.attention."""
            from flash_attn import flash_attn_varlen_func
            q_flat = query.flatten(0, 1)
            k_flat = k_part.flatten(0, 1)
            v_flat = v_part.flatten(0, 1)
            cu_q = torch.tensor([0, query.shape[1]], device=query.device, dtype=torch.int32)
            cu_k = torch.tensor([0, k_part.shape[1]], device=query.device, dtype=torch.int32)
            out, lse, _ = flash_attn_varlen_func(
                q_flat, k_flat, v_flat, cu_q, cu_k,
                query.shape[1], k_part.shape[1],
                dropout_p=0.0, softmax_scale=None, causal=False,
                window_size=(-1, -1), deterministic=False,
                return_attn_probs=True)
            return out.reshape(query.shape), lse

        def merge(k_part, v_part, partition_name, history_id=None):
            nonlocal M, L, O
            nonlocal stock_lse
            if use_stock_flashattn:
                o_i, lse_i = stock_flashattn_with_lse(k_part, v_part)
                if stock_lse is None:
                    O = o_i.float()
                    stock_lse = lse_i
                else:
                    new_lse = torch.logaddexp(stock_lse, lse_i)
                    w_acc = torch.exp(stock_lse - new_lse).transpose(0, 1).unsqueeze(0).unsqueeze(-1)
                    w_i = torch.exp(lse_i - new_lse).transpose(0, 1).unsqueeze(0).unsqueeze(-1)
                    O = w_acc * O + w_i * o_i.float()
                    stock_lse = new_lse
                return
            scores = torch.einsum("bqhd,bkhd->bqhk", query.float(), k_part.float()) * scale
            m = scores.amax(dim=-1)
            exp_scores = torch.exp(scores - m.unsqueeze(-1))
            l = exp_scores.sum(dim=-1)
            o = torch.einsum("bqhk,bkhd->bqhd", exp_scores, v_part.float())
            if M is None:
                M, L, O = m, l, o
            else:
                new_m = torch.maximum(M, m)
                alpha = torch.exp(M - new_m)
                beta = torch.exp(m - new_m)
                O = alpha.unsqueeze(-1) * O + beta.unsqueeze(-1) * o
                L = alpha * L + beta * l
                M = new_m

        resident_k = torch.cat([sink_k, local_k], dim=1)
        resident_v = torch.cat([sink_v, local_v], dim=1)
        resident_start = torch.cuda.Event(enable_timing=True)
        resident_done = torch.cuda.Event(enable_timing=True)
        resident_start.record(compute_stream)
        merge(resident_k, resident_v, "resident", None)
        resident_done.record(compute_stream)
        compute_records.append((resident_start, resident_done))
        timeline.append({"partition": "resident", "history_id": None,
                         "chunk_rank": None, "tokens": int(resident_k.shape[1]),
                         "buffer": None, "h2d_ms": 0.0,
                         "compute_ms": "CUDA_EVENT"})

        for rank, history_id in enumerate(ids):
            buf = rank % 2
            copy_start, copy_done, copy_enqueue_ms = copy_records[rank]
            compute_stream.wait_event(copy_done)
            start = time.perf_counter()
            compute_start = torch.cuda.Event(enable_timing=True)
            compute_done = torch.cuda.Event(enable_timing=True)
            compute_start.record(compute_stream)
            k_part = causal_online_rope(
                k_buffers[buf].unsqueeze(0), one_frame_grid, freqs,
                relative_frame_indices=torch.zeros(1, dtype=torch.long, device=query.device))
            k_part = k_part[0:1].type_as(query)
            v_part = v_buffers[buf].unsqueeze(0).type_as(query)
            merge(k_part, v_part, "history", history_id)
            compute_done.record(compute_stream)
            compute_enqueue_ms = (time.perf_counter() - start) * 1000.0
            compute_records.append((compute_start, compute_done))
            timeline.append({"partition": "history", "history_id": history_id,
                             "chunk_rank": rank, "tokens": frame_seq,
                             "buffer": buf, "h2d_ms": "CUDA_EVENT",
                             "compute_ms": "CUDA_EVENT",
                             "host_copy_enqueue_ms": copy_enqueue_ms,
                             "host_compute_enqueue_ms": compute_enqueue_ms})
            state["ready_events"][buf] = compute_done
            if rank + 1 < len(ids):
                enqueue_fetch(rank + 1, (rank + 1) % 2)

        output = O.type_as(query) if use_stock_flashattn else (O / L.unsqueeze(-1)).type_as(query)
        execution_mode = "async_stock_flashattn" if use_stock_flashattn else "async_double_buffer"
        self._flash_event_records.append({
            "copy": [(v[0], v[1]) for _, v in sorted(copy_records.items())],
            "compute": compute_records,
            "timeline": timeline,
        })
        self.group11_flash_trace.append({
            "layer_id": int(layer_index), "history_ids": ids,
            "tile_tokens": frame_seq, "h2d_ms": "CUDA_EVENT",
            "host_stage_ms": host_stage_ms, "host_stage_new_sources": staged_new,
            "timeline": timeline, "cpu_pinned": True,
            "async_overlap": True, "mode": execution_mode,
            "stock_flashattn_used": bool(use_stock_flashattn),
            "fetch_stream_handle": int(fetch_stream.cuda_stream),
            "compute_stream_handle": int(compute_stream.cuda_stream),
            "h2d_calls": int(2 * len(staged_k)),
            "h2d_bytes": int(sum(int(k.numel() * k.element_size() + v.numel() * v.element_size()) for k, v in zip(staged_k, staged_v))),
        })
        return output, {"mode": execution_mode, "history_ids": ids,
                        "h2d_ms": "CUDA_EVENT", "tile_tokens": frame_seq,
                        "host_stage_ms": host_stage_ms,
                        "cpu_pinned": True, "async_overlap": True}

    def finalize_flash_fetch_timing(self):
        """Materialize CUDA event timings once, after the workload is complete."""
        if not self._flash_event_records:
            return
        torch.cuda.synchronize()
        record_iter = iter(self._flash_event_records)
        for row in self.group11_flash_trace:
            if not row.get("async_overlap"):
                continue
            rec = next(record_iter)
            copy_ms = 0.0
            compute_ms = 0.0
            for start, end in rec["copy"]:
                copy_ms += float(start.elapsed_time(end))
            for start, end in rec["compute"]:
                compute_ms += float(start.elapsed_time(end))
            row["h2d_ms"] = copy_ms
            row["attention_compute_ms"] = compute_ms
            durations = [float(start.elapsed_time(end)) for start, end in rec["compute"]]
            for tile, duration in zip(row.get("timeline", []), durations):
                tile["compute_ms"] = duration
            history_count = max(len(row.get("timeline", [])) - 1, 1)
            for tile in row.get("timeline", []):
                tile["h2d_ms"] = copy_ms / history_count if tile.get("partition") == "history" else 0.0
            if self.group11_profile is not None:
                p = self.group11_profile
                p["FLASH_FETCH_H2D_CUDA_WORK_MS"] = float(p.get("FLASH_FETCH_H2D_CUDA_WORK_MS", 0.0)) + copy_ms
                p["FLASH_FETCH_H2D_HOST_ENQUEUE_MS"] = float(p.get("FLASH_FETCH_H2D_HOST_ENQUEUE_MS", 0.0)) + float(sum(float(t.get("host_copy_enqueue_ms", 0.0)) for t in row.get("timeline", [])))
                p["FLASH_FETCH_H2D_CALLS"] = int(p.get("FLASH_FETCH_H2D_CALLS", 0)) + int(row.get("h2d_calls", 0))
                p["FLASH_FETCH_H2D_BYTES"] = int(p.get("FLASH_FETCH_H2D_BYTES", 0)) + int(row.get("h2d_bytes", 0))
        self._flash_event_records.clear()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        memory_indices=None,
        layer_index=0
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        self._current_layer_index = int(layer_index)
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        v5_active = False
        if os.environ.get("GROUP11_V5_PROFILE", "0") == "1":
            try:
                block_frames = int(os.environ.get("GROUP11_V5_BLOCK_FRAMES", "3"))
                block_index = int(current_start) // max(1, block_frames * int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item()))
                v5_active = (int(os.environ.get("GROUP11_V5_START_BLOCK", "6")) <= block_index < int(os.environ.get("GROUP11_V5_END_BLOCK", "8")))
            except Exception:
                v5_active = False
        self._v5_active = v5_active
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        qkv_start = time.perf_counter()
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        qkv_ctx = record_function("G11_QKV_PROJECTION") if (v5_active and record_function is not None) else nullcontext()
        with qkv_ctx:
            q, k, v = qkv_fn(x)
        if self.group11_profile is not None:
            self.group11_profile["model_phase_ms"]["QKV_projection"] += (time.perf_counter() - qkv_start) * 1000.0
        frame_seqlen = int(grid_sizes[0, 1].item() * grid_sizes[0, 2].item())
        self._queue_draft_q_history(q, kv_cache, layer_index, current_start, frame_seqlen)
        # Group11.4 direct ablation consumes the predicted working set produced
        # by the previous layer.  It deliberately skips current-Q re-retrieval,
        # reconciliation, and correction fetches after bootstrap.
        direct_predicted = None
        if self.group11_fetch_mode == "next_layer_prefetch_direct" and kv_cache is not None:
            owner, state = self._prefetch_state(kv_cache)
            ids = sorted(int(key[1]) for key in state if int(key[0]) == int(layer_index))
            if ids:
                direct_predicted = torch.tensor(ids, dtype=torch.long, device=q.device).view(1, -1)
                self.draftmap_trace.append({
                    "layer": int(layer_index), "retrieval_query_source": "DIRECT_PREDICTED_WORKING_SET",
                    "predicted_history_ids": ids, "selected_history_ids": ids,
                    "candidate_count": len(ids), "selected_count": len(ids),
                    "direct_no_reretrieve": True,
                })
        online_memory_indices = None if direct_predicted is not None else self._online_memory_indices(
            q, kv_cache, layer_index, frame_seqlen, current_start=current_start)
        if direct_predicted is not None:
            memory_indices = direct_predicted
            self._schedule_next_layer_prefetch(kv_cache, layer_index, memory_indices, q.device)
        elif online_memory_indices is not None:
            memory_indices = online_memory_indices
            if self.group11_fetch_mode == "next_layer_prefetch":
                true_ids = [int(v) for v in memory_indices[0].detach().cpu().tolist()]
                wasted = self._reconcile_prefetch(kv_cache, layer_index, true_ids)
                self._schedule_next_layer_prefetch(kv_cache, layer_index, memory_indices, q.device)
                if self.draftmap_trace:
                    self.draftmap_trace[-1]["prefetch_true_ids"] = true_ids
                    self.draftmap_trace[-1]["prefetch_wasted_count"] = len(wasted)
            elif self.group11_fetch_mode == "next_layer_prefetch_direct":
                # Bootstrap only: establish the first predicted set.  Later
                # layers take the direct predicted branch above.
                self._schedule_next_layer_prefetch(kv_cache, layer_index, memory_indices, q.device)
                if self.draftmap_trace:
                    self.draftmap_trace[-1]["direct_bootstrap"] = True
        draft_k_frames = _draft_frame_pools(k, frame_seqlen)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            num_new_frames = grid_sizes[0][0].item()  # F from grid_sizes
            
            current_end = current_start + q.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = q.shape[1]
            
            # Compute cache update parameters without modifying kv_cache directly
            cache_update_info = None
            is_recompute = current_end <= kv_cache["global_end_index"].item() and current_start > 0
            
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # === causal online RoPE ===
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens

                # Compute updated local indices
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation (without modifying the original cache)
                # Create temporary k, v for computation - store UN-ROPED K
                temp_k = kv_cache["k"].clone()  # These are un-roped K values
                temp_v = kv_cache["v"].clone()
                
                # --- CPU OFFLOAD DUMP LOGIC ---
                evicted_k_frames = []
                evicted_v_frames = []
                ev_k_split = []
                ev_v_split = []
                num_evicted_frames = 0
                if self.memory_size > 0 and num_evicted_tokens > 0:
                    num_evicted_frames = num_evicted_tokens // frame_seqlen
                    ev_k = temp_k[:, sink_tokens:sink_tokens + num_evicted_tokens]
                    ev_v = temp_v[:, sink_tokens:sink_tokens + num_evicted_tokens]
                    
                    ev_k_split = ev_k.view(b, num_evicted_frames, frame_seqlen, n, d).split(1, dim=1)
                    ev_v_split = ev_v.view(b, num_evicted_frames, frame_seqlen, n, d).split(1, dim=1)
                    
                    # Pin once at archive creation, outside the prefetch
                    # critical path. Prefetch only copies from these
                    # already-pinned immutable sources.
                    evicted_k_frames = [f.to("cpu", non_blocking=True).contiguous().pin_memory() for f in ev_k_split]
                    evicted_v_frames = [f.to("cpu", non_blocking=True).contiguous().pin_memory() for f in ev_v_split]
                evicted_draft_k_frames = kv_cache.get("local_draft_k_frames", [])[sink_tokens // frame_seqlen:sink_tokens // frame_seqlen + num_evicted_frames]

                # Apply rolling update to the temporary cache
                temp_k[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_k[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                temp_v[:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    temp_v[:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                
                # Insert new key/value into the temporary cache (UN-ROPED K!)
                # Protect sink_tokens only during recomputation
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    # Store UN-ROPED K in cache
                    temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # === causal online RoPE Application ===
                # For query: use relative indices based on position in window
                # Query frames are at the end of the window: [local_attn_size - num_new_frames, ..., local_attn_size - 1]
                query_relative_indices = torch.arange(
                    self.local_attn_size - num_new_frames, 
                    self.local_attn_size, 
                    device=q.device
                )
                rope_start = time.perf_counter()
                roped_query = causal_online_rope(
                    q, grid_sizes, freqs, relative_frame_indices=query_relative_indices
                ).type_as(v)
                if self.group11_profile is not None:
                    self.group11_profile["model_phase_ms"]["norm_rope"] += (time.perf_counter() - rope_start) * 1000.0
                
                # For cached K: apply RoPE dynamically based on current position in window
                # Sink frames: [0, 1, ..., sink_size - 1]
                # Rolling window frames: [sink_size, sink_size + 1, ..., local_end_index/frame_seqlen - 1]
                num_cache_frames = local_end_index // frame_seqlen
                cache_relative_indices = torch.arange(0, num_cache_frames, device=k.device)
                
                # Create a grid_sizes for the cached K
                cache_grid_sizes = grid_sizes.clone()
                cache_grid_sizes[0, 0] = num_cache_frames
                
                # Apply RoPE to cached K using relative indices
                roped_temp_k = causal_online_rope(
                    temp_k[:, :local_end_index].view(b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2),
                    cache_grid_sizes, freqs, relative_frame_indices=cache_relative_indices
                ).type_as(v)

                # Save cache update info for later use - store UN-ROPED K!
                cache_update_info = {
                    "action": "roll_and_insert",
                    "sink_tokens": sink_tokens,
                    "num_rolled_tokens": num_rolled_tokens,
                    "num_evicted_tokens": num_evicted_tokens,
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": k[:, roped_offset:roped_offset + write_len],  # UN-ROPED K!
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "is_recompute": is_recompute,
                    "evicted_k_frames": evicted_k_frames,
                    "evicted_v_frames": evicted_v_frames,
                    "evicted_draft_k_frames": evicted_draft_k_frames,
                    "new_draft_k_frames": draft_k_frames,
                }
            else:
                # === DIRECT INSERT MODE ===
                # Before cache is full, we can still use relative indices that grow sequentially
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens

                # Construct full k, v for attention computation
                temp_k = kv_cache["k"].clone()  # UN-ROPED K
                temp_v = kv_cache["v"].clone()
                
                # Protect sink_tokens only during recomputation
                write_start_index = max(local_start_index, sink_tokens) if is_recompute else local_start_index
                if sink_recache_after_switch:
                    write_start_index = local_start_index
                roped_offset = max(0, write_start_index - local_start_index)
                write_len = max(0, local_end_index - write_start_index)
                if write_len > 0:
                    # Store UN-ROPED K in cache
                    temp_k[:, write_start_index:local_end_index] = k[:, roped_offset:roped_offset + write_len]
                    temp_v[:, write_start_index:local_end_index] = v[:, roped_offset:roped_offset + write_len]

                # === RoPE Application with Relative Indices ===
                # Current frame position in the window
                current_frame_in_window = local_start_index // frame_seqlen
                
                # Query: apply RoPE with relative frame indices
                query_relative_indices = torch.arange(
                    current_frame_in_window,
                    current_frame_in_window + num_new_frames,
                    device=q.device
                )
                roped_query = causal_online_rope(
                    q, grid_sizes, freqs, relative_frame_indices=query_relative_indices
                ).type_as(v)
                
                # Cached K: apply RoPE dynamically
                num_cache_frames = local_end_index // frame_seqlen
                cache_relative_indices = torch.arange(0, num_cache_frames, device=k.device)
                
                cache_grid_sizes = grid_sizes.clone()
                cache_grid_sizes[0, 0] = num_cache_frames
                
                roped_temp_k = causal_online_rope(
                    temp_k[:, :local_end_index].view(b, num_cache_frames, frame_seqlen, n, d).flatten(1, 2),
                    cache_grid_sizes, freqs, relative_frame_indices=cache_relative_indices
                ).type_as(v)

                # Save cache update info - store UN-ROPED K!
                cache_update_info = {
                    "action": "direct_insert",
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "write_start_index": write_start_index,
                    "write_end_index": local_end_index,
                    "new_k": k[:, roped_offset:roped_offset + write_len],  # UN-ROPED K!
                    "new_v": v[:, roped_offset:roped_offset + write_len],
                    "current_end": current_end,
                    "is_recompute": is_recompute,
                    "new_draft_k_frames": draft_k_frames,
                }

            # Use roped K for attention computation
            if sink_tokens > 0 or self.memory_size > 0:
                # Concatenate sink tokens and local window tokens
                local_budget = self.max_attention_size - sink_tokens - (self.memory_size * frame_seqlen)
                k_sink = roped_temp_k[:, :sink_tokens]
                v_sink = temp_v[:, :sink_tokens]
                
                local_start_for_window = max(sink_tokens, local_end_index - local_budget)
                
                if self.group11_fetch_mode in {
                    "flashfetch_serial", "flashfetch_chunk", "flashfetch_async",
                    "async_double_buffer", "async_stock_flashattn"}:
                    if local_budget > 0 and local_start_for_window < local_end_index:
                        flash_local_k = roped_temp_k[:, local_start_for_window:local_end_index]
                        flash_local_v = temp_v[:, local_start_for_window:local_end_index]
                    else:
                        flash_local_k = roped_temp_k[:, :0]
                        flash_local_v = temp_v[:, :0]
                    x, flash_meta = self._flashfetch_online_attention(
                        roped_query, k_sink, v_sink, flash_local_k, flash_local_v,
                        kv_cache, memory_indices, grid_sizes, freqs, layer_index)
                    self.group_runtime_trace.append({
                        "FLASH_FETCH_ACTIVE": "YES",
                        "FLASH_FETCH_TILE_TOKENS": int(frame_seqlen),
                        "FINAL_ATTENTION_DTYPE": str(x.dtype).replace("torch.", ""),
                        **flash_meta,
                    })
                    if self.group11_profile is not None:
                        call_id = int(self.group11_profile["NUM_ATTENTION_CALLS"])
                        ev = [e for e in self.group11_profile.get("rag_fetch_events", []) if int(e.get("attention_call_id", -1)) == call_id]
                        timeline = flash_meta.get("timeline", []) if isinstance(flash_meta, dict) else []
                        self.group11_profile["NUM_ATTENTION_CALLS"] += 1
                        self.group11_profile["rag_attention_rows"].append({
                            "case_id": os.environ.get("RAG_PROFILE_CASE_ID", "case_01"), "step_id": self.current_denoising_step,
                            "layer_id": int(layer_index), "attention_call_id": call_id,
                            "history_count": int(len(kv_cache.get("cpu_k_frames", []))) if kv_cache is not None else 0,
                            "candidate_count": int(len(memory_indices[0])) if memory_indices is not None else 0,
                            "selected_count": int(len(memory_indices[0])) if memory_indices is not None else 0,
                            "fetch_event_count": len(ev), "fetch_bytes_total": sum(int(e["fetch_bytes_total"]) for e in ev),
                            "fetch_work_ms": sum(float(e["fetch_work_ms"]) for e in ev), "fetch_exposed_ms": sum(float(e["fetch_exposed_ms"]) for e in ev),
                            "fetch_hidden_ms": sum(float(e["fetch_hidden_ms"]) for e in ev),
                            "first_fetch_ms": next((float(e["fetch_work_ms"]) for e in ev if e.get("is_first_fetch_in_call")), 0.0),
                            "retrieval_work_ms": 0.0, "routing_work_ms": 0.0,
                            "attention_wrapper_ms": sum(float(t.get("compute_ms", 0.0)) for t in timeline),
                            "prefetch_hit_count": 0, "correction_fetch_count": 0, "cache_hit_count": 0, "cache_miss_count": len(ev),
                        })
                    x = x.flatten(2)
                    x = self.o(x)
                    if kv_cache is not None:
                        return x, (current_end, local_end_index, cache_update_info)
                    return x
                
                # --- MEMORY TOKEN RETRIEVAL (using pre-computed indices) ---
                archive_materialized_k = archive_materialized_v = None
                archive_materialized_meta = {}
                k_mem = None
                v_mem = None
                
                if self.memory_size > 0 and memory_indices is not None:
                    fetch_start = time.perf_counter()
                    lookup_start = time.perf_counter()
                    compressed_entries = kv_cache.get("compressed_history_entries", [])
                    cpu_k_list = compressed_entries if compressed_entries else kv_cache.get("cpu_k_frames", [])
                    cpu_v_list = kv_cache.get("cpu_v_frames", [])
                    archive = kv_cache.get("compressed_history_archive")
                    lookup_ms = (time.perf_counter() - lookup_start) * 1000.0
                    active_memory_indices = memory_indices
                    
                    if len(cpu_k_list) > 0:
                        # active_memory_indices is the Q-sparse retained set for
                        # Group14/15, or the unchanged retrieval set otherwise.
                        k_sel = active_memory_indices.shape[1]  # [B, k_sel]
                        device = q.device
                        gather_start = time.perf_counter()
                        
                        k_mem_unroped_list = []
                        v_mem_list = []
                        k_host_ms = 0.0
                        v_host_ms = 0.0
                        q_sparse_meta = {}
                        if archive is not None:
                            selected_ids = [int(x) for x in memory_indices[0].detach().cpu().tolist()]
                            latest_route = self.draftmap_trace[-1] if self.draftmap_trace else {}
                            route_ids = [int(x) for x in latest_route.get("selected_history_ids", [])]
                            route_scores = latest_route.get("selected_scores", [])
                            score_map = {rid: float(score) for rid, score in zip(route_ids, route_scores)}
                            if self.group_runtime_mode in {"group14_corrected", "group15_corrected"}:
                                retained_ids, removed_ids, q_sparse_meta = select_q_sparse_history(
                                    selected_ids, score_map, self.q_sparse_ratio)
                            else:
                                retained_ids, removed_ids, q_sparse_meta = select_q_sparse_history(
                                    selected_ids, score_map, 1.0)
                            active_memory_indices = memory_indices.new_tensor([retained_ids], dtype=memory_indices.dtype)
                            archive_materialized_k, archive_materialized_v, archive_materialized_meta = archive.materialize_selected(
                                retained_ids,
                                score_map,
                                device,
                                promotion_ratio=self.promotion_ratio,
                                materialization_mode=self.materialization_mode,
                            )
                            archive_materialized_meta.update(q_sparse_meta)
                            archive_materialized_meta["Q_SPARSE_INPUT_IDS"] = selected_ids
                            archive_materialized_meta["DRAFTMAP_IMPORTANCE"] = {str(k): float(v) for k, v in score_map.items()}
                            archive_materialized_meta["FINAL_HISTORY_IDS"] = retained_ids
                            if self.group11_profile is not None:
                                self.group11_profile["CACHE_INIT_H2D_BYTES"] = int(self.group11_profile.get("CACHE_INIT_H2D_BYTES", 0)) + int(getattr(archive, "last_cache_init_bytes", 0))
                                self.group11_profile["CACHE_INIT_H2D_CALLS"] = int(self.group11_profile.get("CACHE_INIT_H2D_CALLS", 0)) + int(getattr(archive, "last_cache_init_calls", 0))
                                self.group11_profile["CACHE_INIT_H2D_TIMING_CLASS"] = "legacy_host_observed_interval"
                            self._last_archive_materialization = archive_materialized_meta
                        k_sel = active_memory_indices.shape[1]
                        for bi in range(b):
                            indices = active_memory_indices[bi]
                            for k_idx in indices:
                                prefetched = None
                                cache_hit = False
                                correction_start_event = correction_end_event = None
                                if archive is not None:
                                    pos = len(k_mem_unroped_list)
                                    src_k, src_v = archive_materialized_k[pos], archive_materialized_v[pos]
                                    if self.group11_profile is not None:
                                        self.group11_profile["EXPOSED_H2D_MS"] = float(self.group11_profile.get("EXPOSED_H2D_MS", 0.0)) + float(getattr(archive, "last_h2d_ms", 0.0))
                                        self.group11_profile["EXPOSED_H2D_CALLS"] = int(self.group11_profile.get("EXPOSED_H2D_CALLS", 0)) + int(getattr(archive, "last_h2d_calls", 0))
                                    src_k = src_k[bi]
                                    src_v = src_v[bi]
                                    kv_cache["retrieved_archived_entries"] = int(kv_cache.get("retrieved_archived_entries", 0)) + 1
                                    deq_bytes = int(src_k.numel() * src_k.element_size() + src_v.numel() * src_v.element_size())
                                    kv_cache["transient_dequant_gpu_bytes"] = int(kv_cache.get("transient_dequant_gpu_bytes", 0)) + deq_bytes
                                    kv_cache["transient_dequant_gpu_peak_bytes"] = max(int(kv_cache.get("transient_dequant_gpu_peak_bytes", 0)), deq_bytes)
                                    kv_cache["transient_promotion_gpu_bytes"] = int(kv_cache.get("transient_promotion_gpu_bytes", 0)) + int(getattr(archive, "last_promotion_bytes", 0))
                                    kv_cache["transient_promotion_gpu_peak_bytes"] = max(int(kv_cache.get("transient_promotion_gpu_peak_bytes", 0)), int(getattr(archive, "last_promotion_bytes", 0)))
                                    cache_key = None
                                else:
                                    prefetched = None
                                    if self.group11_fetch_mode == "temporal_qprev_prefetch":
                                        prefetched = self._consume_temporal_qprev_prefetch(
                                            kv_cache, layer_index,
                                            int(kv_cache.get("temporal_current_valid_invocation_id", 0)),
                                            int(k_idx), bi, device)
                                    elif self.group11_fetch_mode in {"next_layer_prefetch", "next_layer_prefetch_direct"}:
                                        prefetched = self._consume_prefetch_or_fetch(kv_cache, layer_index, int(k_idx), bi, device)
                                    if prefetched is not None:
                                        dst_k, dst_v = prefetched
                                        src_k, src_v = dst_k, dst_v
                                        cache_key = None
                                        cache_hit = True
                                        k_ms = v_ms = 0.0
                                    else:
                                        src_k = cpu_k_list[k_idx][bi, 0]
                                        src_v = cpu_v_list[k_idx][bi, 0]
                                        cache_key = (int(getattr(self, "_current_layer_index", -1)), int(k_idx), int(bi))
                                reuse = os.environ.get("GROUP11_REUSE_CACHE", "0") == "1"
                                cache_hit = bool(locals().get("cache_hit", False) or (reuse and cache_key in getattr(self, "_group11_reuse_cache", {})))
                                if archive is not None:
                                    dst_k, dst_v = src_k, src_v
                                    k_ms = 0.0
                                    v_ms = 0.0
                                    archive_copy_ms = float(getattr(archive, "last_h2d_ms", 0.0))
                                elif cache_hit and prefetched is None:
                                    dst_k, dst_v = self._group11_reuse_cache[cache_key]
                                    k_ms = 0.0
                                    v_ms = 0.0
                                    if self.group11_profile is not None:
                                        self.group11_profile["CACHE_HITS"] += 1
                                        self.group11_profile["AVOIDED_H2D_CALLS"] += 2
                                        self.group11_profile["AVOIDED_H2D_BYTES"] += int(src_k.numel() * src_k.element_size() + src_v.numel() * src_v.element_size())
                                else:
                                    if self.group11_fetch_mode in {"next_layer_prefetch", "temporal_qprev_prefetch"}:
                                        owner, _ = self._prefetch_state(kv_cache)
                                        if self.group11_fetch_mode == "temporal_qprev_prefetch":
                                            owner["temporal_qprev_correction_chunks"] = int(owner.get("temporal_qprev_correction_chunks", 0)) + 1
                                        else:
                                            owner["prefetch_correction_chunks"] = int(owner.get("prefetch_correction_chunks", 0)) + 1
                                        owner["prefetch_correction_bytes"] = int(owner.get("prefetch_correction_bytes", 0)) + int(src_k.numel() * src_k.element_size() + src_v.numel() * src_v.element_size())
                                    k_start = time.perf_counter()
                                    correction_start_event = correction_end_event = None
                                    if self.group11_fetch_mode in {"next_layer_prefetch", "temporal_qprev_prefetch"} and self._rag_profile_on():
                                        correction_start_event = torch.cuda.Event(enable_timing=True)
                                        correction_end_event = torch.cuda.Event(enable_timing=True)
                                        correction_start_event.record(torch.cuda.current_stream(device))
                                    dst_k = src_k.to(device, non_blocking=True)
                                    k_ms = (time.perf_counter() - k_start) * 1000.0
                                    v_start = time.perf_counter()
                                    dst_v = src_v.to(device, non_blocking=True)
                                    if correction_end_event is not None:
                                        correction_end_event.record(torch.cuda.current_stream(device))
                                    v_ms = (time.perf_counter() - v_start) * 1000.0
                                    if reuse:
                                        self._group11_reuse_cache[cache_key] = (dst_k, dst_v)
                                    if self.group11_profile is not None:
                                        self.group11_profile["CACHE_MISSES"] += 1
                                        if reuse:
                                            cache_bytes = sum(int(k.numel() * k.element_size() + vv.numel() * vv.element_size()) for k, vv in self._group11_reuse_cache.values())
                                            self.group11_profile["GPU_CACHE_PEAK_BYTES"] = max(int(self.group11_profile.get("GPU_CACHE_PEAK_BYTES", 0)), cache_bytes)
                                copy_ms = k_ms + v_ms
                                if archive is not None:
                                    copy_ms = archive_copy_ms
                                k_host_ms += k_ms
                                v_host_ms += v_ms
                                copied_bytes = int(src_k.numel() * src_k.element_size() + src_v.numel() * src_v.element_size())
                                if prefetched is None:
                                    self.runtime_counters["full_kv_h2d_bytes"] += copied_bytes
                                    self.runtime_counters["full_kv_h2d_calls"] += 2
                                self.history_fetch_trace.append({
                                    "history_id": int(k_idx), "batch": int(bi),
                                    "source_k_device": str(src_k.device),
                                    "source_v_device": str(src_v.device),
                                    "source_k_ptr": int(src_k.untyped_storage().data_ptr()),
                                    "source_v_ptr": int(src_v.untyped_storage().data_ptr()),
                                    "destination_k_ptr": int(dst_k.untyped_storage().data_ptr()),
                                    "destination_v_ptr": int(dst_v.untyped_storage().data_ptr()),
                                    "copied_bytes": copied_bytes,
                                    "layer": int(getattr(self, "_current_layer_index", -1)),
                                    "copy_time_ms": copy_ms,
                                    "source_pinned_memory": bool(src_k.is_pinned() and src_v.is_pinned()),
                                    "non_blocking": True,
                                    "cache_hit": cache_hit,
                                })
                                fetch_call_id = int(self.group11_profile.get("NUM_ATTENTION_CALLS", 0)) if self.group11_profile is not None else 0
                                self._rag_record_fetch(
                                    layer_id=int(layer_index), attention_call_id=fetch_call_id,
                                    history_id=int(k_idx),
                                    fetch_reason=("prefetch_correction" if self.group11_fetch_mode in {"next_layer_prefetch", "temporal_qprev_prefetch"} and prefetched is None else "demand"),
                                    fetch_bytes_k=int(src_k.numel() * src_k.element_size()),
                                    fetch_bytes_v=int(src_v.numel() * src_v.element_size()),
                                    fetch_work_ms=float(copy_ms), fetch_exposed_ms=float(copy_ms),
                                    is_first_fetch_in_call=(len(k_mem_unroped_list) == 0),
                                    is_duplicate=bool(cache_hit), is_cache_hit=bool(cache_hit),
                                    is_cache_miss=not bool(cache_hit),
                                    prefetch_hit_status=("correction" if self.group11_fetch_mode in {"next_layer_prefetch", "temporal_qprev_prefetch"} and prefetched is None else "not_applicable"))
                                if correction_start_event is not None:
                                    fetch_id = int(self.group11_profile["rag_fetch_events"][-1]["fetch_event_id"])
                                    self._rag_cuda_pending.append({"measurement_kind": "correction_copy", "fetch_event_id": fetch_id,
                                                                   "layer_id": int(layer_index), "history_id": int(k_idx),
                                                                   "start": correction_start_event, "end": correction_end_event})
                                if self.group11_profile is not None:
                                    self.group11_profile["NUM_CPU_KV_FETCH_CALLS"] += 1
                                    self.group11_profile["NUM_H2D_COPY_CALLS"] += 2
                                    self.group11_profile["TOTAL_FULL_KV_H2D_BYTES"] += copied_bytes
                                    self.group11_profile["EXPOSED_H2D_MS"] = float(self.group11_profile.get("EXPOSED_H2D_MS", 0.0)) + float(copy_ms)
                                    self.group11_profile["EXPOSED_H2D_CALLS"] = int(self.group11_profile.get("EXPOSED_H2D_CALLS", 0)) + 2
                                    self.group11_profile["h2d_rows"].append({"source_chunk_id": int(k_idx), "layer": int(getattr(self, "_current_layer_index", -1)), "bytes_K": int(src_k.numel()*src_k.element_size()), "bytes_V": int(src_v.numel()*src_v.element_size()), "total_bytes": copied_bytes, "copy_time_ms": copy_ms, "k_host_ms": k_ms, "v_host_ms": v_ms, "source_pinned_memory": bool(src_k.is_pinned() and src_v.is_pinned()), "non_blocking": True, "copy_stream": "default", "cache_hit": cache_hit})
                                k_mem_unroped_list.append(dst_k)
                                v_mem_list.append(dst_v)
                            
                        gather_start = time.perf_counter()
                        k_mem_unroped = torch.stack(k_mem_unroped_list, dim=0).view(b, k_sel * frame_seqlen, n, d)
                        v_mem = torch.stack(v_mem_list, dim=0).view(b, k_sel * frame_seqlen, n, d)
                        gather_ms = (time.perf_counter() - gather_start) * 1000.0
                        event_state = kv_cache.get("temporal_qprev_event_state", {}).get(
                            int(kv_cache.get("temporal_current_valid_invocation_id", -1))
                        )
                        if event_state is not None:
                            event_state["kv_assembly_ms"] = float(gather_ms)
                        
                        mem_grid_sizes = grid_sizes.clone()
                        mem_grid_sizes[:, 0] = k_sel
                        
                        rope_fn = lambda: causal_online_rope(
                            k_mem_unroped,
                            mem_grid_sizes, freqs, relative_frame_indices=torch.zeros(k_sel, dtype=torch.long, device=device)
                        ).type_as(v)
                        trace = cpt.ACTIVE_TRACE
                        k_mem = (trace.measure('ROPE', rope_fn) if trace else rope_fn())
                        fetch_end_to_end_ms = (time.perf_counter() - fetch_start) * 1000.0
                        trace = cpt.ACTIVE_TRACE
                        if trace:
                            fetch_ids_cpu = trace.measure(
                                "FETCH_SELECTED_IDS_DETACH_CPU",
                                lambda: memory_indices[0].detach().cpu(),
                            )
                            fetch_ids_list = trace.measure(
                                "FETCH_SELECTED_IDS_TOLIST",
                                lambda: fetch_ids_cpu.tolist(),
                            )
                        else:
                            fetch_ids_list = memory_indices[0].detach().cpu().tolist()
                        if self.group11_profile is not None:
                            self.group11_profile["fetch_phase_rows"].append({"layer": int(layer_index), "lookup_ms": lookup_ms, "k_to_device_ms": k_host_ms, "v_to_device_ms": v_host_ms, "gather_ms": gather_ms, "cat_ms": 0.0, "end_to_end_ms": fetch_end_to_end_ms, "selected_chunks": int(k_sel), "selected_ids": [int(x) for x in fetch_ids_list]})
                        if self.retrieval_backend == "draftmap_online" and self.draftmap_trace:
                            self.draftmap_trace[-1]["cpu_gather_ms"] = (time.perf_counter() - gather_start) * 1000.0
                            self.draftmap_trace[-1]["h2d_ms"] = 0.0

                k_parts = [k_sink]
                v_parts = [v_sink]
                
                if k_mem is not None:
                    k_parts.append(k_mem)
                    v_parts.append(v_mem)
                    
                if local_budget > 0 and local_start_for_window < local_end_index:
                    k_local = roped_temp_k[:, local_start_for_window:local_end_index]
                    v_local = temp_v[:, local_start_for_window:local_end_index]
                    k_parts.append(k_local)
                    v_parts.append(v_local)
                    
                cat_start = time.perf_counter()
                trace = cpt.ACTIVE_TRACE
                k_cat = (trace.measure('POST_GATHER_REPACK_K', lambda: torch.cat(k_parts, dim=1))
                         if trace else torch.cat(k_parts, dim=1))
                v_cat = (trace.measure('POST_GATHER_REPACK_V', lambda: torch.cat(v_parts, dim=1))
                         if trace else torch.cat(v_parts, dim=1))
                cat_ms = (time.perf_counter() - cat_start) * 1000.0
                if self.group11_profile is not None and self.group11_profile["fetch_phase_rows"]:
                    self.group11_profile["fetch_phase_rows"][-1]["cat_ms"] += cat_ms
                persistent_owner = (
                    kv_cache.get("compressed_history_archive") is not None
                    and self.group_runtime_mode in {
                        "group12_corrected", "group13_corrected",
                        "group14_corrected", "group15_corrected",
                    }
                )
                original_k_tokens = int(k_cat.shape[1])
                effective_sparse_ratio = 0.0 if self.group_runtime_mode in {
                    "group12_corrected", "group13_corrected",
                    "group14_corrected", "group15_corrected",
                } else self.group_sparse_ratio
                k_cat, v_cat, runtime_meta = prepare_attention_kv(
                    roped_query, k_cat, v_cat,
                    self.group_runtime_mode, effective_sparse_ratio,
                    persistent_owner_already_dequantized=persistent_owner)
                if archive_materialized_meta:
                    runtime_meta.update(archive_materialized_meta)
                runtime_meta["TRACE_LAYER"] = int(layer_index)
                runtime_meta["TRACE_DENOISING_STEP"] = self.current_denoising_step
                self.group_runtime_trace.append(runtime_meta)

                attn_start = time.perf_counter()
                fa_start_host = time.perf_counter()
                attn_ctx = record_function("G11_BF16_ATTN") if (v5_active and record_function is not None) else nullcontext()
                cuda_attn_token = None
                if self.unified_latency_profiler is not None:
                    cuda_attn_token = self.unified_latency_profiler.begin_cuda("ATTENTION_KERNEL")
                with attn_ctx:
                    x = attention(roped_query, k_cat, v_cat)
                fa_end_host = time.perf_counter()
                if self.retrieval_query_mode == "previous_q_prefetch" and kv_cache is not None:
                    event_state = kv_cache.get("temporal_qprev_event_state", {}).get(
                        int(kv_cache.get("temporal_current_valid_invocation_id", -1))
                    )
                    if event_state is not None:
                        event_state["fa_start_host"] = float(fa_start_host)
                        event_state["fa_end_host"] = float(fa_end_host)
                if self.unified_latency_profiler is not None:
                    self.unified_latency_profiler.end_cuda(cuda_attn_token, {
                        "call_id": len(self.unified_latency_profiler._events),
                        "generation_unit": int(current_start // max(1, 3 * frame_seqlen)),
                        "denoising_step": self.current_denoising_step,
                        "layer_id": int(layer_index),
                        "q_shape": list(roped_query.shape),
                        "k_shape": list(k_cat.shape),
                        "v_shape": list(v_cat.shape),
                        "dtype": str(k_cat.dtype),
                        "q_len": int(roped_query.shape[1]),
                        "original_k_len": original_k_tokens,
                        "actual_k_len": int(k_cat.shape[1]),
                        "qk_elements_original": int(roped_query.shape[1] * original_k_tokens * k_cat.shape[2]),
                        "qk_elements_actual": int(roped_query.shape[1] * k_cat.shape[1] * k_cat.shape[2]),
                        "retained_ratio": float(k_cat.shape[1] / max(1, original_k_tokens)),
                    })
                if self.retrieval_query_mode == "previous_q_prefetch" and kv_cache is not None:
                    # The current attention has consumed any target-tensor slots.
                    # Let the dedicated stream wait for this event before reusing
                    # those slots for P_(t+1,l), then launch the next prediction
                    # before output projection/FFN work continues.
                    consumer_event = torch.cuda.Event()
                    consumer_event.record(torch.cuda.current_stream(q.device))
                    kv_cache["temporal_qprev_consumer_event"] = consumer_event
                    self._schedule_temporal_next_from_current_q(
                        q, kv_cache, layer_index, frame_seqlen, current_start,
                        memory_indices)
                if self.group11_profile is not None:
                    self.group11_profile["NUM_ATTENTION_CALLS"] += 1
                    call_id = int(self.group11_profile["NUM_ATTENTION_CALLS"] - 1)
                    ev = [e for e in self.group11_profile.get("rag_fetch_events", []) if int(e.get("attention_call_id", -1)) == call_id]
                    route = self.group11_profile.get("draftmap_rows", [])[-1] if self.group11_profile.get("draftmap_rows") else {}
                    wrapper_ms = (time.perf_counter()-attn_start)*1000.0
                    self.group11_profile["rag_attention_rows"].append({
                        "case_id": os.environ.get("RAG_PROFILE_CASE_ID", "case_01"), "step_id": self.current_denoising_step,
                        "layer_id": int(layer_index), "attention_call_id": call_id,
                        "history_count": int(len(cpu_k_list)) if 'cpu_k_list' in locals() else 0,
                        "candidate_count": int(route.get("num_candidate_chunks", 0)), "selected_count": int(route.get("num_selected_chunks", 0)),
                        "fetch_event_count": len(ev), "fetch_bytes_total": sum(int(e["fetch_bytes_total"]) for e in ev),
                        "fetch_work_ms": sum(float(e["fetch_work_ms"]) for e in ev), "fetch_exposed_ms": sum(float(e["fetch_exposed_ms"]) for e in ev),
                        "fetch_hidden_ms": sum(float(e["fetch_hidden_ms"]) for e in ev),
                        "first_fetch_ms": next((float(e["fetch_work_ms"]) for e in ev if e.get("is_first_fetch_in_call")), 0.0),
                        "retrieval_work_ms": float(route.get("time_score_ms", 0.0)) + float(route.get("time_topk_ms", 0.0)),
                        "routing_work_ms": float(route.get("time_draft_q_pool_ms", 0.0)) + float(route.get("time_score_ms", 0.0)) + float(route.get("time_topk_ms", 0.0)),
                        "attention_wrapper_ms": wrapper_ms,
                        "prefetch_hit_count": sum(1 for e in ev if e.get("prefetch_hit_status") == "hit"),
                        "correction_fetch_count": sum(1 for e in ev if e.get("prefetch_hit_status") == "correction"),
                        "cache_hit_count": sum(1 for e in ev if e.get("is_cache_hit")), "cache_miss_count": sum(1 for e in ev if e.get("is_cache_miss")),
                    })
                    self.group11_profile["attention_rows"].append({"layer": int(layer_index), "q_tokens": int(roped_query.shape[1]), "kv_tokens": int(k_cat.shape[1]), "heads": int(k_cat.shape[2]), "dtype": str(k_cat.dtype), "cpu_wall_ms": wrapper_ms})
            else:
                window_start = max(0, local_end_index - self.max_attention_size)
                roped_temp_k, temp_v, runtime_meta = prepare_attention_kv(
                    roped_query, roped_temp_k[:, window_start:local_end_index],
                    temp_v[:, window_start:local_end_index],
                    self.group_runtime_mode, self.group_sparse_ratio,
                    apply_storage_quant=False)
                runtime_meta["TRACE_LAYER"] = int(layer_index)
                runtime_meta["TRACE_DENOISING_STEP"] = self.current_denoising_step
                self.group_runtime_trace.append(runtime_meta)
                attn_start = time.perf_counter()
                attn_ctx = record_function("G11_BF16_ATTN") if (v5_active and record_function is not None) else nullcontext()
                cuda_attn_token = None
                if self.unified_latency_profiler is not None:
                    cuda_attn_token = self.unified_latency_profiler.begin_cuda("ATTENTION_KERNEL")
                with attn_ctx:
                    x = attention(roped_query, roped_temp_k, temp_v)
                if self.unified_latency_profiler is not None:
                    self.unified_latency_profiler.end_cuda(cuda_attn_token, {
                        "call_id": len(self.unified_latency_profiler._events),
                        "generation_unit": int(current_start // max(1, 3 * frame_seqlen)),
                        "denoising_step": self.current_denoising_step,
                        "layer_id": int(layer_index),
                        "q_shape": list(roped_query.shape),
                        "k_shape": list(roped_temp_k.shape),
                        "v_shape": list(temp_v.shape),
                        "dtype": str(roped_temp_k.dtype),
                        "q_len": int(roped_query.shape[1]),
                        "original_k_len": int(roped_temp_k.shape[1]),
                        "actual_k_len": int(roped_temp_k.shape[1]),
                        "qk_elements_original": int(roped_query.shape[1] * roped_temp_k.shape[1] * roped_temp_k.shape[2]),
                        "qk_elements_actual": int(roped_query.shape[1] * roped_temp_k.shape[1] * roped_temp_k.shape[2]),
                        "retained_ratio": 1.0,
                    })
                if self.group11_profile is not None:
                    self.group11_profile["NUM_ATTENTION_CALLS"] += 1
                    self.group11_profile["attention_rows"].append({"layer": int(layer_index), "q_tokens": int(roped_query.shape[1]), "kv_tokens": int(roped_temp_k.shape[1]), "heads": int(roped_temp_k.shape[2]), "dtype": str(roped_temp_k.dtype), "cpu_wall_ms": (time.perf_counter()-attn_start)*1000.0})

        # output
        self._commit_draft_q_history(kv_cache)
        output_start = time.perf_counter()
        x = x.flatten(2)
        output_ctx = record_function("G11_ATTN_OUTPUT_PROJECTION") if (v5_active and record_function is not None) else nullcontext()
        with output_ctx:
            x = self.o(x)
        if self.group11_profile is not None:
            self.group11_profile["model_phase_ms"]["attention_output_projection"] += (time.perf_counter() - output_start) * 1000.0
        
        # Return both output and cache update info
        if kv_cache is not None:
            return x, (current_end, local_end_index, cache_update_info)
        else:
            return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 memory_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, memory_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        sink_recache_after_switch=False,
        memory_indices=None,
        layer_index=0,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        block_attn_start = time.perf_counter()
        wrapper_ctx = record_function("G11_WRAPPER") if (getattr(self.self_attn, "_v5_active", False) and record_function is not None) else nullcontext()
        with wrapper_ctx:
            self_attn_result = self.self_attn(
                (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
                seq_lens, grid_sizes, freqs, block_mask, kv_cache, current_start, cache_start, sink_recache_after_switch,
                memory_indices=memory_indices, layer_index=layer_index)
        
        if kv_cache is not None:
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None
        if self.self_attn.group11_profile is not None:
            self.self_attn.group11_profile["model_phase_ms"]["attention_wrapper"] += (time.perf_counter() - block_attn_start) * 1000.0

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        ffn_start = time.perf_counter()
        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        if self.self_attn.group11_profile is not None:
            self.self_attn.group11_profile["model_phase_ms"]["FFN_MLP_and_cross_attention"] += (time.perf_counter() - ffn_start) * 1000.0
        
        if cache_update_info is not None:
            # cache_update_info is already in the format (current_end, local_end_index, cache_update_info)
            return x, cache_update_info
        else:
            return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 memory_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            memory_size (`int`, *optional*, defaults to 0):
                Size of the memory pool (number of recent frames to use as top-k context)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    local_attn_size, sink_size, memory_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        return block_mask

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """
        Applies cache updates collected from multiple blocks.
        
        For causal online RoPE, this stores UN-ROPED K values in the cache.
        RoPE is applied dynamically during attention based on the token's current
        relative position in the sliding window.
        
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                
                if update_info["action"] == "roll_and_insert":
                    # Apply rolling update
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # Perform the rolling operation
                    cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v
                    
                    if "evicted_k_frames" in update_info and update_info["evicted_k_frames"]:
                        archive = cache.get("compressed_history_archive")
                        if archive is not None:
                            for kf, vf in zip(update_info["evicted_k_frames"], update_info["evicted_v_frames"]):
                                rec = archive.append(kf[:, 0].to(torch.bfloat16), vf[:, 0].to(torch.bfloat16), layer_id=int(block_index), chunk_id=len(cache.get("compressed_history_entries", [])), history_id=len(cache.get("compressed_history_entries", [])), valid_tokens=int(kf.shape[2]))
                                cache.setdefault("compressed_history_entries", []).append(rec)
                                cache["evicted_compressed_entries"] = int(cache.get("evicted_compressed_entries", 0)) + 1
                        else:
                            cache.setdefault("cpu_k_frames", []).extend(update_info["evicted_k_frames"])
                            cache.setdefault("cpu_v_frames", []).extend(update_info["evicted_v_frames"])
                        cache.setdefault("gpu_draft_k_frames", []).extend(update_info.get("evicted_draft_k_frames", []))
                        
                elif update_info["action"] == "direct_insert":
                    # Direct insert
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    write_start_index = update_info.get("write_start_index", local_start_index)
                    write_end_index = update_info.get("write_end_index", local_end_index)
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    
                    # Insert new key/value
                    if write_end_index > write_start_index and new_k.shape[1] == (write_end_index - write_start_index):
                        cache["k"][:, write_start_index:write_end_index] = new_k
                        cache["v"][:, write_start_index:write_end_index] = new_v

                new_k_frames = update_info.get("new_draft_k_frames", [])
                if update_info["action"] == "roll_and_insert":
                    sink_frames = update_info["sink_tokens"] // 1560
                    evicted_frames = update_info["num_evicted_tokens"] // 1560
                    rolled_frames = update_info["num_rolled_tokens"] // 1560
                    old_k = cache.get("local_draft_k_frames", [])
                    start = sink_frames + evicted_frames
                    cache["local_draft_k_frames"] = old_k[:sink_frames] + old_k[start:start + rolled_frames] + list(new_k_frames)
                else:
                    start = update_info.get("write_start_index", 0) // 1560
                    old_k = list(cache.get("local_draft_k_frames", []))
                    while len(old_k) < start:
                        old_k.append(None)
                    old_k[start:start + len(new_k_frames)] = list(new_k_frames)
                    cache["local_draft_k_frames"] = old_k
            
            # Update indices: do not roll back pointers during recomputation
            is_recompute = False if update_info is None else update_info.get("is_recompute", False)
            if not is_recompute:
                kv_cache[block_index]["global_end_index"].fill_(current_end)
                kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        sink_recache_after_switch=False,
        memory_indices=None
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
        

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32
        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))
        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            sink_recache_after_switch=sink_recache_after_switch,
            memory_indices=memory_indices
        )
        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        cache_update_info = None
        cache_update_infos = []  # Collect cache update info for all blocks
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "layer_index": block_index
                    }
                )
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "layer_index": block_index
                    }
                )
                result = block(x, **kwargs)
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Extract base info for subsequent blocks (without concrete cache update details)
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
        # log_gpu_memory(f"in _forward_inference: {x[0].device}")
        # After all blocks are processed, apply cache updates in a single pass
        if kv_cache is not None and cache_update_infos:
            self._apply_cache_updates(kv_cache, cache_update_infos)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        pass
        raise NotImplementedError()
    
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            assert seq_lens_clean.max() <= seq_len
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros_like(t)
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(
                1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)
        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            # memory_indices is only used in inference, remove for train
            return self._forward_inference(*args, **kwargs)
        else:
            kwargs.pop('memory_indices', None)
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
