"""Debug-only same-state serial/unified replay for Group14/15."""
from __future__ import annotations

import hashlib
import time
import torch

from wan.modules.attention import attention
from utils.persistent_draftmap import route_candidate_chunks
from utils.compressed_history_archive import CompressedHistoryArchive
from utils.attention_fetch_plan import make_plan, submit as submit_fetch, wait as wait_fetch


def _ids(x):
    return [int(v) for v in x]


def _bytes(t):
    return int(t.numel() * t.element_size())


def _digest(t):
    a = t.detach().float().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(a).hexdigest()


def run_same_state_replay(*, q, roped_query, temp_k, temp_v, k_sink, v_sink,
                          local_start_for_window, local_end_index, local_budget,
                          frame_seqlen, kv_cache, memory_indices, grid_sizes,
                          freqs, device, value_dtype, q_sparse_ratio,
                          promotion_ratio, unified_candidate_k,
                          unified_candidate_v, unified_meta, unified_handle):
    """Replay one captured call; does not alter the production path."""
    t0 = time.perf_counter()
    history_ids = _ids(memory_indices[0].detach().cpu().tolist())
    local_frame_ids = list(range(int(local_start_for_window // frame_seqlen),
                                 int(local_end_index // frame_seqlen))) if (
        local_budget > 0 and local_start_for_window < local_end_index) else []
    history_count = len(history_ids)
    local_count = len(local_frame_ids)
    archive = kv_cache.get("compressed_history_archive")
    entries = kv_cache.get("compressed_history_entries", [])
    hs = {}
    for idx in history_ids:
        if archive is not None and 0 <= idx < len(entries):
            rec = entries[idx]
            hs[idx] = (rec.k_payload, rec.v_payload)
    local_ks = kv_cache.get("cpu_local_k_frames", [])
    local_vs = kv_cache.get("cpu_local_v_frames", [])
    ps = {i: (local_ks[i], local_vs[i]) for i in range(local_count)
          if i < len(local_ks) and i < len(local_vs)
          and local_ks[i] is not None and local_vs[i] is not None}

    # A: exact serial two-wait materialization from the same source snapshot.
    serial_history = {i: (k.to(device=device, dtype=torch.bfloat16),
                          v.to(device=device, dtype=torch.bfloat16))
                      for i, (k, v) in hs.items()}
    fallback = {int(fid): (temp_k[:, local_start_for_window + off * frame_seqlen:
                                  local_start_for_window + (off + 1) * frame_seqlen],
                           temp_v[:, local_start_for_window + off * frame_seqlen:
                                  local_start_for_window + (off + 1) * frame_seqlen])
                for off, fid in enumerate(local_frame_ids)}
    store = kv_cache.get("local_lowbit_store")
    if store is not None:
        sk, sv = store.materialize_frames(local_frame_ids, None, fallback=fallback)
    elif local_frame_ids:
        sk = torch.cat([fallback[i][0] for i in local_frame_ids], dim=1)
        sv = torch.cat([fallback[i][1] for i in local_frame_ids], dim=1)
    else:
        sk, sv = temp_k[:, :0], temp_v[:, :0]
    if history_ids:
        ku = torch.stack([serial_history[i][0][bi] for bi in range(q.shape[0]) for i in history_ids], dim=0)
        vu = torch.stack([serial_history[i][1][bi] for bi in range(q.shape[0]) for i in history_ids], dim=0)
        ku = ku.reshape(q.shape[0], -1, q.shape[2], q.shape[3])
        vu = vu.reshape(q.shape[0], -1, q.shape[2], q.shape[3])
        mg = grid_sizes.clone(); mg[:, 0] = history_count
        from wan.modules.causal_model_latentmem import causal_online_rope
        skh = causal_online_rope(ku, mg, freqs,
            relative_frame_indices=torch.zeros(history_count, dtype=torch.long, device=device)).type_as(value_dtype)
    else:
        skh = q.new_empty((q.shape[0], 0, q.shape[2], q.shape[3])); vu = skh
    if local_frame_ids:
        lg = grid_sizes.new_tensor([[len(local_frame_ids), grid_sizes[0, 1], grid_sizes[0, 2]]])
        from wan.modules.causal_model_latentmem import causal_online_rope
        skl = causal_online_rope(sk, lg, freqs,
            relative_frame_indices=torch.tensor(local_frame_ids, dtype=torch.long, device=device)).type_as(value_dtype)
    else:
        skl = sk
    serial_full_k = torch.cat([skh, skl], dim=1)
    serial_full_v = torch.cat([vu, sv], dim=1)
    mandatory = history_count + local_count - 1
    sk_route, sv_route, smeta = route_candidate_chunks(
        roped_query, serial_full_k, serial_full_v, q_sparse_ratio,
        frame_seqlen, mandatory_chunk_ids=(mandatory,))
    retained = [x - history_count for x in smeta.get("ROUTE_SELECTED_CHUNK_IDS", []) if x >= history_count]
    count = max(0, min(len(retained), int(round(len(retained) * promotion_ratio))))
    scores = smeta.get("ROUTE_CHUNK_SCORES", [])
    promoted = sorted(retained, key=lambda i: (float(scores[i]) if i < len(scores) else float(i), i), reverse=True)[:count]
    if not history_ids or not promoted or not retained or len(retained) >= len(history_ids) + local_count:
        return {"status": "SKIP", "reason": "call does not contain history, promotion, and sparse-drop simultaneously"}
    p_sync = {i: (k.to(device=device, dtype=torch.bfloat16), v.to(device=device, dtype=torch.bfloat16)) for i, (k, v) in ps.items() if i in promoted}
    one = grid_sizes.new_tensor([[1, grid_sizes[0, 1], grid_sizes[0, 2]]])
    from wan.modules.causal_model_latentmem import causal_online_rope
    for out_pos, cid in enumerate(smeta.get("ROUTE_SELECTED_CHUNK_IDS", [])):
        lid = cid - history_count
        if lid not in p_sync:
            continue
        pk, pv = p_sync[lid]
        pk = causal_online_rope(pk.to(device, dtype=roped_query.dtype), one, freqs,
             relative_frame_indices=torch.tensor([int(local_start_for_window // frame_seqlen) + lid], device=device)).type_as(roped_query)
        pv = pv.to(device, dtype=value_dtype.dtype)
        lo, hi = out_pos * frame_seqlen, (out_pos + 1) * frame_seqlen
        sk_route[:, lo:hi] = pk; sv_route[:, lo:hi] = pv

    # B is the already executed unified result from the same captured state.
    serial_ids = _ids(smeta.get("ROUTE_SELECTED_CHUNK_IDS", []))
    unified_ids = _ids(unified_meta.get("ROUTE_SELECTED_CHUNK_IDS", []))
    serial_promo = _ids(promoted)
    unified_promo = _ids(unified_meta.get("PROMOTED_LOCAL_IDS", []))
    serial_visible = serial_ids
    unified_visible = _ids(unified_meta.get("FINAL_VISIBLE_IDS", unified_ids))
    if serial_ids != unified_ids:
        return {"status": "FAIL", "first_divergence": "sparse_ids", "history_selected_ids": history_ids,
                "serial_sparse_ids": serial_ids, "unified_sparse_ids": unified_ids}
    if serial_promo != unified_promo:
        return {"status": "FAIL", "first_divergence": "promotion_ids", "serial_promotion_ids": serial_promo,
                "unified_promotion_ids": unified_promo}
    if serial_visible != unified_visible:
        return {"status": "FAIL", "first_divergence": "final_visible_ids", "serial_final_visible_ids": serial_visible,
                "unified_final_visible_ids": unified_visible}
    uk = unified_candidate_k.detach().clone(); uv = unified_candidate_v.detach().clone()
    kdiff = (sk_route.float() - uk.float()).abs()
    vdiff = (sv_route.float() - uv.float()).abs()
    if k_sink is not None and k_sink.shape[1] > 0:
        ao = attention(roped_query, torch.cat([k_sink, sk_route], 1), torch.cat([v_sink, sv_route], 1))
        bo = attention(roped_query, torch.cat([k_sink, uk], 1), torch.cat([v_sink, uv], 1))
    else:
        ao = attention(roped_query, sk_route, sv_route)
        bo = attention(roped_query, uk, uv)
    diff = (ao.float() - bo.float()).abs()
    rel = float(torch.linalg.vector_norm((ao - bo).float()) / torch.clamp(torch.linalg.vector_norm(ao.float()), min=1e-12))
    allclose = bool(torch.allclose(ao, bo, rtol=1e-3, atol=1e-3))
    def serial_trial():
        if device.type == "cuda": torch.cuda.synchronize(device)
        start = time.perf_counter()
        for k, v in hs.values():
            k.to(device=device, dtype=torch.bfloat16, non_blocking=False)
            v.to(device=device, dtype=torch.bfloat16, non_blocking=False)
        if store is not None:
            store.materialize_frames(local_frame_ids, None, fallback=fallback)
        _ = attention(roped_query, torch.cat([k_sink, sk_route], 1) if k_sink is not None and k_sink.shape[1] else sk_route,
                      torch.cat([v_sink, sv_route], 1) if v_sink is not None and v_sink.shape[1] else sv_route)
        if device.type == "cuda": torch.cuda.synchronize(device)
        return (time.perf_counter() - start) * 1000.0

    def unified_trial():
        if device.type == "cuda": torch.cuda.synchronize(device)
        start = time.perf_counter()
        plan = make_plan(history_ids, list(ps), hs, ps)
        uh = wait_fetch(submit_fetch(plan, device), device)
        if store is not None:
            store.materialize_frames(local_frame_ids, None, fallback=fallback)
        _ = attention(roped_query, torch.cat([k_sink, uk], 1) if k_sink is not None and k_sink.shape[1] else uk,
                      torch.cat([v_sink, uv], 1) if v_sink is not None and v_sink.shape[1] else uv)
        if device.type == "cuda": torch.cuda.synchronize(device)
        return (time.perf_counter() - start) * 1000.0, uh

    serial_times = [serial_trial() for _ in range(4)]
    unified_times = []
    unified_wait_ms = []
    for _ in range(4):
        elapsed, uh = unified_trial()
        unified_times.append(elapsed)
        unified_wait_ms.append(float(uh.exposed_wait_s) * 1000.0)
    serial_med = float(torch.tensor(serial_times[1:]).median().item())
    unified_med = float(torch.tensor(unified_times[1:]).median().item())
    speedup = serial_med / unified_med if unified_med > 0 else None
    return {"status":"PASS" if allclose else "FAIL",
            "first_divergence": None if allclose else "attention_output",
            "history_selected_ids":history_ids,"sparse_retained_ids":serial_ids,
            "promotion_ids":serial_promo,"final_visible_ids":serial_visible,
            "max_abs_diff":float(diff.max().item()),"mean_abs_diff":float(diff.mean().item()),
            "relative_l2":rel,"allclose_pass":allclose,
            "candidate_k_max_abs_diff":float(kdiff.max().item()),
            "candidate_v_max_abs_diff":float(vdiff.max().item()),
            "SERIAL_MEDIAN_MS":serial_med,
            "UNIFIED_MEDIAN_MS":unified_med,
            "H2D_EXPOSED_WAIT_MS":float(torch.tensor(unified_wait_ms[1:]).median().item()),
            "FETCH_BARRIER_COUNT":1,
            "SPEEDUP":speedup,
            "microbenchmark_repeats":3,
            "microbenchmark_scope":"same-state fetch/materialization plus attention; excludes model/snapshot load",
            "snapshot_digest_q":_digest(q),
            "snapshot_digest_local_lowbit_k":_digest(temp_k[:, local_start_for_window:local_end_index]),
            "snapshot_digest_local_lowbit_v":_digest(temp_v[:, local_start_for_window:local_end_index]),
            "replay_elapsed_s":time.perf_counter()-t0}
