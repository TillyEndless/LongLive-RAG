import pathlib

p = pathlib.Path("/data/zxl/LongLive-RAG-group11_15_h200/wan/modules/causal_model_latentmem.py")
s = p.read_text()
repls = [
    ("from utils.h200_group_runtime import prepare_attention_kv\n", "from utils.h200_group_runtime import prepare_attention_kv\nfrom utils.compressed_history_archive import CompressedHistoryArchive\n"),
    ('        cpu_k = kv_cache.get("cpu_k_frames", [])\n', '        cpu_k = kv_cache.get("compressed_history_entries", kv_cache.get("cpu_k_frames", []))\n'),
    ('''                    cpu_k_list = kv_cache.get("cpu_k_frames", [])
                    cpu_v_list = kv_cache.get("cpu_v_frames", [])
                    
                    if len(cpu_k_list) > 0:
''', '''                    compressed_entries = kv_cache.get("compressed_history_entries", [])
                    cpu_k_list = kv_cache.get("cpu_k_frames", [])
                    cpu_v_list = kv_cache.get("cpu_v_frames", [])
                    
                    if len(cpu_k_list) > 0 or len(compressed_entries) > 0:
'''),
    ('''                                src_k = cpu_k_list[k_idx][bi, 0]
                                src_v = cpu_v_list[k_idx][bi, 0]
                                copy_start = time.perf_counter()
                                dst_k = src_k.to(device, non_blocking=True)
                                dst_v = src_v.to(device, non_blocking=True)
                                copy_ms = (time.perf_counter() - copy_start) * 1000.0
                                copied_bytes = int(src_k.numel() * src_k.element_size() + src_v.numel() * src_v.element_size())
''', '''                                if compressed_entries:
                                    entry = compressed_entries[int(k_idx)]
                                    dst_k_full, dst_v_full = kv_cache["compressed_history_archive"].fetch(entry, device)
                                    dst_k = dst_k_full[bi]
                                    dst_v = dst_v_full[bi]
                                    copy_ms = 0.0
                                    copied_bytes = int(entry.persistent_bytes())
                                    self.runtime_counters["retrieved_archived_entries"] = self.runtime_counters.get("retrieved_archived_entries", 0) + 1
                                    dequant_bytes = int(dst_k.numel() * dst_k.element_size() + dst_v.numel() * dst_v.element_size())
                                    self.runtime_counters["transient_dequant_gpu_bytes"] = self.runtime_counters.get("transient_dequant_gpu_bytes", 0) + dequant_bytes
                                    self.runtime_counters["transient_dequant_gpu_peak_bytes"] = max(self.runtime_counters.get("transient_dequant_gpu_peak_bytes", 0), self.runtime_counters["transient_dequant_gpu_bytes"])
                                else:
                                    src_k = cpu_k_list[k_idx][bi, 0]
                                    src_v = cpu_v_list[k_idx][bi, 0]
                                    copy_start = time.perf_counter()
                                    dst_k = src_k.to(device, non_blocking=True)
                                    dst_v = src_v.to(device, non_blocking=True)
                                    copy_ms = (time.perf_counter() - copy_start) * 1000.0
                                    copied_bytes = int(src_k.numel() * src_k.element_size() + src_v.numel() * src_v.element_size())
'''),
                                    entry = compressed_entries[int(k_idx)]
                                    dst_k_full, dst_v_full = kv_cache["compressed_history_archive"].fetch(entry, device)
                                    dst_k = dst_k_full[bi]
                                    dst_v = dst_v_full[bi]
                                    copied_bytes = int(entry.persistent_bytes())
                                    self.runtime_counters["retrieved_archived_entries"] = self.runtime_counters.get("retrieved_archived_entries", 0) + 1
                                    dequant_bytes = int(dst_k.numel() * dst_k.element_size() + dst_v.numel() * dst_v.element_size())
                                    self.runtime_counters["transient_dequant_gpu_bytes"] = self.runtime_counters.get("transient_dequant_gpu_bytes", 0) + dequant_bytes
                                    self.runtime_counters["transient_dequant_gpu_peak_bytes"] = max(self.runtime_counters.get("transient_dequant_gpu_peak_bytes", 0), self.runtime_counters["transient_dequant_gpu_bytes"])
                                else:
                                    src_k = cpu_k_list[k_idx][bi, 0]
                                    src_v = cpu_v_list[k_idx][bi, 0]
                                    dst_k = src_k.to(device, non_blocking=True)
                                    dst_v = src_v.to(device, non_blocking=True)
                                    copied_bytes = int(src_k.numel() * src_k.element_size() + src_v.numel() * src_v.element_size())
'''),
    ('''                                    "source_k_device": str(src_k.device),
                                    "source_v_device": str(src_v.device),
                                    "source_k_ptr": int(src_k.untyped_storage().data_ptr()),
                                    "source_v_ptr": int(src_v.untyped_storage().data_ptr()),
''', '''                                    "source_k_device": "cpu_archive" if compressed_entries else str(src_k.device),
                                    "source_v_device": "cpu_archive" if compressed_entries else str(src_v.device),
                                    "source_k_ptr": 0 if compressed_entries else int(src_k.untyped_storage().data_ptr()),
                                    "source_v_ptr": 0 if compressed_entries else int(src_v.untyped_storage().data_ptr()),
'''),
    ('''                    if "evicted_k_frames" in update_info and update_info["evicted_k_frames"]:
                        cache.setdefault("cpu_k_frames", []).extend(update_info["evicted_k_frames"])
                        cache.setdefault("cpu_v_frames", []).extend(update_info["evicted_v_frames"])
''', '''                    if "evicted_k_frames" in update_info and update_info["evicted_k_frames"]:
                        if cache.get("compressed_history_archive") is not None:
                            archive = cache["compressed_history_archive"]
                            entries = cache.setdefault("compressed_history_entries", [])
                            for k_frame, v_frame in zip(update_info["evicted_k_frames"], update_info["evicted_v_frames"]):
                                rec = archive.append(
                                    k_frame[:, 0].contiguous(), v_frame[:, 0].contiguous(),
                                    layer_id=block_index, chunk_id=len(entries),
                                    history_id=len(entries), valid_tokens=int(k_frame.shape[2]))
                                entries.append(rec)
                                cache["evicted_compressed_entries"] = cache.get("evicted_compressed_entries", 0) + 1
                        else:
                            cache.setdefault("cpu_k_frames", []).extend(update_info["evicted_k_frames"])
                            cache.setdefault("cpu_v_frames", []).extend(update_info["evicted_v_frames"])
'''),
]
for old, new in repls:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"anchor count {n}: {old[:80]!r}")
    s = s.replace(old, new)
p.write_text(s)
print(f"edited {p}")
