from pathlib import Path
p = Path('/data/zxl/LongLive-RAG-group11_15_h200/utils/compressed_history_archive.py')
s = p.read_text()
s = s.replace("""        self.last_dequant_bytes = 0
""", """        self.last_dequant_bytes = 0
        self.last_promotion_enqueue_ms = 0.0
        self.last_dequant_enqueue_ms = 0.0
        self.last_join_wait_ms = 0.0
        self.last_materialization_host_ms = 0.0
""", 1)
s = s.replace("""        self.last_dequant_bytes = 0
        use_async = mode == "async_overlap"
""", """        self.last_dequant_bytes = 0
        self.last_promotion_enqueue_ms = 0.0
        self.last_dequant_enqueue_ms = 0.0
        self.last_join_wait_ms = 0.0
        materialize_start = time.perf_counter()
        use_async = mode == "async_overlap"
""", 1)
s = s.replace("""            dequant_stream = torch.cuda.Stream(device=device)
            with torch.cuda.stream(promotion_stream):
""", """            dequant_stream = torch.cuda.Stream(device=device)
            phase_start = time.perf_counter()
            with torch.cuda.stream(promotion_stream):
""", 1)
s = s.replace("""                    self.last_promotion_bytes += int(rec.k_payload.numel() * rec.k_payload.element_size() + rec.v_payload.numel() * rec.v_payload.element_size())
            promotion_event
""", """                    self.last_promotion_bytes += int(rec.k_payload.numel() * rec.k_payload.element_size() + rec.v_payload.numel() * rec.v_payload.element_size())
            self.last_promotion_enqueue_ms = (time.perf_counter() - phase_start) * 1000.0
            promotion_event
""", 1)
s = s.replace("""            promotion_event = torch.cuda.Event(); promotion_event.record(promotion_stream)
            with torch.cuda.stream(dequant_stream):
""", """            promotion_event = torch.cuda.Event(); promotion_event.record(promotion_stream)
            phase_start = time.perf_counter()
            with torch.cuda.stream(dequant_stream):
""", 1)
s = s.replace("""                    self.last_dequant_bytes += int(k_out[p].numel() * k_out[p].element_size() + v_out[p].numel() * v_out[p].element_size())
            dequant_event
""", """                    self.last_dequant_bytes += int(k_out[p].numel() * k_out[p].element_size() + v_out[p].element_size() * v_out[p].numel())
            self.last_dequant_enqueue_ms = (time.perf_counter() - phase_start) * 1000.0
            dequant_event
""", 1)
s = s.replace("""                if p in positions:
                    k_out[p] = rec.k_payload.to(device=device, dtype=torch.bfloat16, non_blocking=True)
""", """                if p in positions:
                    phase_start = time.perf_counter()
                    k_out[p] = rec.k_payload.to(device=device, dtype=torch.bfloat16, non_blocking=True)
""", 1)
s = s.replace("""                    self.last_promotion_bytes += int(rec.k_payload.numel() * rec.k_payload.element_size() + rec.v_payload.numel() * rec.v_payload.element_size())
                else:
                    k_out[p], v_out[p] = self.fetch(rec, device)
""", """                    self.last_promotion_bytes += int(rec.k_payload.numel() * rec.k_payload.element_size() + rec.v_payload.numel() * rec.v_payload.element_size())
                    self.last_promotion_enqueue_ms += (time.perf_counter() - phase_start) * 1000.0
                else:
                    phase_start = time.perf_counter()
                    k_out[p], v_out[p] = self.fetch(rec, device)
                    self.last_dequant_enqueue_ms += (time.perf_counter() - phase_start) * 1000.0
""", 1)
s = s.replace("""        self.last_transient_bf16_bytes = self.last_promotion_bytes + self.last_dequant_bytes
        self.last_metadata = {
""", """        self.last_transient_bf16_bytes = self.last_promotion_bytes + self.last_dequant_bytes
        self.last_materialization_host_ms = (time.perf_counter() - materialize_start) * 1000.0
        self.last_metadata = {
""", 1)
s = s.replace("""            "NO_GLOBAL_SYNC_IN_HOT_PATH": True,
""", """            "NO_GLOBAL_SYNC_IN_HOT_PATH": True,
            "PROMOTION_H2D_HOST_ENQUEUE_MS": self.last_promotion_enqueue_ms,
            "LOWBIT_DEQUANT_HOST_ENQUEUE_MS": self.last_dequant_enqueue_ms,
            "JOIN_WAIT_HOST_MS": self.last_join_wait_ms,
            "MATERIALIZATION_HOST_MS": self.last_materialization_host_ms,
            "MATERIALIZATION_TIMING_SEMANTICS": "host_observed_enqueue_not_cuda_work",
            "SOURCE_BY_ID": {str(i): ("CPU_BF16" if i in promoted_ids else "GPU_LOWBIT_DEQUANT") for i in ids},
""", 1)
p.write_text(s)

