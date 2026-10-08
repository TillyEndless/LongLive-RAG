from pathlib import Path
p=Path('/data/zxl/LongLive-RAG-group11_15_h200/utils/compressed_history_archive.py')
s=p.read_text()
old = """        self.last_dequant_bytes = 0
        use_async = mode == "async_overlap" """
new = """        self.last_dequant_bytes = 0
        self.last_promotion_enqueue_ms = 0.0
        self.last_dequant_enqueue_ms = 0.0
        self.last_join_wait_ms = 0.0
        materialize_start = time.perf_counter()
        use_async = mode == "async_overlap" """
if old not in s:
    raise SystemExit("reset anchor missing")
s=s.replace(old,new,1)
p.write_text(s)

