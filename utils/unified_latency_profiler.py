"""Optional low-perturbation latency composition collector.

CUDA events are recorded without per-call synchronization.  The owning
pipeline calls finalize() once at the end of the profiled inference interval.
"""
from __future__ import annotations

import time
from typing import Any

import torch


class UnifiedLatencyProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self.reset()

    def reset(self):
        self._events = []
        self.records = []
        self.started_at = time.perf_counter()

    def begin_cuda(self, label: str):
        if not self.enabled or not torch.cuda.is_available():
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream())
        return label, start, end

    def end_cuda(self, token, metadata: dict[str, Any]):
        if token is None:
            return
        label, start, end = token
        end.record(torch.cuda.current_stream())
        self._events.append((label, start, end, dict(metadata)))

    def finalize(self):
        if not self.enabled:
            return {"enabled": False, "records": [], "synchronization": "none"}
        if self._events:
            # One synchronization at the end of the bounded profile interval;
            # never synchronize inside the attention hot loop.
            torch.cuda.current_stream().synchronize()
        records = []
        for label, start, end, metadata in self._events:
            row = dict(metadata)
            row["category"] = label
            row["cuda_ms"] = float(start.elapsed_time(end))
            records.append(row)
        self.records = records
        return {
            "enabled": True,
            "synchronization": "one current-stream synchronize at finalize",
            "records": records,
        }
