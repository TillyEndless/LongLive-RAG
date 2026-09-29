"""Shared v2 history/promotion fetch scheduler.

The plan keeps logical fetch classes separate while submitting all CPU-BF16
copies before one consumer-side wait.  It deliberately does not own any
historical GPU cache and does not change attention semantics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import torch


@dataclass
class FetchPlan:
    history_ids: list[int] = field(default_factory=list)
    promotion_ids: list[int] = field(default_factory=list)
    history_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    promotion_sources: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)


@dataclass
class FetchHandle:
    plan: FetchPlan
    history: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    promotion: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    stream: object | None = None
    event: object | None = None
    start_event: object | None = None
    host_enqueue_s: float = 0.0
    cuda_work_s: float = 0.0
    exposed_wait_s: float = 0.0
    history_bytes: int = 0
    promotion_bytes: int = 0
    physical_copy_count: int = 0


def _pinned(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().contiguous()
    if x.device.type == "cpu" and torch.cuda.is_available() and not x.is_pinned():
        x = x.pin_memory()
    return x


def make_plan(history_ids, promotion_ids, history_sources, promotion_sources) -> FetchPlan:
    return FetchPlan(
        history_ids=[int(x) for x in history_ids],
        promotion_ids=[int(x) for x in promotion_ids],
        history_sources=dict(history_sources),
        promotion_sources=dict(promotion_sources),
    )


def submit(plan: FetchPlan, device: torch.device) -> FetchHandle:
    """Submit both logical fetch classes and create one consumer barrier."""
    handle = FetchHandle(plan=plan)
    start = time.perf_counter()
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    stream = torch.cuda.Stream(device=device) if use_cuda else None
    handle.stream = stream

    def copy_one(src, dst_dict, key, is_history):
        k, v = src
        k, v = _pinned(k), _pinned(v)
        dk = k.to(device=device, dtype=torch.bfloat16, non_blocking=use_cuda)
        dv = v.to(device=device, dtype=torch.bfloat16, non_blocking=use_cuda)
        dst_dict[int(key)] = (dk, dv)
        nbytes = int(k.numel() * k.element_size() + v.numel() * v.element_size())
        if is_history:
            handle.history_bytes += nbytes
        else:
            handle.promotion_bytes += nbytes
        handle.physical_copy_count += 2

    if stream is not None:
        with torch.cuda.stream(stream):
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
            for key in plan.history_ids:
                if key in plan.history_sources:
                    copy_one(plan.history_sources[key], handle.history, key, True)
            for key in plan.promotion_ids:
                if key in plan.promotion_sources:
                    copy_one(plan.promotion_sources[key], handle.promotion, key, False)
            event = torch.cuda.Event(enable_timing=True)
            event.record(stream)
            handle.event = event
            handle.start_event = start_event
    else:
        for key in plan.history_ids:
            if key in plan.history_sources:
                copy_one(plan.history_sources[key], handle.history, key, True)
        for key in plan.promotion_ids:
            if key in plan.promotion_sources:
                copy_one(plan.promotion_sources[key], handle.promotion, key, False)
    handle.host_enqueue_s = time.perf_counter() - start
    return handle


def wait(handle: FetchHandle, device: torch.device) -> FetchHandle:
    start = time.perf_counter()
    if handle.event is not None:
        torch.cuda.current_stream(device).wait_event(handle.event)
        handle.event.synchronize()
    handle.exposed_wait_s += time.perf_counter() - start
    # CUDA work is measured between producer-stream events.  The main path
    # still uses one consumer barrier rather than per-copy synchronization.
    if handle.event is not None and handle.start_event is not None:
        try:
            handle.cuda_work_s = float(handle.start_event.elapsed_time(handle.event)) / 1000.0
        except Exception:
            handle.cuda_work_s = 0.0
    return handle


__all__ = ["FetchPlan", "FetchHandle", "make_plan", "submit", "wait"]
