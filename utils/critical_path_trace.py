from __future__ import annotations
import os, time
import torch

class CriticalPathTrace:
    """Optional one-final-sync phase collector; disabled by default."""
    def __init__(self, enabled=False):
        self.enabled = bool(enabled)
        self.reset()
    def reset(self):
        self.events=[]
    def measure(self, label, fn, metadata=None):
        if not self.enabled:
            return fn()
        hs=time.perf_counter_ns()
        start=end=None
        if torch.cuda.is_available():
            start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream())
        result=fn()
        if end is not None: end.record(torch.cuda.current_stream())
        he=time.perf_counter_ns()
        self.events.append({'label':label,'host_ms':(he-hs)/1e6,'cuda_start':start,'cuda_end':end,'metadata':metadata or {}})
        return result
    def finalize(self):
        if not self.enabled: return {'enabled':False,'records':[],'synchronization':'none'}
        if any(e['cuda_end'] is not None for e in self.events): torch.cuda.current_stream().synchronize()
        rows=[]
        for e in self.events:
            row=dict(e['metadata']); row.update({'label':e['label'],'host_ms':e['host_ms']})
            row['cuda_ms']=float(e['cuda_start'].elapsed_time(e['cuda_end'])) if e['cuda_end'] is not None else 'NOT_AVAILABLE'
            rows.append(row)
        return {'enabled':True,'synchronization':'one current-stream synchronize at finalize','records':rows}

ACTIVE_TRACE = None
def set_active_trace(trace):
    global ACTIVE_TRACE; ACTIVE_TRACE=trace
