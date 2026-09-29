"""GPU-resident packed owner for the causal local KV window.

The store exposes only temporary BF16 materialization to attention and never
keeps a dense BF16 shadow between calls.
"""
from __future__ import annotations

from dataclasses import dataclass
import torch

from utils.persistent_kv_storage import PersistentHistoryQuantizer
from utils.quant import quantize_kv
from fouroversix.quantize import QuantizationConfig, QuantizedTensor


@dataclass
class _Slot:
    k: object
    v: object
    tokens: int
    cpu_k: torch.Tensor | None = None
    cpu_v: torch.Tensor | None = None


class LocalLowbitKVStore:
    def __init__(self, max_frames: int, frame_tokens: int, heads: int,
                 head_dim: int, mode: str, device: torch.device,
                 dtype: torch.dtype = torch.bfloat16):
        if mode not in {"int8_fp8", "nvfp4"}:
            raise ValueError(f"unsupported local low-bit mode: {mode}")
        self.max_frames = int(max_frames)
        self.frame_tokens = int(frame_tokens)
        self.heads = int(heads)
        self.head_dim = int(head_dim)
        self.mode = mode
        self.device = torch.device(device)
        self.dtype = dtype
        self.slots: list[_Slot | None] = [None] * self.max_frames
        self._q = PersistentHistoryQuantizer()
        self._fp4_config = QuantizationConfig()

    @property
    def max_tokens(self) -> int:
        return self.max_frames * self.frame_tokens

    def _encode(self, k: torch.Tensor, v: torch.Tensor) -> _Slot:
        expected = (1, self.frame_tokens, self.heads, self.head_dim)
        if tuple(k.shape) != expected or tuple(v.shape) != expected:
            raise ValueError(f"unexpected local KV shape {tuple(k.shape)}")
        if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise ValueError("local KV owner accepts matching BF16 tensors")
        if self.mode == "int8_fp8":
            kq, ks = self._q.quantize_k(k)
            vq, vs = self._q.quantize_v(v)
            return _Slot((kq.contiguous(), ks.contiguous()),
                         (vq.contiguous(), vs.contiguous()), self.frame_tokens,
                         k.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
                         v.detach().to("cpu", dtype=torch.bfloat16).contiguous())
        kq = quantize_kv(k.reshape(-1, self.head_dim), self._fp4_config)
        vq = quantize_kv(v.reshape(-1, self.head_dim), self._fp4_config)
        return _Slot(kq, vq, self.frame_tokens,
                     k.detach().to("cpu", dtype=torch.bfloat16).contiguous(),
                     v.detach().to("cpu", dtype=torch.bfloat16).contiguous())

    def _decode(self, slot: _Slot) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mode == "int8_fp8":
            kq, ks = slot.k
            vq, vs = slot.v
            return self._q.dequantize_k(kq, ks), self._q.dequantize_v(vq, vs)
        k = slot.k.dequantize(dtype=self.dtype).reshape(1, self.frame_tokens, self.heads, self.head_dim)
        v = slot.v.dequantize(dtype=self.dtype).reshape(1, self.frame_tokens, self.heads, self.head_dim)
        return k.contiguous(), v.contiguous()

    def insert_frames(self, start: int, k: torch.Tensor, v: torch.Tensor) -> None:
        if k.shape != v.shape or k.shape[1] % self.frame_tokens:
            raise ValueError("local update must contain whole frames")
        count = k.shape[1] // self.frame_tokens
        for i in range(count):
            pos = int(start) + i
            if not 0 <= pos < self.max_frames:
                raise IndexError(pos)
            self.slots[pos] = self._encode(
                k[:, i * self.frame_tokens:(i + 1) * self.frame_tokens],
                v[:, i * self.frame_tokens:(i + 1) * self.frame_tokens])

    def promote_top(self, scores: list[float], ratio: float) -> list[int]:
        """Return promotion IDs without mutating the low-bit owner."""
        active = [i for i, slot in enumerate(self.slots) if slot is not None]
        count = max(0, min(len(active), int(round(len(active) * float(ratio)))))
        if count == 0:
            return []
        ranked = sorted(active, key=lambda i: (float(scores[i]) if i < len(scores) else 0.0, i), reverse=True)
        return ranked[:count]

    def select_top(self, scores: list[float], ratio: float) -> list[int]:
        """Return local slot IDs without changing persistent ownership.

        This is the v2 promotion policy primitive.  The caller fetches the
        authoritative BF16 copy from the CPU archive for the returned IDs.
        """
        active = [i for i, slot in enumerate(self.slots) if slot is not None]
        count = max(0, min(len(active), int(round(len(active) * float(ratio)))))
        ranked = sorted(active, key=lambda i: (float(scores[i]) if i < len(scores) else 0.0, i), reverse=True)
        return ranked[:count]

    def select_by_query(self, query: torch.Tensor, draft_frames: list[torch.Tensor],
                        ratio: float) -> list[int]:
        """Select local promotion IDs from current Q and local Draft-K."""
        active = [i for i, slot in enumerate(self.slots) if slot is not None]
        count = max(0, min(len(active), int(round(len(active) * float(ratio)))))
        if count == 0 or query.numel() == 0:
            return []
        q = query.float().mean(dim=1)
        scored = []
        for i in active:
            d = draft_frames[i] if i < len(draft_frames) else None
            score = float("-inf") if d is None else float(
                (q * d.to(device=query.device, dtype=torch.float32).mean(dim=1)).mean().item())
            scored.append((score, -i, i))
        scored.sort(reverse=True)
        return [x[2] for x in scored[:count]]

    def roll_and_insert(self, sink_frames: int, evicted_frames: int,
                        rolled_frames: int, write_start_frame: int,
                        new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        old = self.slots
        self.slots = (old[:sink_frames] +
                      old[sink_frames + evicted_frames:sink_frames + evicted_frames + rolled_frames] +
                      [None] * max(0, self.max_frames - sink_frames - rolled_frames))[:self.max_frames]
        self.insert_frames(write_start_frame, new_k, new_v)

    def materialize(self, promoted_ids: set[int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize BF16 working tensors; promotion is call-local only."""
        promoted_ids = set() if promoted_ids is None else {int(x) for x in promoted_ids}
        self.last_materialize_meta = {"promoted_ids": [], "promotion_fetch_bytes": 0,
                                      "dequant_ids": []}
        ks, vs = [], []
        for i, slot in enumerate(self.slots):
            if slot is None:
                break
            if i in promoted_ids:
                if slot.cpu_k is None or slot.cpu_v is None:
                    raise RuntimeError(f"missing CPU BF16 authority for local slot {i}")
                k = slot.cpu_k.to(self.device, dtype=self.dtype, non_blocking=True)
                v = slot.cpu_v.to(self.device, dtype=self.dtype, non_blocking=True)
                self.last_materialize_meta["promoted_ids"].append(i)
                self.last_materialize_meta["promotion_fetch_bytes"] += int(
                    slot.cpu_k.untyped_storage().nbytes() + slot.cpu_v.untyped_storage().nbytes())
            else:
                k, v = self._decode(slot)
                self.last_materialize_meta["dequant_ids"].append(i)
            ks.append(k)
            vs.append(v)
        if not ks:
            shape = (1, 0, self.heads, self.head_dim)
            return (torch.empty(shape, dtype=self.dtype, device=self.device),
                    torch.empty(shape, dtype=self.dtype, device=self.device))
        return torch.cat(ks, dim=1), torch.cat(vs, dim=1)

    def persistent_bytes(self) -> int:
        total = 0
        for slot in self.slots:
            if slot is None:
                continue
            if isinstance(slot.k, tuple):
                values = slot.k + slot.v
            else:
                values = (
                slot.k.values, slot.k.scale_factors, slot.k.amax,
                slot.v.values, slot.v.scale_factors, slot.v.amax)
            for tensor in values:
                if tensor is not None:
                    total += int(tensor.untyped_storage().nbytes())
        return total

    def memory_breakdown(self) -> dict[str, int]:
        out = {"lowbit_k": 0, "lowbit_v": 0, "promoted_k": 0, "promoted_v": 0,
               "scale": 0, "metadata": 0}
        for slot in self.slots:
            if slot is None:
                continue
            values_k = slot.k if isinstance(slot.k, tuple) else (slot.k.values, slot.k.scale_factors, slot.k.amax)
            values_v = slot.v if isinstance(slot.v, tuple) else (slot.v.values, slot.v.scale_factors, slot.v.amax)
            out["lowbit_k"] += int(values_k[0].untyped_storage().nbytes())
            out["lowbit_v"] += int(values_v[0].untyped_storage().nbytes())
            out["scale"] += sum(int(x.untyped_storage().nbytes()) for x in (*values_k[1:], *values_v[1:]) if x is not None)
        return out


__all__ = ["LocalLowbitKVStore"]
