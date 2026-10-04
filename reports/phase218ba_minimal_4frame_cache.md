# Phase BA-1 — Minimal Config-Driven 4-Frame Cache

## Scope

This BA worktree starts from the restored AO implementation at `410948eb7e328513e7751ac2d44193115363cee4`. AO/AY/production trees were not modified. The implementation deliberately keeps the logical configuration at W20 and adds only a physical cache-capacity setting.

Target configuration:

```yaml
local_attn_size: 20
memory_size: 16
sink_size: 1
recent_exclude: 5
physical_kv_cache_frames: 4
```

The intended logical set is Sink 1 + Retrieved 16 + Local 3 = 20. The persistent GPU tensors contain only Sink + current Local 3 = 4 frames.

## Minimal source change

Modified files relative to the AO source snapshot:

1. `pipeline/causal_inference.py`
   - Removes `physical_kv_cache_frames` before constructing `WanDiffusionWrapper`.
   - Validates the field and stores it separately from `local_attn_size`.
   - Uses it only for physical KV allocation size.
   - Adds pending CPU frame lists and the logical cache capacity to each cache record.

2. `wan/modules/causal_model_latentmem.py`
   - Keeps physically early-evicted frames in CPU pending lists.
   - Promotes them to the AO retrieval-visible archive only at the original logical W20 boundary.
   - Uses the logical frame positions for the rolling-branch query/K RoPE calculation. Mature local positions are 17, 18, 19, not 1, 2, 3.

No FA2, retrieval scoring, fetch order, Native Attention call, or AY pending-history architecture was added.

## Why YAML-only is insufficient

AO uses `local_attn_size` for both the logical attention budget and the initial GPU tensor allocation. Changing it from 20 to 4 would also change logical max-attention/RoPE semantics and would not express W20/R16/S1. The new field separates only the physical allocation while leaving `local_attn_size=20` intact.

## Validation

### Physical allocation

Measured by calling the actual BA `_initialize_kv_cache` method on H200 with 30 transformer cache entries and BF16 K/V:

| Configuration | K bytes | V bytes | Total persistent K/V | Shape per tensor |
|---|---:|---:|---:|---|
| AO W20 | 2,875,392,000 | 2,875,392,000 | 5,750,784,000 | `[1, 31200, 12, 128]` |
| BA physical4 | 575,078,400 | 575,078,400 | 1,150,156,800 | `[1, 6240, 12, 128]` |

The result is across all 30 layers; no hidden W20 persistent tensor was allocated by this gate.

### Deterministic state replay

The actual BA `_apply_cache_updates` implementation was exercised with 16 three-frame updates and frame-sentinel K/V. Both capacities produced archive IDs 1–28 and the same retrieval-visible candidate set and selected IDs after `recent_exclude=5`. BA additionally retained 16 earlier physically evicted frames in pending CPU staging, which is not retrieval-visible yet. Final physical BA state was Sink + frames 46–48 and 6,240 tokens.

Status: `RETRIEVAL_EQUIVALENCE = PASS` for the tested replay.

### RoPE gate

The actual `causal_online_rope` implementation and Wan frequency construction were exercised on the same raw tensors. The BA physical view `[0,17,18,19]` matched the corresponding W20 logical positions exactly:

```text
max_abs = 0.0
mean_abs = 0.0
```

Status: `ROPE_PARITY = PASS` for this implementation-level gate.

### Attention parity

No matching all-ready AO-vs-BA Q/K/V capture or final Native FA2 output artifact was present in the BA/AO worktrees. No mismatched video or unsynchronized AO W1 output was substituted. Therefore strict attention parity is `NOT_MEASURED`, and no full model inference was started.

## Current status

```text
MINIMAL_PATCH = YES
PHYSICAL_CACHE_4 = PASS
RETRIEVAL_EQUIVALENCE = PASS
ROPE_PARITY = PASS
ATTENTION_PARITY = NOT_MEASURED
SHORT_RUN_LATENCY = NOT_MEASURED
READY_FOR_F120 = NO
```

The next required gate is a matched all-ready state capture, comparing raw Q/K/V, selected IDs/content, RoPE-transformed operands, final K/V and Native FA2 output. Until that gate is available and passes, no short real-model run or F120 campaign should start.
