# Phase BB-1 Minimal Physical4 Status

## Scope

An isolated copy was created at:

`/data/zxl/LongLive-RAG-group11_1-w1-phase218bb/`

Source of truth was the original Group 11.1 tree. The AO and BA trees were not
modified. No CUDA compilation, model loading, inference, F120, or Unified
Evaluation was run.

## Implemented changes

Only these experimental changes were made:

1. `pipeline/causal_inference.py`
   - Filter and read `physical_kv_cache_frames`.
   - Override only the physical K/V allocation size with
     `physical_kv_cache_frames * frame_seq_length`.
   - Keep `local_attn_size=20` and the logical attention configuration intact.
2. `wan/modules/causal_model_latentmem.py`
   - Added the exact AO `_w1_progressive_final_buffer()` implementation.
   - Added a `w1_zero_acquire` caller branch around the existing retrieval
     materialization.
   - Added a logical eligibility count updated after committed cache updates;
     DraftMap candidate construction clamps to this count before applying the
     existing `recent_exclude` exactly once.
3. `configs/g11_1_bb_physical4.yaml`
   - Original logical settings preserved: W20/R16/S1, F120, seed 0.
   - Added `physical_kv_cache_frames: 4` and an isolated output directory.

No BA Pending/Promotion/Event framework, new cache, FA2 change, W1 scheduling
change, or RoPE function change was added.

## Gate results

```text
GATE_A_PHYSICAL_ALLOCATION             = PASS_STATIC
GATE_B_COMMITTED_STATE_REPLAY          = NOT_MEASURED
GATE_C_ALL_READY_ATTENTION_PARITY      = NOT_MEASURED
GATE_D_AO_W1_SOURCE_WIRING             = PASS_STATIC
READY_FOR_F3                            = NO
READY_FOR_F120                          = NO
UNIFIED_EVALUATION                      = NOT_RUN
```

Gate A calculation for the 30-layer Wan 1.3B geometry:

```text
30 layers × 4 frames × 1560 tokens/frame × 12 KV heads × 128 D
× 2 tensors (K,V) × 2 bytes (BF16)
= 1,150,156,800 bytes = 1.071167 GiB
```

The corresponding W20 allocation is 5,750,784,000 bytes. The exact AO W1
method was copied mechanically; normalized source-body comparison is identical.
The method remains explicitly `W1_STREAMING_UNSYNCHRONIZED_EXPLORATORY`.

## Why execution stops here

The logical eligibility rule is new behavior and must be validated against
identical committed states before any inference. A static formula check cannot
prove that early physical eviction preserves global frame identity, archive
ordering, D2H readiness, Draft-K association, or recompute behavior. Likewise,
no All-ready real Q/K/V state was available in this isolated BB run to prove
final RoPE/KV/Native-FA2 parity.

Therefore this result does not claim Physical4 correctness and does not justify
F120. The next permitted step is a focused deterministic or matched-state
replay for Gates B/C; if the first logical ID or tensor diverges, stop there.

## Provenance

The complete source/configuration and generated diffs are in the review bundle.
The copied source tree remains isolated; original AO, BA and production files
are untouched.
