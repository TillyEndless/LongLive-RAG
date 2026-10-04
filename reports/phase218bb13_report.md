# Phase BB-1.3 — Matched AO/BB Correctness Validation

Worktree: `/data/zxl/LongLive-RAG-group11_1-w1-phase218bb`  
Base source HEAD: `410948eb7e328513e7751ac2d44193115363cee4`  
Generated: `2026-10-04T14:40:05.484929+00:00`

## Minimal source change

The only BB-1.3 production-source change is in
`wan/modules/causal_model_latentmem.py`: after Physical4 rolls, RoPE indices
are constructed as the logical AO W20 tail rather than physical slot indices.
For W20/Physical4/Sink1 with four resident frames this is `[0, 17, 18, 19]`.
`causal_online_rope()` is unchanged. The cache-update and duplicate-override
fixes from BB-1.2 are preserved.

The formula is activated only when physical capacity is smaller than logical
W20; AO Physical20 continues to use `[0, 1, ..., 19]`.

## Authoritative AO input capture

An isolated AO serial_full F30 run was executed with the canonical case,
checkpoint, LoRA and seed. Layer 0 committed update records were captured
before Archive insertion. The capture contains ten unique committed boundaries:

```text
current_end = 4680, 9360, 14040, 18720, 23400,
              28080, 32760, 37440, 42120, 46800
```

This crosses the original W20 logical boundary at `current_end=32760` and
includes later mature states. The raw capture remains on H200 and is not put
in the small review ZIP because it is approximately 1.5 GiB. The compact
metadata/replay result is included instead.

## Actual cache-update replay

The replay invoked the actual AO and BB `CausalWanModel._apply_cache_updates`
methods with the same captured raw K/V and Draft-K tensors. BB update metadata
was derived from the same raw committed stream and its Physical4 geometry;
AO's physical-dependent update_info was not blindly copied into BB.

Results:

```text
records = 10
frame_tokens = 1560
resident raw K/V exact by global frame ID = PASS
BB logical eligible-history count = PASS
single archive owner per implementation = PASS
```

AO and BB naturally have different physical Archive lengths because BB evicts
earlier. After normalization by global frame ID, their resident raw tensors
match the authoritative committed sequence at every replay boundary. This is
the relevant Physical4 invariant; equal Archive list lengths are not expected.

`DraftMap` candidate and selected IDs were not captured in this Layer-0
artifact, so retrieval-content parity is not claimed.

## D2H audit

The real AO F30 eviction path executed the exact expression
`f.to("cpu", non_blocking=True).contiguous().pin_memory()` at four physical
eviction boundaries. A separate audit hook recorded:

```text
current_end=32760  k_pinned=True  v_pinned=True
current_end=37440  k_pinned=True  v_pinned=True
current_end=42120  k_pinned=True  v_pinned=True
current_end=46800  k_pinned=True  v_pinned=True
```

The returned CPU tensors were pinned and had the expected shape
`(1, 1, 1560, 12, 128)`. A bounded CUDA event test also observed the copy
event incomplete before synchronization and correct contents after it. No
unsafe source overwrite was observed in this test. This is a bounded lifetime
finding, not a claim of formal end-to-end latency overlap.

## Real-model serial_full comparison

AO F30 and BB Physical4 serial_full F30 were run sequentially with the same
case, seed, checkpoint and LoRA. Both decoded 117 frames and had zero exactly
black frames, but decoded outputs differed:

```text
pixel_max_abs = 255
pixel_mean_abs = 38.765755115592356
```

This means the strict real-model output gate is not passed. It does not by
itself identify whether the remaining difference is Retrieval, RoPE, final-KV
assembly, or FA2 because tensor-level Q/K/RoPE/final-KV/FA2 capture was not
added to this bounded run. No F3 or F120 was started after this discrepancy.

## Final status

```text
MATCHED_INPUT_CAPTURE = PASS
PHYSICAL_CACHE_4 = PASS
SINGLE_ORIGINAL_ARCHIVE = PASS
ELIGIBILITY_PARITY = PASS
RETRIEVAL_CONTENT_PARITY = NOT_MEASURED
ROPE_PARITY = NOT_MEASURED
FINAL_KV_PARITY = NOT_MEASURED
NATIVE_FA2_PARITY = NOT_MEASURED
D2H_LIFETIME_SAFE = PASS
AO_W1_PIPELINE_UNCHANGED = YES
F3_SMOKE = NOT_RUN
READY_FOR_F120 = NO
REAL_MODEL_F30_OUTPUT_EQUALITY = FAIL
```

The next required action is a narrow tensor-level capture at the first real
divergence, beginning with Current-Q selection and effective RoPE/final-KV.
No cache redesign or synchronization change was made in this phase.
