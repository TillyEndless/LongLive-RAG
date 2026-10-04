# Phase BB-1.1 Integration Repair Report

## Scope

The existing BB worktree was retained:

`/data/zxl/LongLive-RAG-group11_1-w1-phase218bb/`

AO, BA, AY, AZ-W and production sources were not modified. No F120, profiling,
or Unified Evaluation was run.

## Repairs applied

### Active eligibility

`_online_memory_indices()` now uses `_logical_eligible_history_count()` and
clamps the logical committed prefix against actual CPU K/V and Draft-K lengths
before applying `recent_exclude` once. The same helper is also used by the
previous-Q candidate path. DraftMap scoring, Top-K, IDs, archive structures and
fetch ordering are unchanged.

### Direct AO final-buffer use

For `group11_fetch_mode: w1_zero_acquire`, the caller now sets:

```python
k_cat = w1_final_k
v_cat = w1_final_v
```

and sends that buffer through the existing `prepare_attention_kv()` and stock
FA2 call. The ordinary `k_parts/v_parts -> torch.cat()` path is under the
non-W1 branch and is not executed for W1. This removes the previous redundant
repack of the complete final buffer.

### AO post-attention join

Immediately after the stock attention call, BB now waits on
`_w1_pending_fetch_event`, clears the event and stream references, and releases
`_w1_fetch_keepalive`, matching the original AO lifecycle point. The wait is
not moved before FA2. The AO W1 exploratory race caveat remains unchanged.

## Verification

```text
PY_COMPILE = PASS
PHYSICAL4_ALLOCATION = PASS_STATIC
ACTIVE_CURRENT_Q_ELIGIBILITY = STATIC_PASS
AO_W1_HELPER_IDENTICAL = YES (normalized body)
AO_W1_DIRECT_FINAL_BUFFER_PATH = STATIC_PASS
AO_POST_ATTENTION_JOIN = STATIC_PASS
GATE_B_COMMITTED_STATE_REPLAY = NOT_MEASURED
GATE_C_ALL_READY_ATTENTION_PARITY = NOT_MEASURED
D2H_LIFETIME_SAFE = UNRESOLVED
READY_FOR_F120 = NO
```

The static checks prove call ordering and source structure only. They do not
prove global frame identity, D2H readiness, effective RoPE parity, or Native
FA2 output parity. No matched AO/BB state capture was available in this repair
turn, so execution stops before any smoke or F120.

## Physical allocation

```text
30 × 4 × 1560 × 12 × 128 × 2 × 2
= 1,150,156,800 bytes = 1.071167 GiB
```

## Remaining required gate

The next authorized action is a small matched committed-state replay covering
first physical eviction, original W20 logical eviction and recompute, followed
by serial_full All-ready final-K/V and Native FA2 parity. If the first global ID
or tensor diverges, stop at that location. F120 remains blocked.
