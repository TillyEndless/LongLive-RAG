# Unified RAG strategy profile

Instrumentation-only profile; no retrieval IDs, fetch policy, precision, or attention semantics were changed.

| Strategy | Attention calls | Fetch events | Fetch work (s) | Exposed (s) | Hidden (s) | P50 fetch (ms) | Cache hits | Prefetch hits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CURRENT_Q | 6000 | 30270 | 0.916738 | 0.916738 | 0.000000 | 0.029406510293483734 | 0 | 0 |
| PREVIOUS_Q | 6000 | 30270 | 0.859441 | 0.859441 | 0.000000 | 0.027754344046115875 | 0 | 0 |
| FLASH_FETCH | 6000 | 30270 | 1.461386 | 1.461386 | 0.000000 | 0.0461433082818985 | 0 | 0 |
| NEXT_LAYER_PREFETCH | 6000 | 59531 | 0.503420 | 0.503420 | 0.000000 | 0.0012507662177085876 | 15286 | 0 |

## Definitions

FETCH_WORK is summed physical copy timing; FETCH_EXPOSED is the recorded critical-path component; FETCH_HIDDEN is work minus exposed, clipped at zero. ATTENTION_KERNEL remains the existing backend timer and is not recomputed from wrapper/fetch timings.

## Validation

- All four runs completed with 6000 attention calls and 6000 per-call rows.
- CURRENT_Q event count (30270) matches its existing H2D row count (30270); DraftMap structural rows match 5220/5220 against the existing case_01 artifact.
- PREVIOUS_Q, FLASH_FETCH and NEXT_LAYER_PREFETCH use their pre-existing flags; no ID-selection code or attention computation was changed.
- The source contains only existing phase-boundary CUDA synchronizations; the new event recorder adds no per-event global synchronization.

## Strategy notes

- CURRENT_Q and PREVIOUS_Q each recorded 30270 demand fetch events.
- FLASH_FETCH recorded 30270 flash-tile fetch events; its attention-call counter is now recorded at the early-return branch as well.
- NEXT_LAYER_PREFETCH recorded both scheduled prefetch and demand/correction events (59531 total); prefetch copies have no independent CUDA-event elapsed timing in the existing trace, so their work/exposed split is not claimed as a measured overlap result.

## Limitations

Per-event CUDA start/end timestamps were not added because this would require retaining CUDA Event objects or synchronizing per event. `fetch_hidden_ms` is therefore zero for blocking demand/flash copies and not a proof of global overlap; prefetch overlap must be treated as NOT_AVAILABLE until an asynchronous event-pair collector is added. Minimum-useful bytes use unique (layer, history_id) keys; this is a stable lower-bound proxy, not a semantic attention-usefulness measurement. No canonical10 or Group12–19 run was started.
