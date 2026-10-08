# Group11.1–11.4 static/configuration audit

Audit date: 2026-09-28

## Executive conclusion

The four rows do **not** all have the same validity status.

| Row | Mechanism | Config/code | Canonical inference | Canonical evaluation | Conclusion |
|---|---|---|---:|---:|---|
| Group11.1 | original current-layer Q DraftMap retrieval | PASS | 10/10 | 10/10 | valid reference |
| Group11.2 | previous-Q retrieval variant | PASS for its intended ablation | incomplete; one process currently running | 0/10 verified | not a complete comparison row |
| Group11.3 | serial current-layer Flash Fetch smoke path | PASS for smoke isolation | 0/10 canonical | 0/10 | smoke only; async overlap not proven |
| Group11.4 | next-layer prefetch/correction variant | PASS for smoke isolation | 0/10 canonical | 0/10 | smoke only; exactness gates pending |

## Group11.1

The canonical reference uses `local_attn_size=12`, `memory_size=6`, `recent_exclude=5`, `retrieval_backend=draftmap_online`, seed 0, the canonical 10-case manifest, and BF16 attention. Its generated videos use the frozen 474-frame evaluation protocol. The config field `num_output_frames=120` is a latent-frame count; it decodes to the 474-frame video protocol and is not a mismatch.

The evaluator uses frame 0 vs frame 237, DINOv2-small CLS cosine, raw RGB SSIM and PSNR, with LPIPS unavailable. The 10-case result in `results/group11_final.csv` is the valid reference.

## Group11.2

The active process is in `/data/zxl/LongLive-RAG-group11_qprev_h200` and is a previous-Q retrieval experiment. It is not the same current-Q mechanism as Group11.1. Its existing alignment evidence records different selected sets; therefore Group11.1 quality cannot be inherited. It requires independent canonical evaluation after inference completes.

## Group11.3

The Flash Fetch implementation is isolated from Group11.1 and reports serial blocking fetch behavior. The smoke metadata reports `async_overlap=false`, so it does not establish asynchronous copy/compute overlap. Its smoke videos are not canonical 474-frame inputs and must not enter the Group11.1 evaluator.

## Group11.4

The next-layer prefetch implementation is a separate speculative path. Smoke traces include hit/correction/waste counters, but they do not establish selected-ID exactness or final-latent equivalence. It must not inherit Group11.1 quality or latency.

## Evaluation pipeline finding

The frozen Group11.1 evaluator semantics are correct for Group11.1. The existing `evaluate_all_canonical_aligned.py` is not a universal evaluator for 11.2–11.4: it hard-codes Group12–15 paths and does not evaluate the Group11 variants. Therefore 11.2–11.4 are not evaluation complete merely because the Group11.1 table exists.

## Required actions

1. Leave the active Group11.2 process untouched; finish or record it as its own previous-Q experiment.
2. Run independent canonical10 inference/evaluation for Group11.2.
3. For Group11.3, prove selected-ID equality, online-softmax correctness and actual overlap before reporting a quality/latency row.
4. For Group11.4, prove current-layer working-set correctness and final-latent equivalence before reporting a quality/latency row.
5. Use separate provenance and evaluator outputs for each variant.

## Final fields

```text
GROUP11_1_CONFIG = PASS
GROUP11_1_PIPELINE = PASS
GROUP11_1_CANONICAL = 10/10 inference + 10/10 evaluation
GROUP11_2_CONFIG = PASS_FOR_PREVIOUS_Q_VARIANT
GROUP11_2_CANONICAL = INCOMPLETE
GROUP11_3_CONFIG = PASS_FOR_SMOKE_ONLY
GROUP11_3_CANONICAL = NOT_AVAILABLE
GROUP11_4_CONFIG = PASS_FOR_SMOKE_ONLY
GROUP11_4_CANONICAL = NOT_AVAILABLE
GROUP11_1_11_4_ALL_VALID = NO
```
