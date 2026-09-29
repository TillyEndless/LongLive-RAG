# Group11–15 Local KV Promotion Parity

This change freezes the confirmed requirement that every Group11–15 path
contains the same promotion mechanism. It does not run canonical inference.

## Policy

The v2 worktree uses `local_kv_promotion_ratio=0.2` for the formal Group12–15
configs and the Group11 reference config.

| Group | Local owner | Promotion behavior |
|---|---|---|
| 11 | BF16 | identity promotion: same selection/bookkeeping, no dtype conversion |
| 12 | INT8 K / FP8 V | selected local chunks use transient CPU-BF16 overlay |
| 13 | NVFP4 K/V | selected local chunks use transient CPU-BF16 overlay |
| 14 | INT8 K / FP8 V | same promotion policy after sparse retained-set construction |
| 15 | NVFP4 K/V | same promotion policy after sparse retained-set construction |

Group11 cannot show a numerical precision-promotion effect because its local
owner is already BF16. The implementation nevertheless computes and records
the same policy-selected IDs; it is an identity operation.

## Pairwise fairness

The formal configs now satisfy:

```text
Group12 promotion policy == Group14 promotion policy
Group13 promotion policy == Group15 promotion policy
```

Promotion remains transient for low-bit groups. It does not replace the
persistent low-bit owner. The Group11–15 history fetch/promotion path remains
the Group11-compatible CPU-BF16 archive → temporary GPU working-set path.

## Important remaining scope note

The separate sparse-scope correction is not silently claimed here. The current
v2 attention path still applies Q-sparse routing to the retrieved-history
working set before appending the dense local window. A future semantic change is
required if the final frozen definition is sparse Top-K over the complete
`local window + retrieved history` candidate universe. No canonical quality
result should be interpreted as validating that broader sparse definition until
that change and its tests are complete.

