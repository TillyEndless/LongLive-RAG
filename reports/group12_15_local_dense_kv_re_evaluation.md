# Group12–15 Local Dense KV Re-evaluation

This is a post-processing evaluation of existing outputs; no inference was run.
GPU KV means only current local dense BF16 K/V. Historical packed KV, CPU history, Draft-K, and transient dequant buffers are excluded.

| Group | Variant | Cases | Local dense KV GiB | Running time |
|---|---|---:|---:|---|
| 12 | group12 | 10 | 3.2135009766 | NOT_AVAILABLE |
| 13 | group13 | 10 | 3.2135009766 | NOT_AVAILABLE |
| 14 | group14_sparse05, group14_sparse10, group14_sparse20, group14_sparse30 | 40 | 3.2135009766 | NOT_AVAILABLE |
| 15 | group15_sparse05, group15_sparse10, group15_sparse20, group15_sparse30 | 40 | 3.2135009766 | NOT_AVAILABLE |

## Runtime note

Existing runtime JSON files contain execution metadata but no scalar end-to-end latency field; therefore `running_time_s` is NOT_AVAILABLE where absent.
No latency was inferred from file timestamps.

Original outputs were not modified.
