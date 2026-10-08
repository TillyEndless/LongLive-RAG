# Corrected Group14/15 GPU memory value provenance trace

## Scope

本次只审计已有源码和 artifacts；未运行 inference、未 benchmark、未修改源码或已有结果。

## Corrected interpretation

目标表的语义应为：

`GPU actual persistent KV = GPU local window KV + GPU history KV`

`GPU_DRAFT_PERSISTENT_BYTES` 不加入第三列；它是独立的 Draft-K persistent memory。以 Group14.1 为例：`3.2135 + 3.7889 = 7.0024 GiB`。

## Source code evidence

| Field | Source | Meaning | Draft included | Transient included |
|---|---|---|---:|---:|
| `GPU_KV_ACTUAL_PERSISTENT_BYTES` | `utils/runtime_memory_measurement.py:239-246` | measured local KV + resident packed K/V + scales/metadata | No | No |
| `GPU_KV_MEASURED_BYTES` / `GPU_LOCAL_BF16_KV_BYTES` | `utils/runtime_memory_measurement.py:170-175` | current local dense K/V | No | No |
| `GPU_DRAFT_PERSISTENT_BYTES` | `utils/runtime_memory_measurement.py:198-203,239-242` | `gpu_draft_k_frames` only | Separate | No |

源码示例中 local W12 为 `3.2135009765625 GiB`，Draft-K 为 `0.23174285888671875 GiB`。源码的 actual-persistent 公式明确没有加 Draft-K。

## Requested table reconstruction

| Group | GPU local window KV | GPU history KV | GPU actual persistent KV | Residual | Provenance |
|---:|---:|---:|---:|---:|---|
| 9 | 1.6068 GiB | 0.0000 GiB | 1.6068 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 10 | 3.2135 GiB | 0.0000 GiB | 3.2135 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 4.1 | 1.6068 GiB | 0.0000 GiB | 1.6068 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 4.2 | 3.2135 GiB | 0.0000 GiB | 3.2135 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 11.1 | 3.2135 GiB | 0.0000 GiB | 3.2135 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 11.2 | 3.2135 GiB | 0.0000 GiB | 3.2135 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 11.3 | 3.2135 GiB | 0.0000 GiB | 3.2135 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 11.4 | 3.2135 GiB | 0.0000 GiB | 3.2135 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 12 | 3.2135 GiB | 4.5517 GiB | 7.7652 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 13 | 3.2135 GiB | 2.4783 GiB | 5.6918 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 14.1 | 3.2135 GiB | 3.7889 GiB | 7.0024 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 14.2 | 3.2135 GiB | 2.6303 GiB | 5.8438 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 14.3 | 3.2135 GiB | 1.9970 GiB | 5.2105 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 14.4 | 3.2135 GiB | 3.2270 GiB | 6.4405 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 15.1 | 3.2135 GiB | 2.1532 GiB | 5.3667 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 15.2 | 3.2135 GiB | 1.5181 GiB | 4.7316 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 15.3 | 3.2135 GiB | 1.0684 GiB | 4.2819 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |
| 15.4 | 3.2135 GiB | 1.7853 GiB | 4.9988 GiB | 0.0000 GiB | arithmetic confirmed; raw producer not found |

## Search result

已按这组三列的精确数值搜索 `results/` 和 `reports/` 中的 JSON/CSV/Markdown；没有找到包含这些完整 memory rows 的 artifact 或生成脚本。部分相同小数可能出现在 profiling timing 字段中，不能据此认定为 memory。

因此：

- `3.2135 GiB` 的 local window KV 有源码和 raw memory artifact 支持；
- 第三列的定义与源码 `GPU_KV_ACTUAL_PERSISTENT_BYTES` 语义一致，即不含 Draft-K；
- historical KV 数值和这张汇总表的直接生产来源仍未定位；
- `0.2317 GiB` 应单独报告为 Draft GPU memory，不应加到第三列。

## Final field names

- `GPU_LOCAL_WINDOW_KV_GIB`
- `GPU_HISTORY_KV_GIB`
- `GPU_KV_ACTUAL_PERSISTENT_GIB`
- `GPU_DRAFT_PERSISTENT_GIB`
- `TRANSIENT_BF16_DEQUANT_PEAK_GIB`
