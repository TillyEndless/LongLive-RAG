# Corrected Group12–Group19 implementation

## Changes applied

1. The shared H200 archive now stores only detached contiguous CPU BF16 K/V.
2. Fetch returns BF16 tensors and rejects non-BF16 archive records.
3. Group12/14 fake INT8 K + FP8 E4M3 V is applied after H2D and dequantized
   before the unchanged BF16 attention call.
4. Group13/15 fake NVFP4 K/V is applied after H2D and dequantized before BF16
   attention.
5. Runtime metadata identifies CPU history dtype, H2D source dtype, GPU
   working-set format, and BF16 final attention.

## Files

- `/data/zxl/LongLive-RAG-group11_15_h200/utils/compressed_history_archive.py`
- `/data/zxl/LongLive-RAG-group11_15_h200/utils/h200_group_runtime.py`

## Not changed

Group11, q_prev, longlive_reuse, checkpoint, dataset, and prior result
directories were not modified. No canonical inference was launched.

## Remaining blockers

Groups16–19 are RTX5090 native Anemoi groups and have no complete H200
implementation in this checkout. They remain blocked rather than silently
mapped to the H200 fake-quant path. Real per-group smoke and memory
measurement for Groups12–15 must be run before corrected canonical results
are reported.
