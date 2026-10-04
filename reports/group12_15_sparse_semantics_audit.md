# Group14/15 sparse semantics

The corrected campaign uses `group_sparse_ratio` as the retained interaction ratio. For each sparse row, DraftMap routing retains approximately `ceil(ratio * K)` historical blocks; the remainder is not attended. Ratios are 30%, 20%, 10%, and 5%. Final attention is BF16 over dequantized K/V.

The old outputs with CPU-compressed history are incompatible and are not reused.
