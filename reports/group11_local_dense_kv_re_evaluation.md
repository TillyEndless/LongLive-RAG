# Group11 Local Dense KV Re-evaluation

This re-evaluation reports only the current local dense BF16 K/V cache.
Historical CPU KV, historical packed KV, and Draft-K memory are excluded.

| Group | Cases with measured local dense KV | Local dense KV GiB | Source field |
|---|---:|---:|---|
| 11.1 | 1 | 3.2135009766 | GPU_KV_MEASURED_BYTES (legacy local dense) |
| 11.2 | 1 | 3.2135009766 | GPU_LOCAL_BF16_KV_BYTES |
| 11.3 | 0 | NOT_AVAILABLE | NOT_AVAILABLE |
| 11.4 | 0 | NOT_AVAILABLE | NOT_AVAILABLE |

## Field definition

`GPU_LOCAL_DENSE_KV_GiB` = BF16 K/V tensors belonging to the current local attention window.
It is not the historical packed KV owner and does not include CPU historical KV.

Original memory JSON files were not modified.
