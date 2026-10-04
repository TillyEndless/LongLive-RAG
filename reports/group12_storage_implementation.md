# Group12 implementation and smoke audit

- Smoke status: PASS
- Canonical10: NOT RUN
- Full evaluation: NOT RUN
- Smoke workload: 30 output frames / 10 blocks
- CPU archive: authoritative BF16 K/V, pre-pinned
- Persistent GPU owner: INT8 K / FP8_E4M3 V
- Final attention: BF16 stock attention; native low-bit kernel: NO
- Persistent low-bit bytes: 1540969968
- CPU archive bytes: 5175705600
- Transient BF16 peak bytes: 9584640

Q-side sparse selection: disabled.
KV-side promotion: disabled.
All results above are from the minimal smoke only.
