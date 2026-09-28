# Group11–15 shared-branch handoff

- Remote: git@github.com:TillyEndless/LongLive-RAG.git
- Branch: group11-15-h200-audited-sync
- Base commit preserved: ad4f0728d48dfe9c318cb7d98cae6625af2a5b6a
- Current pushed SHA: 3340f40
- Tag: none

## Current mode status

| Mode | Status |
|---|---|
| Group11.1 current-Q | included in base |
| Group11.2 previous-Q direct retrieval | included, static-checked |
| Group11.3 serial Flash Fetch | included, static-checked; async overlap not implemented |
| Group11.4 next-layer prefetch | not yet integrated |
| Group12/13 | included, persistent low-bit owner + BF16 compute |
| Group14/15 | included, corrected sparse + persistent low-bit semantics |

Group11.4 was not force-merged because its older cache lifecycle conflicts with
the corrected Group12–15 cache ownership and accounting path. No inference was
run during this integration.

## Validation

- Python syntax checks: passed for the integrated Group11.2/11.3 and Group12–15 files.
- Group12–15 config parsing: 100 configs passed previously.
- Group11.3 async status: SERIAL_ONLY.
- Group11.4 exactness status: NOT_YET_VALIDATED.
- Runtime benchmark: not run.
