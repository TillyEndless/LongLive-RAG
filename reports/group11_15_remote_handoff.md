# Group11–15 H200 audited source handoff

## Git identity

- Remote: `https://github.com/qiz1907/LongLive-RAG.git`
- Branch: `group11-15-h200-audited-sync`
- Local commit: `0e1da21350051b37f3cf74848efbffa21430d1a1`
- Tag: not created
- Push status: pending; GitHub rejected unauthenticated HTTPS push (`could not read Username`).

The commit was created on the H200 host from `/data/zxl/LongLive-RAG-group11_15_h200`.
No inference was launched for this synchronization task.

## Included

- Group12–15 corrected runtime assertions and config matrix.
- Persistent GPU low-bit ownership path and BF16 materialization accounting.
- Group11 runtime instrumentation producers in the H200 authoritative worktree.
- Group12/13 persistent configs and Group14/15 sparse configs for 10 cases.
- Static evaluator/config validation sources staged from the H200 worktree.

## Semantic status

| Mode | Status in this commit |
|---|---|
| Group11.1 current-Q | included in authoritative H200 source |
| Group11.2 previous-Q | separate H200 worktree; not merged into this commit |
| Group11.3 Flash Fetch | separate H200 worktree; not merged into this commit |
| Group11.4 next-layer prefetch | separate H200 worktree; not merged into this commit |
| Group12 | persistent GPU INT8 K / FP8 E4M3 V, BF16 attention |
| Group13 | persistent GPU NVFP4 K/V, BF16 attention |
| Group14 | Group12 storage plus sparse routing |
| Group15 | Group13 storage plus sparse routing |

Because Groups11.2–11.4 remain separate worktrees, this commit must not yet be
described as a complete single-branch reproduction of every Group11 mode.

## Persistent-storage assertions

The corrected Group12–15 path uses:

```text
CPU BF16 authoritative history
 -> BF16 H2D
 -> GPU quantize/pack
 -> persistent GPU low-bit owner
 -> bounded temporary BF16 materialization
 -> BF16 attention
```

The runtime metadata distinguishes persistent packed bytes from transient BF16
materialization. The source path does not intentionally retain a persistent
GPU BF16 shadow after packing.

## Validation

- Python bytecode compilation: passed for the staged runtime modules.
- OmegaConf parsing: passed for 100 corrected Group12–15 configs.
- GPU inference: not run.
- Native kernel validation: not run.

## RTX5090 compatibility

| Group | Classification |
|---|---|
| 11.1 | NEEDS_REBUILD / backend validation |
| 11.2 | NOT_INCLUDED_IN_THIS_COMMIT |
| 11.3 | NOT_INCLUDED_IN_THIS_COMMIT |
| 11.4 | NOT_INCLUDED_IN_THIS_COMMIT |
| 12/14 | NEEDS_REBUILD; fake/portable low-bit path must be verified on SM89 |
| 13/15 | NEEDS_5090_BACKEND or portable FourOverSix validation |

Do not silently substitute fake quantization for a native kernel benchmark.

## Intentionally excluded

Checkpoints, model weights, videos, generated results, traces, tmux logs,
CUDA caches, credentials, SSH keys, and tokens were not staged.
