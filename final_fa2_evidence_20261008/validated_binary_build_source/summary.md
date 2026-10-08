# Grouped Ready Frontier Validation

## Scope
H200 temporary platform, B01 geometry:
- Q=4680, KV=28080, H=12, D=128, BF16
- 8 historical frames streamed from mapped pinned CPU memory
- 8 * 1560 = 12480 historical tokens
- 76,677,120 bytes historical K+V
- 196 streamed BlockN=64 tiles
- stock FA2 math unchanged

## Design
Resident-region bypass + contiguous ready frontier.
Consumer checks frontier only once per group of G tiles. Producer publishes per-tile readiness after K+V writes and advances frontier only across a contiguous ready suffix.

## All-ready overhead sweep
| G tiles | Tokens/group | Gated FA2 median | Overhead vs stock | Exact |
|---:|---:|---:|---:|---|
| 1 | 64 | 2.9892 ms | +8.72% | yes |
| 2 | 128 | 2.9689 ms | +7.98% | yes |
| 4 | 256 | 2.9444 ms | +7.09% | yes |
| 8 | 512 | 2.9349 ms | +6.74% | yes |
| 16 | 1024 | 2.9281 ms | +6.49% | yes |

Stock median: 2.7495 ms.

## Dynamic 8-frame group sweep, 16 producer CTAs
| G | Tokens/group | Joint median | Reduction vs serial | Exact |
|---:|---:|---:|---:|---|
| 1 | 64 | 3.1069 ms | 26.37% | yes |
| 2 | 128 | 3.0996 ms | 26.54% | yes |
| 4 | 256 | 3.0905 ms | 26.76% | yes |
| 8 | 512 | 3.0982 ms | 26.58% | yes |
| 16 | 1024 | 3.1184 ms | 26.10% | yes |
| 32 | 2048 | 3.1715 ms | 24.84% | yes |

Dynamic optimum: G=4.

## G=4 producer CTA sweep
| Producer CTA | Serial total median | Streaming joint median | Reduction | Speedup | Exact |
|---:|---:|---:|---:|---:|---|
| 8 | 4.3741 ms | 3.2603 ms | 25.46% | 1.342x | yes |
| 16 | 4.2129 ms | 3.0916 ms | 26.62% | 1.363x | yes |
| 24 | 4.2117 ms | 3.1000 ms | 26.39% | 1.359x | yes |
| 32 | 4.2111 ms | 3.1239 ms | 25.82% | 1.348x | yes |

Best validated configuration: G=4, 16 producer CTAs.

## Baseline comparison
- Full-range per-tile gate streaming: ~3.1372 ms/layer
- Resident-region bypass per-tile gate: ~3.1101 ms/layer
- Grouped frontier G=4, 16 CTA: 3.0916 ms/layer
- Serial fetch then stock FA2: 4.2129 ms/layer

## Interpretation
Grouped frontier is functionally correct and improves over resident-bypass. Pure all-ready gate overhead drops from +8.61% (resident bypass) to +7.09% at G=4. Larger G further reduces pure gate overhead, but dynamic latency worsens beyond G=4 because the consumer must wait for an entire larger group to become contiguous-ready, creating bubbles.

Therefore G=4 is the current H200 latency optimum for the 8-frame B01 microbenchmark.

All reported dynamic results are bitwise exact versus stock FA2 with max_abs=0.
