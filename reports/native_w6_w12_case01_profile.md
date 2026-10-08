# Native W6/W12 matched case_01 runtime profile

This report adds profiling-only latency and memory measurements for the existing Native W6/W12 baselines. No canonical10 inference or quality evaluation was rerun.

## Provenance

- Quality: existing canonical10 results.
- Latency and memory: new matched `case_01` profiling runs, 120 latent output frames / 474 decoded frames, 16 FPS, 832x480.
- Semantics: native local sliding window; `use_latentmem=false`, `memory_size=0`; no retrieval, DraftMap, low-bit, or sparse path.
- Code commit: `f519727eb2a9e6d9b748220ce1f6bb1144406446`.

## Results

| Window | E2E (s) | Transformer (s) | Self-attention wrapper (s) | Attention kernel (s) | GPU KV (GiB) | CPU KV (GiB) | Calls | Quality DINO | SSIM | PSNR |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 40.43552218191326 | 27.766428761184216 | 14.990464705973864 | 6.1854243491292 | 1.60675048828125 | 0.0 | 6000 | 0.7487808406352997 | 0.32644455234919245 | 9.815110998214726 |
| 12 | 43.00493859965354 | 30.181652868166566 | 17.479824762791395 | 10.6594258223176 | 3.2135009765625 | 0.0 | 6000 | 0.7914137989282608 | 0.33470661431094184 | 10.029504634828326 |

## Validation and limitations

- Both windows recorded 6000 attention calls and 6000 attention-kernel event pairs; validity is `PASS`.
- Native fetch fields are zero because these runs do not use CPU history retrieval.
- Peak allocated/reserved GPU memory was not emitted by the current memory artifact and is therefore `NOT_AVAILABLE`; it was not inferred from `nvidia-smi`.
- QKV, cross-attention, FFN, norm/residual, VAE, and video-encode components were not instrumented in this non-invasive native probe and remain `NOT_AVAILABLE`.
- The latency values are matched-case measurements, not ten-case means. They are directly comparable to other runs only where timer boundaries and workload are identical.
