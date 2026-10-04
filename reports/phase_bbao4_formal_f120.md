# BB-AO.4 Formal F120 case_01

## Status

- `ALGORITHM_SOURCE_UNCHANGED = YES` (frozen source hashes verified before run)
- `ORIGINAL_AO_ENVIRONMENT = YES` (`/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python`)
- `F120 = PASS` (`RC=0`, 40 generation blocks, 120 latent frames)
- `ACTUAL_DECODED_FRAMES = 474`
- `BLACK_FRAME_COUNT = 0`
- `NEAR_BLACK_MEAN_LT5_COUNT = 0`
- `W1_RUNTIME_BRANCH = PASS`
- `CORRECTNESS_CERTIFIED = NO`: this is the original exploratory `w1_zero_acquire` path; successful frame integrity does not prove the historical readiness race is eliminated.

## Runtime

| metric | value |
|---|---:|
| E2E | 112.0036315592 s |
| Transformer | 99.1232423251 s |
| Wrapper | 12.8803892341 s |
| Persistent GPU KV | 1,150,156,800 bytes = 1.0711669922 GiB |
| CPU historical BF16 measured | 33,354,547,200 bytes = 31.0638427734 GiB |
| GPU Draft-K persistent | 267,264,000 bytes = 0.248909 GiB |
| Native Attention calls | 6,000 |
| CPU KV fetch calls | 68,400 |
| H2D copy calls | 136,800 |
| Full-KV H2D bytes | 655,589,376,000 |
| DraftMap calls / Top-K calls | 4,650 / 4,650 |

Runtime metadata records one warmup source and 15 remaining sources (16 submitted historical sources), reverse consumption order, no full producer wait before attention, and final buffer length 31,200 tokens. Maximum observed DraftMap selection was 16.

## Video integrity

The output is 474 frames at 16 FPS and 832x480. No exactly-black frames and no frames with mean intensity below 5 were detected. Mean intensity range was 96.920954–120.397121; mean 107.736926. Frames 201–236 all had max value 255 and means 105.173271–117.928383.

## Quality protocol

Original canonical evaluator: `/data/zxl/DraftMap-RAG/scripts/evaluate_canonical3_case.py`, SHA256 `85e642015bb185009a7f68648735ead32dd3c85485a66c8c1de301c9f66e10e5`.

BB-AO Physical4 F120 case_01: DINO `0.8695040345`, SSIM `0.1871621280`, PSNR `9.1454139791`, LPIPS `0.5651229024`, status `OK`.

Historical same-case references from the existing Phase AP artifact (not rerun here):

| method | DINO | SSIM | PSNR | LPIPS | E2E |
|---|---:|---:|---:|---:|---:|
| Original baseline | 0.8445296288 | 0.1743881702 | 9.0320051550 | 0.5627127886 | 119.1131180562 s |
| Historical AO W1 exploratory | 0.8707666993 | 0.1732239530 | 8.9870800380 | 0.5634891391 | 113.3973726407 s |
| Current BB Physical4 W1 | 0.8695040345 | 0.1871621280 | 9.1454139791 | 0.5651229024 | 112.0036315592 s |

The historical AO W1 row is explicitly exploratory/unsynchronized and is not a correctness-certified oracle.

## Reproducibility

Config: `configs/g11_1_bb_ao_physical4_f120.yaml` (SHA256 `3120ee1446c91118a46459eb5bef33377134bd28f21a21a7dda6aec55a3757c4`).

Frozen runtime source hashes:

- `pipeline/causal_inference.py`: `4b17ca6c4b9af0f541e30ef56b80f6eb0ec17c524c0391cb47d7cbee8e3c36db`
- `wan/modules/causal_model_latentmem.py`: `babac411c7209ac490d99ea1e92c14985222fddbf320067d8d5bd242dc388ec9`

Launch used `CUDA_VISIBLE_DEVICES=1 /data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python -u inference.py --config_path configs/g11_1_bb_ao_physical4_f120.yaml`; no other cases were launched.
