# Phase 2.18an — Group 11.1 configuration audit

No inference and no CUDA compilation were run.

## Source provenance

- Repository: `/data/zxl/LongLive-RAG-group11_15_h200`
- HEAD: `410948eb7e328513e7751ac2d44193115363cee4`
- Original YAML: `/data/zxl/LongLive-RAG-group11_15_h200/configs/g11_1.yaml`
- Original YAML SHA256: `43ffe72af61608eb7b0b772e9986cae0ed4e5554f6057c4796f5d6d25062d0f9`
- Launcher: `/data/zxl/LongLive-RAG-group11_15_h200/scripts/night_campaign_runner.py`

## A. Complete original `configs/g11_1.yaml`

```yaml
denoising_step_list: [1000, 750, 500, 250]
warp_denoising_step: true
num_frame_per_block: 3
model_name: Wan2.1-T2V-1.3B
group_id: 11
model_kwargs:
  local_attn_size: 12
  timestep_shift: 5.0
  sink_size: 1
  memory_size: 6
  recent_exclude: 5
  use_latentmem: true
  compression_method: avg_pool
  ae_ckpt: /data/zxl/LongLive-RAG-assets/models/LongLive-RAG/checkpoints/ae_latent_mem.pt
  retrieval_backend: draftmap_online
  group_runtime_mode: baseline
  group_sparse_ratio: 0.0
generator_ckpt: /data/zxl/LongLive-RAG-assets/models/LongLive-RAG/checkpoints/longlive_base.pt
lora_ckpt: /data/zxl/LongLive-RAG-assets/models/LongLive-RAG/checkpoints/longlive_lora.pt
adapter: {type: lora, rank: 256, alpha: 256, dropout: 0.0, dtype: bfloat16, verbose: false}
data_path: /data/zxl/strict_latency_case01_20260928/case_01.txt
output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/rag_strategy_profile_runs/group11_1/case_01
inference_iter: -1
skip_existing: false
num_output_frames: 120
use_ema: false
seed: 0
num_samples: 1
save_with_index: true
global_sink: true
context_noise: 0
```

## B. Launcher behavior

`night_campaign_runner.py:11-20` fixes the repository, Python interpreter,
manifest and GPU pool. `prepare()` at lines 50-80 uses
`configs/g11_1.yaml` unchanged for Group 11.1, then `make_cfg()` at lines
39-48 rewrites only:

- `data_path` to the campaign case file;
- `output_folder` to the campaign output directory;
- `skip_existing` to `false`.

The worker command at lines 137-140 is:

```text
CUDA_VISIBLE_DEVICES=<gpu> /data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python \
  inference.py --config_path <generated-config>
```

There is no launcher override for `local_attn_size`, `memory_size`,
`sink_size`, `recent_exclude`, `num_output_frames`, or
`retrieval_backend`. The launcher does not inherit a base YAML. The canonical
YAML uses absolute checkpoint/data paths, so those two paths do not depend on
environment variables. The launcher’s `MANIFEST` is used for campaign case
selection, not as an override inside the original YAML before `make_cfg()`.

## C. Exact parameter semantics

Evidence is from the canonical repository:

- `causal_inference.py:245-259`: `kv_cache_size` is
  `local_attn_size * frame_seq_length`.
- `causal_inference.py:301-328`: the legacy retrieval path computes
  `eligible = num_evicted - recent_exclude`, then selects
  `min(memory_size, eligible)` ranked entries. This path indexes descriptors
  after `sink_size`.
- `wan/modules/causal_model_latentmem.py:167-174`: the attention module sets
  `max_attention_size = local_attn_size * 1560`.
- `wan/modules/causal_model_latentmem.py:604-680`: the canonical
  `draftmap_online` path computes `eligible = len(cpu_k)-recent_exclude`
  and calls `select_topk(..., self.memory_size)`.
- `wan/modules/causal_model_latentmem.py:1428-1440`: committed historical
  CPU K/V is formed from `temp_k[:, sink_tokens:...]`; the sink is not in the
  CPU historical candidate list.
- `wan/modules/causal_model_latentmem.py:1571-1587`: final local budget is
  `max_attention_size - sink_tokens - memory_size * frame_seq_length`.

Answers:

1. `memory_size` counts only ranked historical CPU candidates. It does not
   include the sink.
2. `sink_size` is added independently as the resident sink prefix.
3. The sink cannot be selected by this Top-K path because the archived
   candidate slice begins after `sink_tokens`; therefore there is no sink
   Top-K duplicate to deduplicate.
4. `local_attn_size=1` is not a valid way to express one local/current chunk
   when ranked retrieval is also enabled. It is the complete configured
   attention budget, and the local-budget expression would become negative.
5. `recent_exclude=5` is compatible with the derived configuration, but it
   reduces the eligible pool by five. It must remain unchanged unless the
   user approves a retrieval-policy change.
6. No additional retrieval-count cap was found in the canonical path beyond
   `min(memory_size, eligible)` and the available history length. The physical
   attention budget is the `local_attn_size` expression above.
7. The launcher rewrites data/output/skip only; it does not block the derived
   attention fields. Direct invocation avoids those three launcher rewrites.
8. With 160 latent output frames and `num_frame_per_block=1`, the run can
   accumulate at least 21 archived historical frame units, so after
   `recent_exclude=5` it can reach 16 ranked entries. With the original
   `num_frame_per_block=3`, 160 is rejected by the hard divisibility assertion
   at `causal_inference.py:212-213`.

## D. Smallest derived config

Created without overwriting the original:

`/data/zxl/LongLive-RAG-group11_15_h200/configs/g11_1_w1_r16_s1_f160.yaml`

Derived SHA256: `19eca4c491328c2e391fb9098a876c4d0a07dead7d0ffd8a98841536cb03ccab`

The exact unified diff is:

```diff
--- configs/g11_1.yaml
+++ configs/g11_1_w1_r16_s1_f160.yaml
@@
-num_frame_per_block: 3
+num_frame_per_block: 1
@@
-  local_attn_size: 12
+  local_attn_size: 18
@@
-  memory_size: 6
+  memory_size: 16
@@
-output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/rag_strategy_profile_runs/group11_1/case_01
+output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/phase218an/g11_1_w1_r16_s1_f160/case_01
@@
-num_output_frames: 120
+num_output_frames: 160
```

The `num_frame_per_block` change is strictly required by the existing hard
assertion for 160. The `local_attn_size=18` change is strictly required by the
existing budget equation:

```text
local_budget = (18 - 1 - 16) * 1560 = 1 * 1560 tokens
```

Thus the derived runtime expression at mature history is:

```text
Local/current count       = 1
Ranked DraftMap count     = min(16, len(cpu_history) - 5) = 16
Sink count                = 1
Unique historical count   = 1 sink + 16 ranked = 17
Total logical attention   = 1 sink + 16 ranked + 1 local/current = 18
```

The sink is “included in retrieved 17” as accounting, not as a Top-K item.
`num_output_frames=160` is the model’s latent-frame argument; the decoded MP4
frame count must be measured after an approved run and must not be assumed to
equal 160.

## Gate status

`ORIGINAL_G11_1_READ = PASS`

`MINIMAL_DERIVED_CONFIG_WRITTEN = YES`

`CUDA_COMPILED = NO`

`INFERENCE_STARTED = NO`

The derived YAML is ready for independent review. No baseline/W1 execution was
started.
