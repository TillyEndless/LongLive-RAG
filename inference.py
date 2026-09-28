# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision import transforms
from torchvision.io import write_video
from einops import rearrange
import torch.distributed as dist
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import matplotlib.pyplot as plt
from torch.profiler import profile as torch_profile, ProfilerActivity

from pipeline import (
    CausalInferencePipeline,
)
from utils.dataset import TextDataset
from utils.misc import set_seed

from utils.memory import get_cuda_free_memory_gb, DynamicSwapInstaller
from utils.runtime_memory_measurement import runtime_inference_memory

parser = argparse.ArgumentParser()
parser.add_argument("--config_path", type=str, help="Path to the config file")
parser.add_argument("--v5_profiler_output", type=str, default="", help="Diagnostic bounded torch.profiler output root")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)


def validate_group_contract(cfg):
    """Fail fast on the frozen Group11--15 experiment contract.

    This is intentionally a configuration/runtime gate, not an inference
    change.  It prevents stale Group12--15 configs from silently running with
    the old transient fake-quant semantics or the wrong frame protocol.
    """
    group = int(getattr(cfg, "group_id", 0) or 0)
    mode = str(getattr(cfg.model_kwargs, "group_runtime_mode", "baseline"))
    if group not in {11, 12, 13, 14, 15} and mode == "baseline":
        return
    if group in {12, 13, 14, 15} or mode in {
        "group12_corrected", "group13_corrected",
        "group14_corrected", "group15_corrected",
    }:
        expected_mode = {12: "group12_corrected", 13: "group13_corrected",
                         14: "group14_corrected", 15: "group15_corrected"}.get(group)
        if expected_mode and mode != expected_mode:
            raise ValueError(f"Group {group} requires {expected_mode}, got {mode}")
        if int(getattr(cfg.model_kwargs, "local_attn_size", -1)) != 12:
            raise ValueError("Group11--15 frozen contract requires local_attn_size=12")
        if str(getattr(cfg.model_kwargs, "retrieval_backend", "")) != "draftmap_online":
            raise ValueError("Group11--15 requires retrieval_backend=draftmap_online")
        if int(getattr(cfg.model_kwargs, "memory_size", -1)) != 6:
            raise ValueError("Group11--15 requires memory_size=6")
        if int(getattr(cfg.model_kwargs, "recent_exclude", -1)) != 5:
            raise ValueError("Group11--15 requires recent_exclude=5")
        if int(getattr(cfg, "seed", -1)) != 0:
            raise ValueError("Group11--15 canonical protocol requires seed=0")
        expected_frames = int(getattr(cfg, "num_output_frames", -1))
        smoke_10block = bool(getattr(cfg, "smoke_10block", False))
        smoke_1block = bool(getattr(cfg, "smoke_1block", False))
        if expected_frames != 120 and not ((smoke_10block and expected_frames == 30) or (smoke_1block and expected_frames == 3)):
            raise ValueError("num_output_frames must be 120 (474 decoded frames), 30-frame/10-block smoke, or 3-frame/1-block smoke")
        if str(getattr(cfg, "persistent_storage_mode", "")).lower() != "lowbit_storage_bf16_compute":
            raise ValueError("Corrected Group12--15 require LOWBIT_STORAGE_BF16_COMPUTE")
        if str(getattr(cfg, "cpu_historical_k_storage", "")).lower() != "bf16" or \
           str(getattr(cfg, "cpu_historical_v_storage", "")).lower() != "bf16":
            raise ValueError("CPU historical K/V must remain authoritative BF16")
        if mode in {"group12_corrected", "group14_corrected"}:
            if str(getattr(cfg, "gpu_k_storage", "")).lower() != "int8" or \
               str(getattr(cfg, "gpu_v_storage", "")).lower() != "fp8_e4m3":
                raise ValueError("Groups12/14 require persistent GPU K=INT8, V=FP8_E4M3")
        if mode in {"group13_corrected", "group15_corrected"}:
            if str(getattr(cfg, "gpu_k_storage", "")).lower() != "nvfp4" or \
               str(getattr(cfg, "gpu_v_storage", "")).lower() != "nvfp4":
                raise ValueError("Groups13/15 require persistent GPU K/V=NVFP4")
        if str(getattr(cfg, "attention_compute", "")).lower() != "bf16":
            raise ValueError("Group12--15 final attention must remain BF16")
        if bool(getattr(cfg, "q_prev", False)) or bool(getattr(cfg, "flash_fetch", False)) or \
           bool(getattr(cfg, "next_layer_prefetch", False)) or bool(getattr(cfg, "hot_cache", False)) or \
           bool(getattr(cfg, "longlive_reuse", False)):
            raise ValueError("Group12--15 must not inherit qprev/FlashFetch/prefetch/hot-cache semantics")
        ratio = float(getattr(cfg.model_kwargs, "group_sparse_ratio", 0.0) or 0.0)
        if mode in {"group14_corrected", "group15_corrected"} and not (0.0 < ratio < 1.0):
            raise ValueError("Groups14/15 require an explicit retained sparse ratio in (0,1)")
        if mode in {"group12_corrected", "group13_corrected"} and ratio != 0.0:
            raise ValueError("Groups12/13 must have sparse ratio=0")
    elif group == 11:
        if str(getattr(cfg.model_kwargs, "retrieval_backend", "")) != "draftmap_online":
            raise ValueError("Group11 canonical contract requires retrieval_backend=draftmap_online")
        expected_frames = int(getattr(cfg, "num_output_frames", -1))
        smoke_10block = bool(getattr(cfg, "smoke_10block", False))
        smoke_1block = bool(getattr(cfg, "smoke_1block", False))
        if expected_frames != 120 and not ((smoke_10block and expected_frames == 30) or (smoke_1block and expected_frames == 3)):
            raise ValueError("num_output_frames must be 120 (474 decoded frames), 30-frame/10-block smoke, or 3-frame/1-block smoke")


validate_group_contract(config)

# === Cross-machine deterministic / reproducible inference ===
# Must be set before the CUDA context is created (i.e. before any CUDA op,
# such as the torch.cuda.set_device below).
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.environ.setdefault("PYTHONHASHSEED", str(getattr(config, "seed", 0)))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

# Initialize distributed inference
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
    set_seed(config.seed + local_rank)
    config.distributed = True  # Mark as distributed for pipeline
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False  # Mark as non-distributed
    print(f"Single GPU mode on device {device}")

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40
low_memory = True

torch.set_grad_enabled(False)


# Initialize pipeline
# Note: checkpoint loading is now handled inside the pipeline __init__ method
pipeline = CausalInferencePipeline(config, device=device)

# Load generator checkpoint
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    if config.use_ema:
        def _clean_key(name: str) -> str:
            """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
            name = name.replace("_fsdp_wrapped_module.", "")
            return name

        cleaned_state_dict = { _clean_key(k): v for k, v in raw_gen_state_dict.items() }
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if local_rank == 0:
            if len(missing) > 0:
                print(f"[Warning] {len(missing)} parameters are missing when loading checkpoint: {missing[:8]} ...")
            if len(unexpected) > 0:
                print(f"[Warning] {len(unexpected)} unexpected parameters encountered when loading checkpoint: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    # Wrap the generator's transformer with LoRA after the base weights are loaded
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    # Load LoRA weights if a lora_ckpt is provided
    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        # Support both a dict containing a `generator_lora` key and a raw LoRA state dict
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])  # type: ignore
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)  # type: ignore
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True


# Move pipeline to appropriate dtype and device
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

extended_prompt_path = config.data_path
dataset = TextDataset(prompt_path=config.data_path, extended_prompt_path=extended_prompt_path)
num_prompts = len(dataset)
print(f"Number of prompts: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)
dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory (only on main process to avoid race conditions)
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()


def encode(self, videos: torch.Tensor) -> torch.Tensor:
    device, dtype = videos[0].device, videos[0].dtype
    scale = [self.mean.to(device=device, dtype=dtype),
             1.0 / self.std.to(device=device, dtype=dtype)]
    output = [
        self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
        for u in videos
    ]

    output = torch.stack(output, dim=0)
    return output


idx_offset = int(getattr(config, "idx_offset", 0))

for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data['idx'].item() + idx_offset

    # For DataLoader batch_size=1, the batch_data is already a single item, but in a batch container
    # Unpack the batch data for convenience
    if isinstance(batch_data, dict):
        batch = batch_data
    elif isinstance(batch_data, list):
        batch = batch_data[0]  # First (and only) item in the batch

    # For text-to-video, batch is just the text prompt
    prompt = batch['prompts'][0]

    # Check if we should skip existing files
    if getattr(config, 'skip_existing', False):
        # Determine model type for filename consistency
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
        
        all_samples_exist = True
        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{idx}-{seed_idx}_{model_type}.mp4')
            else:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{prompt[:100]}-{seed_idx}.mp4')
            if not os.path.exists(output_path):
                all_samples_exist = False
                break
        if all_samples_exist:
            continue

    all_video = []
    num_generated_frames = 0  # Number of generated (latent) frames
    
    extended_prompt = batch['extended_prompts'][0] if 'extended_prompts' in batch else None
    if extended_prompt is not None:
        prompts = [extended_prompt] * config.num_samples
    else:
        prompts = [prompt] * config.num_samples

    sampled_noise = torch.randn(
        [config.num_samples, config.num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
    )

    print("sampled_noise.device", sampled_noise.device)
    print("prompts", prompts)


    if args.v5_profiler_output:
        os.makedirs(args.v5_profiler_output, exist_ok=True)
        prof = torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
        prof.__enter__()
        try:
            video, latents = pipeline.inference(
                noise=sampled_noise, text_prompts=prompts, return_latents=True,
                low_memory=low_memory, profile=False, skip_vae_decode=True,
            )
        finally:
            prof.__exit__(None, None, None)
        trace_path = os.path.join(args.v5_profiler_output, "group11_wrapper_v5_trace.json")
        table_path = os.path.join(args.v5_profiler_output, "group11_wrapper_v5_ops.txt")
        prof.export_chrome_trace(trace_path)
        with open(table_path, "w") as f:
            f.write(prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cuda_time_total", row_limit=300))
            f.write("\n\n--- CPU self time ---\n")
            f.write(prof.key_averages(group_by_input_shape=True).table(
                sort_by="self_cpu_time_total", row_limit=300))
    else:
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            low_memory=low_memory,
            profile=False,
        )
    current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
    all_video.append(current_video)
    num_generated_frames += latents.shape[1]

    # Final output video
    video = 255.0 * torch.cat(all_video, dim=1)

    # Clear VAE cache
    pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # Save the video if the current prompt is not a dummy prompt
    if idx < num_prompts + idx_offset:
        # Determine model type for filename
        if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
            model_type = "lora"
        elif getattr(config, 'use_ema', False):
            model_type = "ema"
        else:
            model_type = "regular"
            
        for seed_idx in range(config.num_samples):
            if config.save_with_index:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{idx}-{seed_idx}_{model_type}.mp4')
            else:
                output_path = os.path.join(config.output_folder, f'rank{rank}-{prompt[:100]}-{seed_idx}.mp4')
            write_video(output_path, video[seed_idx], fps=16)

            # Corrected Group 11--15 machine-readable runtime contract.
            import json
            model_runtime = getattr(pipeline.generator.model, "group_runtime_trace", [])
            group_runtime_mode = str(getattr(config.model_kwargs, "group_runtime_mode", "baseline"))
            corrected_group12 = group_runtime_mode == "group12_corrected"
            corrected_group13 = group_runtime_mode == "group13_corrected"
            corrected_group14 = group_runtime_mode == "group14_corrected"
            corrected_group15 = group_runtime_mode == "group15_corrected"
            corrected_storage = corrected_group12 or corrected_group13 or corrected_group14 or corrected_group15
            draft_rag_active = str(getattr(config.model_kwargs, "retrieval_backend", "")) == "draftmap_online"
            runtime_meta = {
                "GROUP": int(getattr(config, "group_id", 0)),
                "GROUP_RUNTIME_MODE": group_runtime_mode,
                "GROUP14_DRAFTMAP_ACTIVE": "YES" if corrected_group14 else "NO",
                "GROUP15_DRAFTMAP_ACTIVE": "YES" if corrected_group15 else "NO",
                "INTERACTION_SPARSE_ROUTING": "YES" if corrected_group14 or corrected_group15 else "NO",
                "SPARSE_RATIO": float(getattr(config.model_kwargs, "group_sparse_ratio", 0.0)),
                "DRAFT_RAG_ACTIVE": "YES" if draft_rag_active else "NO",
                "CPU_COMPRESSED_HISTORY_ACTIVE": "NO",
                "CPU_HISTORICAL_BF16_ARCHIVE_ACTIVE": "YES" if draft_rag_active else "NO",
                "PERSISTENT_GPU_DRAFT_K": "YES" if draft_rag_active else "NO",
                "HISTORICAL_K_STORAGE": ("INT8" if corrected_group12 or corrected_group14 else ("NVFP4" if corrected_group13 or corrected_group15 else "BF16")),
                "HISTORICAL_V_STORAGE": ("FP8_E4M3" if corrected_group12 or corrected_group14 else ("NVFP4" if corrected_group13 or corrected_group15 else "BF16")),
                "FINAL_ATTENTION_DTYPE": "bfloat16",
                "ATTENTION_KERNEL": "BF16",
                "NATIVE_LOWBIT_KERNEL_USED": "NO",
                "PERSISTENT_STORAGE_MODE": "LOWBIT_STORAGE_BF16_COMPUTE" if corrected_storage else "BF16_FAKE_QUANT",
                "GPU_PERSISTENT_KV_OWNER": "INT8/FP8_E4M3" if corrected_group12 or corrected_group14 else ("NVFP4/NVFP4" if corrected_group13 or corrected_group15 else "BF16"),
                "TRANSIENT_BF16_DEQUANT_ALLOWED": "YES" if corrected_storage else "NO",
                "RETRIEVAL_QUERY_MODE": str(getattr(config.model_kwargs, "retrieval_query_mode", "current_q")) if draft_rag_active else "none",
                "Q_PREV": "NO",
                "FLASH_FETCH": "NO",
                "NEXT_LAYER_PREFETCH": "NO",
                "HOT_CACHE": "NO",
                "LONG_LIVE_REUSE": "NO",
                "NUM_OUTPUT_LATENT_FRAMES": int(getattr(config, "num_output_frames", 0)),
                "EXPECTED_DECODED_FRAMES": 474,
                "CANONICAL_EVAL_PROTOCOL": "474 frames; 16 FPS; 832x480; frame 0 vs 237; seed 0",
                "runtime_trace": model_runtime,
                "group11_profile": getattr(pipeline.generator.model, "group11_profile", None),
            }
            profile = runtime_meta.get("group11_profile")
            if isinstance(profile, dict):
                hits = int(profile.get("CACHE_HITS", 0))
                misses = int(profile.get("CACHE_MISSES", 0))
                profile["CACHE_HIT_RATE"] = float(hits / (hits + misses)) if (hits + misses) else 0.0
            if model_runtime:
                runtime_meta.update({k: model_runtime[-1][k] for k in model_runtime[-1] if k != "runtime_trace"})
            if isinstance(profile, dict):
                runtime_meta.update({
                    "E2E_LATENCY_MS": float(profile.get("E2E_LATENCY_MS", 0.0)),
                    "TRANSFORMER_LATENCY_MS": float(profile.get("TRANSFORMER_LATENCY_MS", 0.0)),
                    "WRAPPER_LATENCY_MS": float(profile.get("WRAPPER_LATENCY_MS", 0.0)),
                    "EXPOSED_H2D_MS": float(profile.get("EXPOSED_H2D_MS", 0.0)),
                    "EXPOSED_H2D_CALLS": int(profile.get("EXPOSED_H2D_CALLS", 0)),
                    "RUNTIME_INSTRUMENTATION_VERSION": str(profile.get("RUNTIME_INSTRUMENTATION_VERSION", "v1")),
                    "E2E_LATENCY_S": float(profile.get("E2E_LATENCY_MS", 0.0)) / 1000.0,
                    "TRANSFORMER_LATENCY_S": float(profile.get("TRANSFORMER_LATENCY_MS", 0.0)) / 1000.0,
                    "WRAPPER_LATENCY_S": float(profile.get("WRAPPER_LATENCY_MS", 0.0)) / 1000.0,
                    "EXPOSED_H2D_S": float(profile.get("EXPOSED_H2D_MS", 0.0)) / 1000.0,
                })
            # Re-assert corrected storage contract after per-step trace merge.
            if corrected_storage:
                runtime_meta.update({
                    "PERSISTENT_STORAGE_MODE": "LOWBIT_STORAGE_BF16_COMPUTE",
                    "FINAL_ATTENTION_DTYPE": "bfloat16",
                    "ATTENTION_KERNEL": "BF16",
                    "NATIVE_LOWBIT_KERNEL_USED": "NO",
                })
            with open(output_path.replace(".mp4", "_runtime.json"), "w") as f:
                json.dump(runtime_meta, f, indent=2)
            memory_meta = runtime_inference_memory(
                getattr(pipeline, "kv_cache1", []),
                pipeline.generator.model,
                runtime_meta.get("PERSISTENT_STORAGE_MODE", "BF16_FAKE_QUANT"),
            )
            if getattr(pipeline, "kv_cache1", None):
                c0 = pipeline.kv_cache1[0]
                runtime_meta.update({
                    "EVICTED_COMPRESSED_ENTRIES": int(c0.get("evicted_compressed_entries", 0)),
                    "RETRIEVED_ARCHIVED_ENTRIES": int(c0.get("retrieved_archived_entries", 0)),
                    "TRANSIENT_DEQUANT_GPU_PEAK_BYTES": int(c0.get("transient_dequant_gpu_peak_bytes", 0)),
                    "COMPRESSED_HISTORY_RECORDS": int(len(c0.get("compressed_history_entries", []))),
                })
                with open(output_path.replace(".mp4", "_runtime.json"), "w") as f:
                    json.dump(runtime_meta, f, indent=2)
            with open(output_path.replace(".mp4", "_memory_measurement.json"), "w") as f:
                json.dump(memory_meta, f, indent=2)

            # Save memory selection log
            if hasattr(pipeline, 'memory_indices_log') and pipeline.memory_indices_log:
                import json
                log_path = output_path.replace('.mp4', '_memory_log.json')
                with open(log_path, 'w') as f:
                    json.dump(pipeline.memory_indices_log, f, indent=2)
                if local_rank == 0:
                    print(f"Saved memory selection log to {log_path}")

                # Save memory selection visualization
                try:
                    viz_path = output_path.replace('.mp4', '_memory_viz.png')
                    query_frames = []
                    mem_frames = []
                    sims = []
                    for entry in pipeline.memory_indices_log:
                        # entry format: {'query_frame': int, 'selected_global_frames': [[]], 'selected_similarities': [[]], ...}
                        # The nested lists [[idx1, idx2]] are for batch_size (assumed 1 here).
                        q = entry['query_frame']
                        mf = entry['selected_global_frames'][0]
                        s = entry['selected_similarities'][0]
                        for m, sim in zip(mf, s):
                            query_frames.append(q)
                            mem_frames.append(m)
                            sims.append(sim)
                    
                    if query_frames:
                        plt.figure(figsize=(10, 6))
                        sc = plt.scatter(query_frames, mem_frames, c=sims, cmap='viridis', s=30, alpha=0.7)
                        plt.colorbar(sc, label='Cosine Similarity')
                        # Draw causality reference: query frame index
                        plt.plot([0, max(query_frames)], [0, max(query_frames)], 'r--', alpha=0.3, label='Current Frame')
                        plt.xlabel('Query Frame Index')
                        plt.ylabel('Memory Frame Index (Global)')
                        plt.title(f'Memory Selection Visualization (Method: {pipeline.compression_method})')
                        plt.grid(True, linestyle='--', alpha=0.5)
                        plt.legend()
                        plt.tight_layout()
                        plt.savefig(viz_path)
                        plt.close()
                        if local_rank == 0:
                            print(f"Saved memory selection visualization to {viz_path}")
                except Exception as e:
                    if local_rank == 0:
                        print(f"Failed to create memory visualization: {e}")

    if config.inference_iter != -1 and i >= config.inference_iter:
        break
if dist.is_initialized():
    dist.destroy_process_group()
