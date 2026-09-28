"""Static Group11–19 experiment semantics; deliberately contains no results."""
from copy import deepcopy

ROOTS = [
    "/home/zju/work/zxl/LongLive-RAG-common-group16_19-v2/results",
    "/home/zju/work/zxl/anemoi_native_draftattention_5090/results",
    "/data/zxl/LongLive-RAG-group11_15_h200/results",
]

def _base(experiment_id, group, method, config, **extra):
    item = {
        "experiment_id": experiment_id, "group": group, "method": method,
        "window": 12, "inference_config_path": config,
        "entrypoint": "inference.py", "quality_protocol": "canonical10",
        "retrieval_strategy": "NOT_IDENTIFIABLE", "fetch_strategy": "NOT_IDENTIFIABLE",
        "cache_strategy": "NOT_IDENTIFIABLE", "prefetch_strategy": "NOT_APPLICABLE",
        "cpu_kv_precision": "NOT_IDENTIFIABLE", "gpu_k_precision": "NOT_IDENTIFIABLE",
        "gpu_v_precision": "NOT_IDENTIFIABLE", "attention_precision": "NOT_IDENTIFIABLE",
        "sparse_policy": "NOT_APPLICABLE", "sparse_retained_ratio": "NOT_APPLICABLE",
        "sparse_dropped_ratio": "NOT_APPLICABLE", "persistent_storage_policy": "NOT_IDENTIFIABLE",
        "attention_interaction_policy": "NOT_IDENTIFIABLE", "storage_k_bit": "NOT_IDENTIFIABLE",
        "storage_v_bit": "NOT_IDENTIFIABLE", "query_compute_bit": "NOT_IDENTIFIABLE",
        "key_compute_bit": "NOT_IDENTIFIABLE", "value_compute_bit": "NOT_IDENTIFIABLE",
        "kernel_backend": "NOT_IDENTIFIABLE", "kernel_precision": "NOT_IDENTIFIABLE",
        "dequant_before_attention": "NOT_IDENTIFIABLE", "direct_lowbit_attention": "NOT_IDENTIFIABLE",
        "mixed_precision_policy": "NOT_IDENTIFIABLE", "is_real_lowbit_kernel": "UNVERIFIED",
        "flash_fetch_execution_class": "NOT_APPLICABLE", "expected_artifact_roots": ROOTS,
        "implementation_semantics_confirmed": "YES", "source_files": [], "config_files": [config],
        "notes": "Static metadata only; no measured values are stored.",
    }
    item.update(extra)
    return item

REGISTRY = {}

REGISTRY["group11.1"] = _base("group11.1", 11, "LongLive-RAG + Draft Attention", "configs/final_group11_bounded.yaml",
    retrieval_strategy="CURRENT_Q_DRAFTMAP", fetch_strategy="SYNCHRONOUS_H2D",
    cache_strategy="BF16_CPU_HISTORY_BF16_GPU_KV", prefetch_strategy="NONE",
    cpu_kv_precision="BF16", gpu_k_precision="BF16", gpu_v_precision="BF16",
    attention_precision="BF16", persistent_storage_policy="BF16_GPU_KV",
    attention_interaction_policy="DENSE_RETAINED", storage_k_bit=16, storage_v_bit=16,
    kernel_backend="DRAFT_ATTENTION", kernel_precision="BF16", direct_lowbit_attention="NO",
    is_real_lowbit_kernel="NO", source_files=["wan/modules/causal_model_latentmem.py"])
REGISTRY["group11.2"] = deepcopy(REGISTRY["group11.1"])
REGISTRY["group11.2"].update(experiment_id="group11.2", method="Group11 previous-Q retrieval",
    retrieval_strategy="PREVIOUS_Q_DIRECT_RETRIEVAL", config_files=["configs/qprev_case01.yaml"],
    inference_config_path="configs/qprev_case01.yaml")
REGISTRY["group11.3"] = deepcopy(REGISTRY["group11.1"])
REGISTRY["group11.3"].update(experiment_id="group11.3", method="Group11 serial FlashFetch",
    fetch_strategy="FLASHFETCH_SERIAL", flash_fetch_execution_class="SERIAL",
    config_files=["configs/flashfetch_case01.yaml"], inference_config_path="configs/flashfetch_case01.yaml",
    notes="FlashFetch correctness path is blocking; async_overlap is false.")
REGISTRY["group11.4"] = deepcopy(REGISTRY["group11.1"])
REGISTRY["group11.4"].update(experiment_id="group11.4", method="Group11 next-layer prefetch",
    fetch_strategy="NEXT_LAYER_PREFETCH", prefetch_strategy="TRUE_ASYNC_OVERLAP",
    config_files=["configs/group11_4_next_layer_prefetch_case01.yaml"],
    inference_config_path="configs/group11_4_next_layer_prefetch_case01.yaml",
    notes="Selected set is reconciled against I_(l+1) before use.")

def _corrected(group, method, config, k, v, sparse="NOT_APPLICABLE", retained="NOT_APPLICABLE"):
    return _base(f"group{group}", group, method, config,
        retrieval_strategy="CURRENT_Q_DRAFTMAP", fetch_strategy="SYNCHRONOUS_H2D",
        cache_strategy="CPU_BF16_ARCHIVE_GPU_PERSISTENT_LOWBITS", prefetch_strategy="NONE",
        cpu_kv_precision="BF16", gpu_k_precision=k, gpu_v_precision=v, attention_precision="BF16",
        sparse_policy="RETAINED_INTERACTION_RATIO" if sparse != "NOT_APPLICABLE" else "DENSE_RETAINED",
        sparse_retained_ratio=retained, sparse_dropped_ratio=(1.0-retained if isinstance(retained, float) else "NOT_APPLICABLE"),
        persistent_storage_policy="GPU_PERSISTENT_LOWBITS_CPU_BF16_ARCHIVE",
        attention_interaction_policy="DRAFTMAP_RETAINED_BLOCKS", storage_k_bit=k, storage_v_bit=v,
        kernel_backend="H200_GROUP_RUNTIME_FAKE_QUANT", kernel_precision="BF16",
        dequant_before_attention="YES", direct_lowbit_attention="NO", mixed_precision_policy="BF16_COMPUTE",
        is_real_lowbit_kernel="NO", source_files=["inference.py", "utils/h200_group_runtime.py", "utils/persistent_draftmap.py"])

REGISTRY["group12"] = _corrected(12, "Group12 corrected INT8/FP8 storage", "configs/group12_15_corrected_campaign/group12_case01.yaml", 8, "FP8_E4M3")
REGISTRY["group13"] = _corrected(13, "Group13 corrected NVFP4 storage", "configs/group12_15_corrected_campaign/group13_case01.yaml", "NVFP4", "NVFP4")
for group in (14, 15):
    for suffix, ratio in (("sparse05", .05), ("sparse10", .10), ("sparse20", .20), ("sparse30", .30)):
        ident = f"group{group}.{('05','10','20','30').index(suffix[-2:]) + 1}"
        k, v = ((8, "FP8_E4M3") if group == 14 else ("NVFP4", "NVFP4"))
        REGISTRY[ident] = _corrected(group, f"Group{group} {suffix} retained interaction", f"configs/group12_15_corrected_campaign/group{group}_{suffix}_case01.yaml", k, v, suffix, ratio)
        REGISTRY[ident]["experiment_id"] = ident
        REGISTRY[ident]["notes"] = "Persistent storage policy is separate from retained interaction sparsity; source route retains ceil(ratio*K)."

def _native(group, method, config, storage, backend, mixed, notes):
    return _base(f"group{group}", group, method, config,
        retrieval_strategy="CURRENT_Q_DRAFTMAP", fetch_strategy="CPU_ARCHIVE_TO_TEMP_GPU",
        cache_strategy=storage, prefetch_strategy="NONE", cpu_kv_precision="COMPRESSED_ARCHIVE",
        attention_precision="FP16", persistent_storage_policy="GPU_CURRENT_WINDOW_CPU_COMPRESSED_ARCHIVE",
        attention_interaction_policy="NATIVE_ROUTE", kernel_backend=backend,
        kernel_precision="SM120_NATIVE", direct_lowbit_attention="YES",
        mixed_precision_policy=mixed, is_real_lowbit_kernel="YES", source_files=["wan/modules/causal_model.py", "utils/anemoi_mpa.py"], notes=notes)

REGISTRY["group16"] = _native(16, "Group16 native mixed HIGH/INT8/ZERO", "configs/group16_19_native/group16_native_w12.yaml", "INT8_K_FP8_V", "sm120_q64_int8_perchunk_vscale_compact_high_attention", "HIGH_FP16_PLUS_INT8_PLUS_ZERO", "Route ratios are runtime defaults/overrides; registry does not store measurements.")
REGISTRY["group17"] = _native(17, "Group17 native NVFP4", "configs/group16_19_native/group17_native_w12.yaml", "NVFP4_KV", "sm120_q64_nvfp4_fp16_attention", "NVFP4_ONLY", "Native all-4-bit route; archived compressed payload is materialized per call.")
REGISTRY["group18"] = _native(18, "Group18 native INT8/FP8", "configs/group16_19_native/group18_native_w12.yaml", "INT8_K_FP8_V", "sm120_q64_int8_perchunk_vscale_attention", "INT8_FP8_PLUS_ZERO", "Source confirms direct low-bit native attention; HIGH/4-bit routes are rejected.")
REGISTRY["group19"] = _base("group19", 19, "Group19 unresolved disabled", "configs/group16_19_native/group19_unresolved_disabled.yaml",
    implementation_semantics_confirmed="NO", is_real_lowbit_kernel="UNVERIFIED", expected_artifact_roots=[],
    notes="Disabled and unresolved mixed 8/4/zero owner/route contract; do not run.")

def resolve_experiment(experiment_id):
    return REGISTRY[experiment_id]

def iter_experiments():
    return (REGISTRY[key] for key in sorted(REGISTRY))

def registry_to_dict():
    return {key: deepcopy(value) for key, value in REGISTRY.items()}
