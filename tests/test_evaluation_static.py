import json

from evaluation.experiment_registry import REGISTRY
from evaluation.static_audit import action_plan, discover_artifacts


def test_registry_contains_all_required_experiments():
    assert set(REGISTRY) == {"group11.1", "group11.2", "group11.3", "group11.4", "group12", "group13",
                             "group14.1", "group14.2", "group14.3", "group14.4", "group15.1", "group15.2",
                             "group15.3", "group15.4", "group16", "group17", "group18", "group19"}


def test_registry_preserves_semantic_distinctions():
    assert REGISTRY["group11.2"]["retrieval_strategy"] == "PREVIOUS_Q_DIRECT_RETRIEVAL"
    assert REGISTRY["group11.3"]["flash_fetch_execution_class"] == "SERIAL"
    assert REGISTRY["group11.4"]["prefetch_strategy"] == "TRUE_ASYNC_OVERLAP"
    assert REGISTRY["group12"]["persistent_storage_policy"] == "GPU_PERSISTENT_LOWBITS_CPU_BF16_ARCHIVE"
    assert REGISTRY["group14.1"]["sparse_retained_ratio"] == .05
    assert REGISTRY["group14.1"]["persistent_storage_policy"] != REGISTRY["group14.1"]["sparse_policy"]
    assert REGISTRY["group16"]["is_real_lowbit_kernel"] == "YES"
    assert "nvfp4" in REGISTRY["group17"]["kernel_backend"]
    assert REGISTRY["group19"]["implementation_semantics_confirmed"] == "NO"


def test_legacy_latency_is_discoverable_but_not_reusable(tmp_path):
    (tmp_path / "runtime.csv").write_text("metric,value\nE2E_LATENCY_S,1\n")
    artifacts = discover_artifacts(REGISTRY["group16"], [tmp_path])
    assert artifacts["latency"]["found"] is True
    assert artifacts["latency"]["reusable"] is False


def test_versioned_latency_and_memory_can_be_reused(tmp_path):
    (tmp_path / "profile.json").write_text(json.dumps({
        "timer_boundary_version": "v1", "e2e_latency_s": 2, "persistent_gpu_kv_bytes": 8,
        "persistent_cpu_kv_bytes": 16}))
    (tmp_path / "manifest.json").write_text("{}")
    artifacts = discover_artifacts(REGISTRY["group16"], [tmp_path])
    assert artifacts["latency"]["reusable"] is True
    assert artifacts["memory"]["reusable"] is True


def test_planner_keeps_disabled_group19_out_of_execution():
    artifacts = {key: {"reusable": True} for key in ("quality", "latency", "memory", "provenance")}
    assert action_plan(REGISTRY["group19"], artifacts) == "PROVENANCE_AUDIT_REQUIRED"


def test_planner_requires_full_inference_when_no_evidence_exists():
    artifacts = {key: {"found": False, "reusable": False} for key in ("quality", "latency", "memory", "provenance")}
    assert action_plan(REGISTRY["group12"], artifacts) == "FULL_INFERENCE_REQUIRED"
