import unittest
import torch
from unified_evaluation_schema import normalize_h2d_runtime, as_csv_row
from utils.runtime_memory_measurement import runtime_inference_memory, update_persistent_kv_peak

class H2DSemanticsTests(unittest.TestCase):
    def test_zero_is_not_missing(self):
        x = normalize_h2d_runtime({"H2D_CUDA_WORK_S": 0.0, "H2D_WORK_VALID": True})
        self.assertEqual(x["H2D_CUDA_WORK_S"], 0.0)
        self.assertTrue(x["H2D_WORK_VALID"])

    def test_legacy_mapping_is_explicit(self):
        x = normalize_h2d_runtime({"h2d_latency_s": 7.3})
        self.assertEqual(x["H2D_CUDA_WORK_S"], 7.3)
        self.assertEqual(x["H2D_SOURCE_CLASS"], "LEGACY_HISTORICAL_FETCH")
        self.assertIn("legacy_cuda_interval", x["H2D_TIMING_CLASS"])

    def test_missing_fields_are_not_available(self):
        x = normalize_h2d_runtime({})
        self.assertEqual(x["H2D_CUDA_WORK_S"], "NOT_AVAILABLE")
        self.assertEqual(x["H2D_EXPOSED_WAIT_S"], "NOT_AVAILABLE")
        self.assertEqual(x["H2D_HIDDEN_S"], "NOT_AVAILABLE")

    def test_persistent_peak_survives_eviction(self):
        cache = {"k": torch.zeros(1, 8, 2, 2), "v": torch.zeros(1, 8, 2, 2), "gpu_draft_k_frames": []}
        first = update_persistent_kv_peak([cache])
        self.assertGreater(first["GPU_KV_PERSISTENT_PEAK_BYTES"], 0)
        cache["k"] = torch.zeros(1, 2, 2, 2)
        cache["v"] = torch.zeros(1, 2, 2, 2)
        second = update_persistent_kv_peak([cache])
        self.assertLess(second["GPU_KV_ACTUAL_PERSISTENT_BYTES"], first["GPU_KV_PERSISTENT_PEAK_BYTES"])
        self.assertEqual(second["GPU_KV_PERSISTENT_PEAK_BYTES"], first["GPU_KV_PERSISTENT_PEAK_BYTES"])

    def test_draft_is_separate_from_kv_compression(self):
        cache = {
            "k": torch.zeros(1, 4, 1, 2), "v": torch.zeros(1, 4, 1, 2),
            "gpu_draft_k_frames": [torch.zeros(1, 3, 1, 2)],
        }
        m = runtime_inference_memory([cache], None)
        self.assertGreater(m["DRAFT_GPU_PERSISTENT_BYTES"], 0)
        self.assertEqual(m["GPU_KV_ACTUAL_PERSISTENT_BYTES"], m["GPU_LOCAL_PERSISTENT_KV_BYTES"])
        self.assertNotEqual(m["GPU_METHOD_PERSISTENT_BYTES"], m["GPU_KV_ACTUAL_PERSISTENT_BYTES"])

    def test_schema_serializes_explicit_columns(self):
        summary = {
            "quality": {"DINO": 0.1, "SSIM": 0.2, "PSNR": 1.0, "LPIPS": "NOT_AVAILABLE"},
            "memory": {"GPU_KV_GIB": 1.0, "CPU_KV_GIB": 0.0},
            "latency": {"E2E_INFERENCE_S": 1.0, "TRANSFORMER_S": 0.8, "SELF_ATTN_WRAPPER_S": 0.2, "ATTENTION_KERNEL_S": 0.1},
            "provenance": {}, "validity": {},
        }
        row = as_csv_row(summary)
        self.assertIn("H2D CUDA Work", row)
        self.assertIn("H2D Source Class", row)

if __name__ == "__main__":
    unittest.main()
