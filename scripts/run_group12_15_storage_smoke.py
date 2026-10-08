import os
import subprocess
from pathlib import Path

configs = {
    12: "configs/g12_currentq_int8_fp8_smoke.yaml",
    13: "configs/g13_currentq_nvfp4_smoke.yaml",
    14: "configs/g14_currentq_int8_fp8_promotion_smoke.yaml",
    15: "configs/g15_currentq_nvfp4_promotion_smoke.yaml",
}
root = Path.cwd()
py = "/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python"
for group, config in configs.items():
    print(f"START_GROUP{group}", flush=True)
    log = root / "results" / f"group{group}_storage_smoke.log"
    with log.open("w") as fh:
        proc = subprocess.Popen([py, "inference.py", "--config_path", config],
                                cwd=root, env={**os.environ, "CUDA_VISIBLE_DEVICES": "1"},
                                stdout=fh, stderr=subprocess.STDOUT)
        rc = proc.wait()
    print(f"GROUP{group}_RC={rc}", flush=True)
    if rc != 0:
        raise SystemExit(rc)
print("SMOKE_COMPLETE", flush=True)
