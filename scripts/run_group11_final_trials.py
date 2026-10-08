#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from omegaconf import OmegaConf


STRATEGIES = {
    "CURRENT_Q": ("/data/zxl/LongLive-RAG-group11_15_h200", "configs/g11_1.yaml"),
    "PREVIOUS_Q": ("/data/zxl/LongLive-RAG-group11_15_h200", "configs/g11_2.yaml"),
    "FLASH_FETCH": ("/data/zxl/LongLive-RAG-group11_flashfetch_h200", "configs/flashfetch_stock_case01_v2.yaml"),
    "NEXT_LAYER_PREFETCH": ("/data/zxl/LongLive-RAG-group11_15_h200", "configs/g11_4_overlap.yaml"),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(root):
    commit = subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", root, "status", "--short"], text=True)
    return {"root": root, "commit": commit, "status": status}


def run_one(strategy, trial, gpu, outroot):
    root, cfg_rel = STRATEGIES[strategy]
    source_cfg = Path(root) / cfg_rel
    run_dir = Path(outroot) / strategy.lower() / f"trial_{trial:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.yaml"
    cfg = OmegaConf.load(str(source_cfg))
    cfg.output_folder = str(run_dir / "inference")
    cfg.skip_existing = False
    cfg.inference_iter = -1
    cfg.data_path = "/data/zxl/strict_latency_case01_20260928/case_01.txt"
    OmegaConf.save(cfg, str(cfg_path))
    (run_dir / "source_config.sha256").write_text(sha256(source_cfg) + "\n")
    (run_dir / "git_info.json").write_text(json.dumps(git_info(root), indent=2))
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "RAG_STRATEGY_PROFILE": "1",
        "RAG_PROFILE_CASE_ID": "case_01",
        "PYTHONPATH": root,
        "HF_HOME": "/data/zxl/.cache/huggingface",
    })
    py = "/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python"
    log = open(run_dir / "runtime.log", "w")
    start = time.time()
    meta = {"strategy": strategy, "trial": trial, "gpu": gpu, "start_epoch": start,
            "source_root": root, "source_config": str(source_cfg), "config": str(cfg_path)}
    (run_dir / "trial_start.json").write_text(json.dumps(meta, indent=2))
    proc = subprocess.run([py, "-u", "inference.py", "--config_path", str(cfg_path)],
                          cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    end = time.time()
    meta.update({"end_epoch": end, "wall_process_s": end - start, "returncode": proc.returncode})
    runtime_files = sorted((run_dir / "inference").glob("*_runtime.json"))
    if runtime_files:
        shutil.copy2(runtime_files[-1], run_dir / "runtime.json")
    memory_files = sorted((run_dir / "inference").glob("*_memory_measurement.json"))
    if memory_files:
        shutil.copy2(memory_files[-1], run_dir / "memory.json")
    (run_dir / "trial_end.json").write_text(json.dumps(meta, indent=2))
    if proc.returncode:
        raise SystemExit(f"{strategy} trial {trial} failed with rc={proc.returncode}; see {run_dir / 'runtime.log'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--gpu", default="1")
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()
    order = ["CURRENT_Q", "PREVIOUS_Q", "FLASH_FETCH", "NEXT_LAYER_PREFETCH"]
    root = Path(args.outroot)
    root.mkdir(parents=True, exist_ok=True)
    for strategy in order:
        for trial in range(1, args.trials + 1):
            run_one(strategy, trial, args.gpu, root)
    (root / "RUN_COMPLETE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n")


if __name__ == "__main__":
    main()
