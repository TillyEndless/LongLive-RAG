#!/usr/bin/env bash
set -Eeuo pipefail
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
FLASH=/data/zxl/LongLive-RAG-group11_flashfetch_h200
PREF=/data/zxl/LongLive-RAG-group11_next_layer_prefetch_h200
ROOT=/data/zxl/LongLive-RAG-group11_15_h200/results/group11_accel_smoke_gpu0
mkdir -p "$ROOT"
echo "START_UTC=$(date -u +%FT%TZ)" > "$ROOT/state.log"
nvidia-smi -i 0 > "$ROOT/gpu_before.txt" 2>&1 || true
echo "START_11_3=$(date -u +%FT%TZ)" >> "$ROOT/state.log"
(cd "$FLASH" && CUDA_VISIBLE_DEVICES=0 "$PY" -u inference.py --config_path configs/flashfetch_smoke_gpu0.yaml) > "$ROOT/group11_3.log" 2>&1 || echo "GROUP11_3_RC=$?" >> "$ROOT/state.log"
echo "END_11_3=$(date -u +%FT%TZ)" >> "$ROOT/state.log"
nvidia-smi -i 0 > "$ROOT/gpu_between.txt" 2>&1 || true
echo "START_11_4=$(date -u +%FT%TZ)" >> "$ROOT/state.log"
(cd "$PREF" && CUDA_VISIBLE_DEVICES=0 "$PY" -u inference.py --config_path configs/prefetch_smoke_gpu0.yaml) > "$ROOT/group11_4.log" 2>&1 || echo "GROUP11_4_RC=$?" >> "$ROOT/state.log"
echo "END_11_4=$(date -u +%FT%TZ)" >> "$ROOT/state.log"
nvidia-smi -i 0 > "$ROOT/gpu_after.txt" 2>&1 || true
echo "FINISHED_UTC=$(date -u +%FT%TZ)" >> "$ROOT/state.log"
