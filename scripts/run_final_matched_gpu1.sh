#!/usr/bin/env bash
set -u
set -o pipefail
ROOT=/data/zxl/LongLive-RAG-group11_15_h200
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
STAMP=20260927_1828
G11_OUT=$ROOT/results/final_group11_gpu1_${STAMP}
LL_OUT=$ROOT/results/final_longlive_gpu1_${STAMP}
mkdir -p "$G11_OUT" "$LL_OUT"

echo "GPU1 matched profiling start $(date -Is)" | tee "$ROOT/results/final_matched_gpu1_${STAMP}.log"
nvidia-smi | tee -a "$ROOT/results/final_matched_gpu1_${STAMP}.log"

cd "$ROOT"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$ROOT PROFILE_REPO=$ROOT PROFILE_CONFIG=$ROOT/configs/final_group11_gpu1.yaml PROFILE_OUT=$G11_OUT PROFILE_MODULE=latentmem PROFILE_START_BLOCK=6 PROFILE_END_BLOCK=8 PYTHONUNBUFFERED=1 \
  "$PY" "$ROOT/scripts/final_matched_event_launcher.py" >> "$ROOT/results/final_matched_gpu1_${STAMP}.log" 2>&1
echo "GROUP11_EXIT=$?" | tee -a "$ROOT/results/final_matched_gpu1_${STAMP}.log"

nvidia-smi | tee -a "$ROOT/results/final_matched_gpu1_${STAMP}.log"
cd /data/zxl/LongLive-RAG
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/data/zxl/LongLive-RAG PROFILE_REPO=/data/zxl/LongLive-RAG PROFILE_CONFIG=/data/zxl/LongLive-RAG-profile/config_final_longlive_gpu1.yaml PROFILE_OUT=$LL_OUT PROFILE_MODULE=native PROFILE_START_BLOCK=6 PROFILE_END_BLOCK=8 PYTHONUNBUFFERED=1 \
  "$PY" "$ROOT/scripts/final_matched_event_launcher.py" >> "$ROOT/results/final_matched_gpu1_${STAMP}.log" 2>&1
echo "LONGLIVE_EXIT=$?" | tee -a "$ROOT/results/final_matched_gpu1_${STAMP}.log"
nvidia-smi | tee -a "$ROOT/results/final_matched_gpu1_${STAMP}.log"
echo "DONE $(date -Is)" | tee -a "$ROOT/results/final_matched_gpu1_${STAMP}.log"
