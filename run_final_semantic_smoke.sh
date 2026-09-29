#!/usr/bin/env bash
set -u
ROOT=/data/zxl/LongLive-RAG-group12_15_local_lowbit_v2
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
cd "$ROOT"
for g in 12 13 14 15; do
  echo "RUN group$g $(date -Is)"
  CUDA_VISIBLE_DEVICES=0 "$PY" -u inference.py --config_path "configs/group11_15_final_semantic_smoke/group${g}.yaml"
  rc=$?
  echo "DONE group$g rc=$rc $(date -Is)"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
echo SMOKE_RUN_COMPLETE
