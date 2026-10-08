#!/usr/bin/env bash
set -u
ROOT=/data/zxl/LongLive-RAG-group11_15_h200
OUT=$ROOT/results/group11_2_temporal_prefetch_case01_profile
CFG=$OUT/config.yaml
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
while true; do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F',' '$1 ~ /^ *1 *$/ {gsub(/ /,"",$2); print $2}')
  if [ -n "${used:-}" ] && [ "$used" -le 1000 ]; then
    break
  fi
  echo "WAITING_FOR_GPU1 used=${used:-unknown} MiB $(date -Is)" >> "$OUT/queue.log"
  sleep 20
done
echo "START_GPU1 $(date -Is)" >> "$OUT/queue.log"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=1 RAG_PROFILE_CASE_ID=case_01 RAG_STRATEGY_PROFILE=1 QPREV_ALIGNMENT_ASSERT=1
/usr/bin/time -p -o "$OUT/process_wall.txt" "$PY" inference.py --config_path "$CFG" > "$OUT/inference.log" 2>&1
rc=$?
echo "EXIT_CODE=$rc $(date -Is)" >> "$OUT/queue.log"
exit "$rc"
