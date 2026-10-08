#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/zxl/LongLive-RAG-group11_15_h200
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
for i in $(seq 2 10); do
  case_id=$(printf "case_%02d" "$i")
  out="$ROOT/results/group12_corrected_canonical10/$case_id"
  v="$out/rank0-0-0_lora.mp4"
  if test -s "$v"; then echo "SKIP_VALID $case_id"; continue; fi
  cfg="$ROOT/results/group12_corrected_canonical10/.runtime_tmp/$case_id.yaml"
  log="$ROOT/results/group12_corrected_canonical10/logs/$case_id.log"
  rc=0
  { echo "START_UTC=$(date -u +%FT%TZ) CASE=$case_id GPU=0"; CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" --config_path "$cfg"; } >"$log" 2>&1 || rc=$?
  if test "$rc" -ne 0 || ! test -s "$v"; then
    echo "RETRY_ONCE $case_id" >>"$log"
    { CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" --config_path "$cfg"; } >>"$log" 2>&1 || rc=$?
  fi
  test -s "$v" || { echo "FAILED $case_id rc=$rc" >>"$log"; exit 1; }
  echo "DONE $case_id" >>"$log"
done
echo "GROUP12_CORRECTED_CANONICAL10_DONE $(date -u +%FT%TZ)"
