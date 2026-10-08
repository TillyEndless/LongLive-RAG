#!/usr/bin/env bash
set -u
ROOT=/data/zxl/LongLive-RAG-group11_15_h200
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
cd "$ROOT"
export RAG_STRATEGY_PROFILE=1
export PYTHONPATH="$ROOT"
for g in 2 3 4; do
  echo "START_GROUP11_${g} $(date -u +%FT%TZ)"
  CUDA_VISIBLE_DEVICES=1 "$PY" -u inference.py --config_path "$ROOT/configs/g11_${g}.yaml" > "/tmp/ragprof_g11_${g}.log" 2>&1
  rc=$?
  echo "END_GROUP11_${g} RC=$rc $(date -u +%FT%TZ)"
  if [ "$rc" -ne 0 ]; then exit "$rc"; fi
done
"$PY" scripts/aggregate_rag_strategy_profile.py "$ROOT/results/rag_strategy_profile_runs" "$ROOT/results" "$ROOT/reports/rag_strategy_profile.md"
