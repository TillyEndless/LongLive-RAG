#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/zxl/LongLive-RAG-group11_15_h200
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
CFG=$ROOT/configs/group11_15_reference
LOG=$ROOT/results/group11_15_acceptance_logs
mkdir -p "$LOG"
cd "$ROOT"
run_one() {
  local group="$1" gpu="$2"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" \
    --config_path "$CFG/h200_group${group}_case01.yaml" \
    >"$LOG/group${group}.log" 2>&1
}
run_one 11 0 & p11=$!
run_one 12 1 & p12=$!
run_one 13 0 & p13=$!
run_one 14 1 & p14=$!
run_one 15 0 & p15=$!
rc=0
for p in $p11 $p12 $p13 $p14 $p15; do wait "$p" || rc=1; done
exit "$rc"
