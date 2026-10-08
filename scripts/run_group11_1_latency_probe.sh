#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/zxl/LongLive-RAG-group11_15_h200
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
BASE=$ROOT/configs/group11_15_reference/h200_group11_case01.yaml
MANIFEST=/data/zxl/LongLive-RAG-profile/prompts10.txt
OUT=$ROOT/results/latency_instrumented/group11_1
TMP=$OUT/configs
LOG=$OUT/runner.log
mkdir -p "$TMP" "$OUT"
test "$(sha256sum "$MANIFEST" | awk '{print $1}')" = db9462c0954186b9128543b58fb6d1d496a8b89e0754333beca698d3cf461fec
for i in 1 2 3; do
  c=$(printf 'case_%02d' "$i"); d=$OUT/$c; mkdir -p "$d"; p=$TMP/$c.txt; cfg=$TMP/$c.yaml
  if [ -s "$d"/rank0-0-0_lora.mp4 ] && [ -s "$d"/rank0-0-0_lora_runtime.json ]; then echo "SKIP_VALID $c" >> "$LOG"; continue; fi
  sed -n "${i}p" "$MANIFEST" > "$p"
  sed -e "s#^data_path: .*#data_path: $p#" -e "s#^output_folder: .*#output_folder: $d#" -e 's#^inference_iter: .*#inference_iter: -1#' "$BASE" > "$cfg"
  echo "START $c GPU1 $(date -u +%FT%TZ)" >> "$LOG"
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" --config_path "$cfg" >> "$LOG" 2>&1
  echo "DONE $c $(date -u +%FT%TZ)" >> "$LOG"
done
echo COMPLETE >> "$LOG"
