#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/zxl/LongLive-RAG-group11_15_h200
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
BASE=$ROOT/configs/h200_group14_corrected_screen20_case01.yaml
OUTROOT=$ROOT/results/group14_corrected_screening
mkdir -p "$OUTROOT"
while ps -eo args | grep -F "$ROOT/configs/h200_group14_corrected_screen20_case01.yaml" | grep -v grep >/dev/null; do
  echo "WAIT_CURRENT_20_CASE01 $(date -u +%FT%TZ)"
  sleep 20
done
for ratio in 30 10 5; do
  for idx in 1 5 9; do
    case_id=$(printf "case_%02d" "$idx")
    tag="screen${ratio}_${case_id}"
    out="$OUTROOT/$tag"
    cfg="$OUTROOT/$tag.yaml"
    log="$OUTROOT/$tag.log"
    if test -s "$out/rank0-0-0_lora.mp4"; then echo "SKIP_VALID $tag"; continue; fi
    prompt="$OUTROOT/$tag.txt"
    sed -n "${idx}p" /data/zxl/LongLive-RAG-profile/prompts10.txt > "$prompt"
    sed -e "s#^data_path: .*#data_path: $prompt#" \
        -e "s#^output_folder: .*#output_folder: $out#" \
        -e "s#^  group_sparse_ratio:.*#  group_sparse_ratio: 0.$ratio#" \
        -e "s#^inference_iter: .*#inference_iter: 0#" "$BASE" > "$cfg"
    mkdir -p "$out"
    rc=0
    { echo "START_UTC=$(date -u +%FT%TZ) TAG=$tag GPU=0"; CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" --config_path "$cfg"; } >"$log" 2>&1 || rc=$?
    test -s "$out/rank0-0-0_lora.mp4" || { echo "FAILED $tag rc=$rc" >>"$log"; exit 1; }
    echo "DONE $tag" >>"$log"
  done
done
echo "GROUP14_SCREENING_DONE $(date -u +%FT%TZ)"
