#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/zxl/LongLive-RAG-group11_15_h200
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
OUT=$ROOT/results/group14_15_canonical30_run
mkdir -p "$OUT"
cd "$ROOT"

echo "START_UTC=$(date -u +%FT%TZ) GPU=1" > "$OUT/runner.log"

for group in 14 15; do
  if test "$group" -eq 14; then
    indices=$(seq 7 10)
  else
    indices=$(seq 2 10)
  fi
  for idx in $indices; do
    case_id=$(printf 'case%02d' "$idx")
    tag="group${group}_${case_id}_retained30"
    cfg="$ROOT/configs/${tag}.yaml"
    out="$ROOT/results/$tag"
    log="$OUT/${tag}.log"

    if test -s "$out/rank0-0-0_lora.mp4"; then
      echo "SKIP_EXISTING $tag" | tee -a "$OUT/runner.log"
      continue
    fi

    echo "START $tag $(date -u +%FT%TZ)" | tee -a "$OUT/runner.log"
    nvidia-smi >> "$log" 2>&1
    mkdir -p "$out"
    set +e
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" --config_path "$cfg" >> "$log" 2>&1
    rc=$?
    set -e
    if test "$rc" -ne 0 || ! test -s "$out/rank0-0-0_lora.mp4"; then
      echo "FAILED $tag rc=$rc $(date -u +%FT%TZ)" | tee -a "$OUT/runner.log"
      exit 1
    fi
    frames=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$out/rank0-0-0_lora.mp4" 2>/dev/null || true)
    if test "$frames" != "474"; then
      echo "INVALID_FRAMES $tag frames=$frames" | tee -a "$OUT/runner.log"
      exit 1
    fi
    echo "DONE $tag frames=$frames $(date -u +%FT%TZ)" | tee -a "$OUT/runner.log"
  done
done

echo "FINISHED_UTC=$(date -u +%FT%TZ)" | tee -a "$OUT/runner.log"
