#!/usr/bin/env bash
set -Eeuo pipefail

ROOT=/data/zxl/LongLive-RAG-group11_15_h200
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
BASE_CFG=$ROOT/configs/group11_15_reference/h200_group11_case01.yaml
MANIFEST=/data/zxl/LongLive-RAG-profile/prompts10.txt
OUT_ROOT=$ROOT/results/group11_canonical10
LOG_ROOT=$OUT_ROOT/logs
TMP_ROOT=$OUT_ROOT/.runtime_tmp
mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$TMP_ROOT"
cd "$ROOT"

expected_sha=db9462c0954186b9128543b58fb6d1d496a8b89e0754333beca698d3cf461fec
actual_sha=$(sha256sum "$MANIFEST" | awk '{print $1}')
test "$actual_sha" = "$expected_sha" || { echo "MANIFEST_MISMATCH"; exit 2; }
test "$(wc -l < "$MANIFEST" | tr -d ' ')" = 10 || { echo "MANIFEST_CASE_COUNT_MISMATCH"; exit 2; }

valid_case() {
  local d="$1"
  local v="$d/rank0-0-0_lora.mp4"
  test -s "$v" && test -s "$d/rank0-0-0_lora_runtime.json" && test -s "$d/rank0-0-0_lora_memory_measurement.json" || return 1
  local frames fps width height
  frames=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 "$v" 2>/dev/null || true)
  fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 "$v" 2>/dev/null || true)
  width=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 "$v" 2>/dev/null || true)
  height=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of csv=p=0 "$v" 2>/dev/null || true)
  test "$frames" = 474 && test "$fps" = 16/1 && test "$width" = 832 && test "$height" = 480
}

make_case_config() {
  local case_id="$1" idx="$2" cfg="$TMP_ROOT/h200_group11_${case_id}.yaml" prompt="$TMP_ROOT/${case_id}.txt" out="$OUT_ROOT/$case_id"
  sed -n "${idx}p" "$MANIFEST" > "$prompt"
  test -s "$prompt"
  sed -e "s#^data_path: .*#data_path: $prompt#" \
      -e "s#^output_folder: .*#output_folder: $out#" \
      -e 's#^inference_iter: .*#inference_iter: -1#' \
      "$BASE_CFG" > "$cfg"
  printf '%s\n' "$cfg"
}

run_case() {
  local idx="$1" gpu="$2" case_id
  case_id=$(printf 'case_%02d' "$idx")
  local out="$OUT_ROOT/$case_id" log="$LOG_ROOT/$case_id.log" cfg
  if [ "$idx" -eq 1 ]; then out="$ROOT/results/group11_case01"; fi
  if valid_case "$out"; then
    test -s "$out/config_used.yaml" || cp "$BASE_CFG" "$out/config_used.yaml"
    test -s "$out/latency.json" || printf '%s\n' '{"latency_s":"NOT_AVAILABLE","source":"pre-existing case01 artifact without wall-clock profile"}' > "$out/latency.json"
    echo "SKIP_VALID $case_id"
    return 0
  fi
  cfg=$(make_case_config "$case_id" "$idx")
  mkdir -p "$out"
  cp "$cfg" "$out/config_used.yaml"
  local start_ns end_ns rc=0
  start_ns=$(date +%s%N)
  {
    echo "START_UTC=$(date -u +%FT%TZ)"
    echo "CASE=$case_id GPU=$gpu"
    echo "CONFIG=$cfg"
    CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" --config_path "$cfg"
  } >"$log" 2>&1 || rc=$?
  end_ns=$(date +%s%N)
  "$PY" - "$out/latency.json" "$start_ns" "$end_ns" "$rc" <<'PY'
import json, sys
out, start, end, rc = sys.argv[1:]
json.dump({"latency_s": (int(end)-int(start))/1e9, "exit_code": int(rc), "measured_by": "runner wall clock"}, open(out,"w"), indent=2)
PY
  if [ "$rc" -ne 0 ] || ! valid_case "$out"; then
    echo "FAILED_OR_INVALID $case_id rc=$rc" >> "$log"
    return 1
  fi
  echo "DONE $case_id" >> "$log"
}

worker() {
  local gpu="$1" idx
  for idx in $(seq "$2" 2 10); do
    local case_id; case_id=$(printf 'case_%02d' "$idx")
    while ! nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | awk '($1+0)>20000 {ok=1} END{exit !ok}'; do
      echo "WAITING_FOR_GPU gpu=$gpu case=$case_id $(date -u +%FT%TZ)"
      sleep 20
    done
    if ! run_case "$idx" "$gpu"; then
      echo "RETRY_ONCE case=$case_id" | tee -a "$LOG_ROOT/$case_id.log"
      run_case "$idx" "$gpu" || echo "FAILED_AFTER_RETRY case=$case_id" | tee -a "$LOG_ROOT/$case_id.log"
    fi
  done
}

echo "GROUP11_CANONICAL10_START $(date -u +%FT%TZ)"
worker 0 1 & p0=$!
worker 1 2 & p1=$!
wait "$p0"; r0=$?
wait "$p1"; r1=$?
echo "GROUP11_CANONICAL10_END $(date -u +%FT%TZ) rc0=$r0 rc1=$r1"
exit $((r0 || r1))
