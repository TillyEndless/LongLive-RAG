#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/zxl/LongLive-RAG-group11_15_h200
cd "$ROOT"
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
OUT=$ROOT/results/group13_corrected_canonical10
export TMPDIR=$ROOT/.tmp_group13
mkdir -p "$OUT/case_01"
for f in "$ROOT/results/group13_corrected_case01_rerun4/"*; do
  b=$(basename "$f")
  ln -sfn "$f" "$OUT/case_01/$b"
done
for i in $(seq 2 10); do
  cid=$(printf "case_%02d" "$i")
  out="$OUT/$cid"
  v="$out/rank0-0-0_lora.mp4"
  if test -s "$v"; then
    echo "SKIP_VALID $cid"
    continue
  fi
  cfg="$OUT/.runtime_tmp/$cid.yaml"
  log="$OUT/logs/$cid.log"
  rc=0
  { echo "START_UTC=$(date -u +%FT%TZ) CASE=$cid GPU=1"; CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" --config_path "$cfg"; } >"$log" 2>&1 || rc=$?
  if test "$rc" -ne 0 || ! test -s "$v"; then
    echo "RETRY_ONCE $cid" >>"$log"
    { CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" --config_path "$cfg"; } >>"$log" 2>&1 || rc=$?
  fi
  test -s "$v" || { echo "FAILED $cid rc=$rc" >>"$log"; exit 1; }
  echo "DONE $cid" >>"$log"
done
echo "GROUP13_CORRECTED_CANONICAL10_DONE $(date -u +%FT%TZ)"
