#!/usr/bin/env bash
set -Eeuo pipefail
ROOT=/data/zxl/LongLive-RAG-group11_15_h200
PY=/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python
export TMPDIR=$ROOT/.tmp_group13
cd "$ROOT"
LOG=$ROOT/results/group15_screening/supervisor.log
mkdir -p "$ROOT/results/group15_screening"
echo "SUPERVISOR_START_UTC=$(date -u +%FT%TZ) GPU=1" >> "$LOG"
while pgrep -f "run_group13_corrected_gpu1.sh" >/dev/null; do
  echo "WAITING_FOR_GROUP13 $(date -u +%FT%TZ)" >> "$LOG"
  sleep 20
done
count=$(find "$ROOT/results/group13_corrected_canonical10" -path "*/rank0-0-0_lora.mp4" -type f | wc -l)
if [ "$count" -ne 10 ]; then echo "GROUP13_CANONICAL10_FAIL videos=$count" >> "$LOG"; exit 1; fi
CUDA_VISIBLE_DEVICES=1 "$PY" "$ROOT/scripts/evaluate_group13_15.py" --root "$ROOT/results/group13_corrected_canonical10" --out "$ROOT/results/group13_corrected_canonical10" >> "$LOG" 2>&1
echo GROUP13_EVALUATION_DONE_STOP_BEFORE_GROUP15 $(date -u +%FT%TZ) >> "$LOG"
exit 0
for pct in 20 30 10 05; do
  for cid in case_01 case_05 case_09; do
    out="$ROOT/results/group15_screening/sparsity_\${pct}/$cid"; v="$out/rank0-0-0_lora.mp4"
    [ -s "$v" ] && continue
    cfg="$ROOT/results/group15_screening/.runtime_tmp/\${pct}_\${cid}.yaml"; log="$ROOT/results/group15_screening/logs/\${pct}_\${cid}.log"; mkdir -p "$(dirname "$log")"; rc=0
    { echo "START GPU=1 SPARSITY=$pct CASE=$cid"; CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" --config_path "$cfg"; } > "$log" 2>&1 || rc=$?
    if [ "$rc" -ne 0 ] || [ ! -s "$v" ]; then { CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" --config_path "$cfg"; } >> "$log" 2>&1 || rc=$?; fi
    [ -s "$v" ] || { echo "SCREEN_FAILED pct=$pct case=$cid" >> "$LOG"; exit 1; }
  done
  CUDA_VISIBLE_DEVICES=1 "$PY" "$ROOT/scripts/evaluate_group13_15.py" --root "$ROOT/results/group15_screening/sparsity_\${pct}" --out "$ROOT/results/group15_screening/sparsity_\${pct}" >> "$LOG" 2>&1
done
"$PY" - <<'PY' >> "$LOG"
import json
from pathlib import Path
root=Path('/data/zxl/LongLive-RAG-group11_15_h200')
ref=json.loads((root/'results/group13_corrected_canonical10/summary.json').read_text())
rows=[]
for pct in ('05','10','20','30'):
 p=root/f'results/group15_screening/sparsity_{pct}/summary.json'
 if p.exists():
  d=json.loads(p.read_text()); d['sparsity_target']=int(pct)/100; d['pass']=d['DINO']>=.98*ref['DINO'] and d['SSIM']>=.98*ref['SSIM']; rows.append(d)
(root/'results/group15_screening/selection.json').write_text(json.dumps({'reference':ref,'candidates':rows},indent=2))
passed=[x for x in rows if x['pass']]
selected=max(passed,key=lambda x:x['sparsity_target']) if passed else None
(root/'results/group15_screening/selected.json').write_text(json.dumps(selected or {'selected':None},indent=2))
print('REFERENCE',ref,'CANDIDATES',rows,'SELECTED',selected)
PY
selected=$("$PY" -c "import json; d=json.load(open('$ROOT/results/group15_screening/selected.json')); print('' if d.get('selected') is None else int(round(d['sparsity_target']*100)))")
if [ -z "$selected" ]; then echo "GROUP15_SELECTED_SPARSITY=NONE" >> "$LOG"; exit 0; fi
echo "GROUP15_SELECTED_SPARSITY=$selected" >> "$LOG"
"$PY" - <<PY
from pathlib import Path
import yaml
root=Path('$ROOT'); pct=int('$selected'); out=root/'results/group15_corrected_canonical10'; tmp=out/'.runtime_tmp'; tmp.mkdir(parents=True,exist_ok=True)
lines=[x for x in Path('/data/zxl/LongLive-RAG-profile/prompts10.txt').read_text().splitlines() if x.strip()]; base=root/'results/group13_corrected_canonical10/.runtime_tmp/case_02.yaml'
for i in range(1,11):
 cid=f'case_{i:02d}'; d=yaml.safe_load(base.read_text()); d['group_id']=15; d['data_path']=str(tmp/f'{cid}.txt'); (tmp/f'{cid}.txt').write_text(lines[i-1]+'\\n'); d['output_folder']=str(out/cid); d['model_kwargs']['group_runtime_mode']='group15_corrected'; d['model_kwargs']['group_sparse_ratio']=pct/100; d['model_kwargs']['retrieval_backend']='draftmap_online'; d['model_kwargs']['memory_size']=6
 for k,v in {'historical_kv_archive':'nvfp4','draft_rag':True,'persistent_gpu_draft_k':True,'cpu_historical_kv':True,'historical_k_storage':'nvfp4','historical_v_storage':'nvfp4','attention_compute':'bf16'}.items(): d[k]=v
 (tmp/f'{cid}.yaml').write_text(yaml.safe_dump(d,sort_keys=False))
PY
mkdir -p "$ROOT/results/group15_corrected_canonical10/logs"
for i in $(seq 1 10); do
 cid=$(printf "case_%02d" "$i"); out="$ROOT/results/group15_corrected_canonical10/$cid"; v="$out/rank0-0-0_lora.mp4"; [ -s "$v" ] && continue
 cfg="$ROOT/results/group15_corrected_canonical10/.runtime_tmp/$cid.yaml"; log="$ROOT/results/group15_corrected_canonical10/logs/$cid.log"; rc=0
 { CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" --config_path "$cfg"; } > "$log" 2>&1 || rc=$?
 if [ "$rc" -ne 0 ] || [ ! -s "$v" ]; then { CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$ROOT" "$PY" -u "$ROOT/inference.py" --config_path "$cfg"; } >> "$log" 2>&1 || rc=$?; fi
 [ -s "$v" ] || { echo "GROUP15_CANONICAL_FAIL $cid" >> "$LOG"; exit 1; }
done
CUDA_VISIBLE_DEVICES=1 "$PY" "$ROOT/scripts/evaluate_group13_15.py" --root "$ROOT/results/group15_corrected_canonical10" --out "$ROOT/results/group15_corrected_canonical10" >> "$LOG" 2>&1
echo "GROUP15_CANONICAL10_DONE $(date -u +%FT%TZ)" >> "$LOG"
