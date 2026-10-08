#!/usr/bin/env python3
import csv, json
from pathlib import Path

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
SOURCES = {
    '11.1': ROOT / 'results/group11_profile_case01',
    '11.2': Path('/data/zxl/LongLive-RAG-group11_qprev_h200/results/group11_qprev_original_wrapper_case01'),
    '11.3': Path('/data/zxl/LongLive-RAG-group11_flashfetch_h200/results/flashfetch_canonical10'),
    '11.4': Path('/data/zxl/LongLive-RAG-group11_next_layer_prefetch_h200/results/prefetch_canonical10'),
}
OUT = ROOT / 'results/group11_local_dense_kv_re_evaluation.csv'
REPORT = ROOT / 'reports/group11_local_dense_kv_re_evaluation.md'

def read_memory(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def local_dense_bytes(d):
    # New corrected field first. Legacy field is local dense BF16 KV in the
    # historical wrappers, not historical packed storage.
    for k in ('GPU_LOCAL_BF16_KV_BYTES', 'GPU_LOCAL_KV_BYTES'):
        if d.get(k) not in (None, 'N/A', 'NOT_AVAILABLE'):
            return int(d[k]), k
    # Legacy dense/retrieval measurements used GPU_KV_MEASURED_BYTES for the
    # current local window. Keep this fallback explicit and label it legacy.
    if d.get('GPU_KV_MEASURED_BYTES') not in (None, 'N/A', 'NOT_AVAILABLE'):
        return int(d['GPU_KV_MEASURED_BYTES']), 'GPU_KV_MEASURED_BYTES (legacy local dense)'
    return None, 'NOT_AVAILABLE'

rows=[]
for group, root in SOURCES.items():
    files=sorted(root.glob('**/*memory*.json'))
    seen=set()
    for p in files:
        d=read_memory(p)
        b, source_key=local_dense_bytes(d)
        if b is None:
            continue
        case='case_01'
        for part in p.parts:
            if part.startswith('case_'):
                case=part
        key=(group,case)
        if key in seen:
            continue
        seen.add(key)
        rows.append({'group':group,'case':case,
            'GPU_LOCAL_DENSE_KV_BYTES':b,
            'GPU_LOCAL_DENSE_KV_GiB':b/(1<<30),
            'source_key':source_key,'source_path':str(p),
            'CPU_HISTORICAL_PACKED_KV_INCLUDED':'NO',
            'definition':'current local dense BF16 K/V only'})

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', newline='') as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['group','case','GPU_LOCAL_DENSE_KV_BYTES'])
    w.writeheader(); w.writerows(rows)

lines=['# Group11 Local Dense KV Re-evaluation','',
       'This re-evaluation reports only the current local dense BF16 K/V cache.',
       'Historical CPU KV, historical packed KV, and Draft-K memory are excluded.', '',
       '| Group | Cases with measured local dense KV | Local dense KV GiB | Source field |',
       '|---|---:|---:|---|']
for g in ('11.1','11.2','11.3','11.4'):
    rr=[r for r in rows if r['group']==g]
    vals=sorted({round(float(r['GPU_LOCAL_DENSE_KV_GiB']),10) for r in rr})
    keys=sorted({r['source_key'] for r in rr})
    lines.append(f"| {g} | {len(rr)} | {', '.join(map(str, vals)) or 'NOT_AVAILABLE'} | {', '.join(keys) or 'NOT_AVAILABLE'} |")
lines += ['', '## Field definition', '',
          '`GPU_LOCAL_DENSE_KV_GiB` = BF16 K/V tensors belonging to the current local attention window.',
          'It is not the historical packed KV owner and does not include CPU historical KV.', '',
          'Original memory JSON files were not modified.']
REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text('\n'.join(lines)+'\n')
print(OUT)
print(REPORT)
