#!/usr/bin/env python3
import csv, json, re
from pathlib import Path

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
SEARCH_ROOTS = [
    ROOT / 'results/group12_15_persistent_campaign',
    ROOT / 'results/group12_15_corrected_campaign',
    ROOT / 'results/group13_corrected_canonical10',
    ROOT / 'results/group14_15_canonical30_run',
]
OUT = ROOT / 'results/group12_15_local_dense_kv_re_evaluation.csv'
REPORT = ROOT / 'reports/group12_15_local_dense_kv_re_evaluation.md'

def memory(path):
    try: return json.loads(path.read_text())
    except Exception: return {}

def local_dense(d):
    for k in ('GPU_LOCAL_BF16_KV_BYTES', 'GPU_LOCAL_KV_BYTES', 'GPU_KV_MEASURED_BYTES'):
        if d.get(k) not in (None, 'N/A', 'NOT_AVAILABLE'):
            return int(d[k]), k
    return None, 'NOT_AVAILABLE'

rows=[]
for root in SEARCH_ROOTS:
    for p in sorted(root.glob('**/*memory_measurement.json')):
        d=memory(p); b,key=local_dense(d)
        if b is None: continue
        rel=p.relative_to(root).parts
        group='NOT_AVAILABLE'; case='NOT_AVAILABLE'; variant=root.name
        for part in rel:
            m=re.fullmatch(r'(group1[2-5](?:_sparse\d+)?)', part)
            if m: group=m.group(1)
            if re.fullmatch(r'case\d+', part): case=part
        if group=='NOT_AVAILABLE':
            # corrected_canonical10 has group13 as root-level experiment
            if 'group13_corrected' in str(root): group='group13'
        rows.append({'group':group,'case':case,'experiment_root':variant,
            'GPU_LOCAL_DENSE_KV_BYTES':b,'GPU_LOCAL_DENSE_KV_GiB':b/(1<<30),
            'source_key':key,'source_path':str(p),
            'historical_packed_KV_included':'NO',
            'definition':'current local dense BF16 K/V only'})

OUT.parent.mkdir(parents=True,exist_ok=True)
fields=list(rows[0]) if rows else ['group','case','experiment_root','GPU_LOCAL_DENSE_KV_BYTES']
with OUT.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)

lines=['# Group12–15 Local Dense KV Re-evaluation','',
       'Only current local dense BF16 K/V is reported.',
       'Historical packed KV, CPU historical KV, Draft-K, and transient dequant buffers are excluded.', '',
       '| Group/variant | Cases with artifacts | Local dense KV GiB |',
       '|---|---:|---:|']
keys=sorted({(r['group'],r['experiment_root']) for r in rows})
for g,v in keys:
    rr=[r for r in rows if r['group']==g and r['experiment_root']==v]
    vals=sorted({round(float(r['GPU_LOCAL_DENSE_KV_GiB']),10) for r in rr})
    lines.append(f"| {g} / {v} | {len(rr)} | {', '.join(map(str,vals))} |")
lines += ['', 'The original memory JSON files were not modified.',
          'Legacy `GPU_KV_MEASURED_BYTES` is used only when explicit local-dense fields are absent, and is labeled in the CSV.']
REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text('\n'.join(lines)+'\n')
print(OUT); print(REPORT); print('rows=',len(rows))
