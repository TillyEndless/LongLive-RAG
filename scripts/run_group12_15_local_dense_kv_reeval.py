#!/usr/bin/env python3
import csv, json, re
from pathlib import Path

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
BASE = ROOT / 'results/group12_15_persistent_campaign'
OUT = ROOT / 'results/group12_15_local_dense_kv_re_evaluation.csv'
REPORT = ROOT / 'reports/group12_15_local_dense_kv_re_evaluation.md'

def load(p):
    try: return json.loads(p.read_text())
    except Exception: return {}

def local_dense(d):
    for k in ('GPU_LOCAL_BF16_KV_BYTES', 'GPU_LOCAL_KV_BYTES', 'GPU_KV_MEASURED_BYTES'):
        if d.get(k) not in (None, 'N/A', 'NOT_AVAILABLE'):
            return int(d[k]), k
    return None, 'NOT_AVAILABLE'

def runtime_value(d):
    for k in ('E2E_LATENCY_S','e2e_latency_s','LATENCY_S','latency_s','TOTAL_LATENCY_S'):
        if isinstance(d.get(k), (int,float)):
            return d[k], k
    return 'NOT_AVAILABLE', 'NOT_AVAILABLE'

rows=[]
for mem in sorted(BASE.glob('**/*memory_measurement.json')):
    d=load(mem); b, key=local_dense(d)
    if b is None: continue
    rel=mem.relative_to(BASE)
    parts=rel.parts
    variant=parts[0] if parts else 'unknown'
    group_match=re.search(r'group(1[2-5])', variant)
    group=group_match.group(1) if group_match else 'unknown'
    case=next((x for x in parts if x.startswith('case')), 'unknown')
    runtime_file=mem.with_name(mem.name.replace('_memory_measurement.json','_runtime.json'))
    rt,rtkey=runtime_value(load(runtime_file)) if runtime_file.exists() else ('NOT_AVAILABLE','NOT_AVAILABLE')
    rows.append({'group':group,'variant':variant,'case':case,
        'GPU_LOCAL_DENSE_KV_BYTES':b,'GPU_LOCAL_DENSE_KV_GiB':b/(1<<30),
        'running_time_s':rt,'running_time_source':rtkey,
        'source_key':key,'memory_source':str(mem),
        'CPU_HISTORICAL_PACKED_KV_INCLUDED':'NO',
        'definition':'current local dense BF16 K/V only'})

with OUT.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['group','variant','case','GPU_LOCAL_DENSE_KV_BYTES'])
    w.writeheader(); w.writerows(rows)

lines=['# Group12–15 Local Dense KV Re-evaluation','',
       'This is a post-processing evaluation of existing outputs; no inference was run.',
       'GPU KV means only current local dense BF16 K/V. Historical packed KV, CPU history, Draft-K, and transient dequant buffers are excluded.', '',
       '| Group | Variant | Cases | Local dense KV GiB | Running time |',
       '|---|---|---:|---:|---|']
for g in ('12','13','14','15'):
    rr=[r for r in rows if r['group']==g]
    vals=sorted({round(float(r['GPU_LOCAL_DENSE_KV_GiB']),10) for r in rr})
    times=sorted({str(r['running_time_s']) for r in rr if r['running_time_s']!='NOT_AVAILABLE'})
    variants=sorted({r['variant'] for r in rr})
    lines.append(f"| {g} | {', '.join(variants) or 'NOT_AVAILABLE'} | {len(rr)} | {', '.join(map(str,vals)) or 'NOT_AVAILABLE'} | {', '.join(times) or 'NOT_AVAILABLE'} |")
lines += ['', '## Runtime note', '',
          'Existing runtime JSON files contain execution metadata but no scalar end-to-end latency field; therefore `running_time_s` is NOT_AVAILABLE where absent.',
          'No latency was inferred from file timestamps.', '',
          'Original outputs were not modified.']
REPORT.write_text('\n'.join(lines)+'\n')
print(OUT)
print(REPORT)
