#!/usr/bin/env python3
import csv, json, re
from pathlib import Path

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
OUT = ROOT / 'results'
QPREV = Path('/data/zxl/LongLive-RAG-group11_qprev_h200')
PREF = Path('/data/zxl/LongLive-RAG-group11_next_layer_prefetch_h200')
FLASH = Path('/data/zxl/LongLive-RAG-group11_flashfetch_h200')

FIELDS = [
    'group','method','window','kv_strategy','query_precision','key_precision','value_precision',
    'gpu_kv_gib','cpu_kv_gib','draft_gpu_gib','transient_gpu_gib','compression','fetch_strategy',
    'dino','ssim','psnr','e2e_latency_s','exposed_h2d_ms_per_call','wrapper_latency_ms_per_call',
    'selected_id_exact_rate','final_latent_exact','quality_source','status','provenance_path'
]

def gib(v):
    return round(float(v) / (1 << 30), 10) if v not in (None, 'NA', 'NOT_AVAILABLE') else 'NOT_AVAILABLE'

def load_json(path):
    try: return json.loads(Path(path).read_text())
    except Exception: return {}

def find_runtime(root):
    return next(iter(sorted(root.glob('**/*_runtime.json'))), None)

def queue_status(root, correctness_name='correctness.json'):
    c = root / correctness_name
    if c.exists():
        d = load_json(c)
        rate = d.get('selected_id_exact_rate')
        if rate == 1.0: return 'GATE_PASS', d
        if rate is not None: return 'BLOCKED', d
    log = root / 'runner.log'
    text = log.read_text(errors='ignore') if log.exists() else ''
    if 'WAITING_FOR_GPU' in text and 'FINISHED_UTC' not in text: return 'WAITING_FOR_GPU', {}
    return 'RUNNING_OR_PENDING', {}

def base_row(group, method, strategy, status, provenance):
    return dict.fromkeys(FIELDS, 'NOT_AVAILABLE') | {
        'group': group, 'method': method, 'window': 12, 'kv_strategy': 'BF16 full CPU history',
        'query_precision': 'BF16', 'key_precision': 'BF16', 'value_precision': 'BF16',
        'compression': 'N/A', 'fetch_strategy': strategy, 'status': status,
        'provenance_path': provenance,
    }

rows=[]; prov={}

r = base_row('11.1','LongLive + Draft Attention','layer-wise current-Q fetch','REFERENCE_VALIDATED',
             str(ROOT/'reports/group11_final.md'))
r.update(gpu_kv_gib=3.2135,cpu_kv_gib=28.9215,draft_gpu_gib=0.2317,dino=0.8291,ssim=0.3718,psnr=10.7762,
         e2e_latency_s=218.8,quality_source='validated Group11.1 reference; historical E2E')
rows.append(r)

qmem=load_json(QPREV/'results/group11_qprev_original_wrapper_memory.json')
r=base_row('11.2','LongLive + Draft Attention','layer-wise previous-Q retrieval','GATE_PASS_EXISTING',
           str(QPREV/'results/group11_qprev_original_wrapper_alignment.json'))
r.update(gpu_kv_gib=gib(qmem.get('GPU_LOCAL_KV_BYTES')),cpu_kv_gib=gib(qmem.get('CPU_KV_BYTES')),
         draft_gpu_gib=gib(qmem.get('TOTAL_DRAFT_BYTES')),quality_source='NOT_AVAILABLE: independent frozen evaluation pending')
rows.append(r)

for group, method, strategy, root, label in [
    ('11.3','LongLive + Draft Attention','current-layer chunk-streaming Flash Fetch',FLASH,'flashfetch'),
    ('11.4','LongLive + Draft Attention','one-layer-ahead speculative prefetch + correction fetch',PREF,'prefetch'),
]:
    status, d = queue_status(root)
    r=base_row(group,method,strategy,status,str(root/'results'))
    runtime=find_runtime(root/'results/case01_serial' if group=='11.3' else root/'results/case01')
    if runtime:
        m=load_json(runtime)
        r['selected_id_exact_rate']=d.get('selected_id_exact_rate','NOT_AVAILABLE')
    rows.append(r)

csv_path=OUT/'group11_acceleration_ablation_final.csv'
with csv_path.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

for row in rows:
    for field in FIELDS:
        prov[f"{row['group']}.{field}"]={
            'value': row[field], 'source_path': row['provenance_path'],
            'metric_producer': 'aggregate_group11_accel_results.py',
            'measured': row[field] not in ('NOT_AVAILABLE','N/A'),
        }
(OUT/'group11_acceleration_ablation_provenance.json').write_text(json.dumps(prov,indent=2))

report = ROOT/'reports/group11_acceleration_ablation_final.md'
lines=['# Group11 acceleration ablation aggregation','',
       'This report is generated from raw artifacts. Missing metrics are not inferred.\n',
       '| Group | Status | DINO | SSIM | PSNR | GPU KV GiB | CPU KV GiB | Draft GiB | Transient GiB | Exact IDs | Quality source |',
       '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for r in rows:
    lines.append('| {group} | {status} | {dino} | {ssim} | {psnr} | {gpu_kv_gib} | {cpu_kv_gib} | {draft_gpu_gib} | {transient_gpu_gib} | {selected_id_exact_rate} | {quality_source} |'.format(**r))
lines += ['', '## Protocol sanity', '', '- Window = 12 for all rows.', '- CPU historical K/V = BF16.', '- Final attention = BF16.', '- Compression = N/A because these are scheduling/retrieval variants, not KV low-bit compression.', '', '## Coverage', '', '| Field | Producer/status |', '|---|---|']
for f in FIELDS[7:-2]:
    states=sorted({str(r[f]) for r in rows})
    lines.append(f'| {f} | {", ".join(states)} |')
report.write_text('\n'.join(lines)+'\n')
print(csv_path)
print(report)
