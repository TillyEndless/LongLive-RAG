#!/usr/bin/env python3
import csv, json, hashlib
from pathlib import Path
import cv2
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from transformers import AutoImageProcessor, AutoModel

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
OUT = ROOT / 'results/group14_15_corrected_sparse'
manifest = json.loads((OUT / 'corrected_sparse_manifest.json').read_text())
processor = AutoImageProcessor.from_pretrained('/data/zxl/eval_models/dinov2-small')
model = AutoModel.from_pretrained('/data/zxl/eval_models/dinov2-small').to('cuda').eval()

def frame(video, idx):
    cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, img = cap.read(); cap.release()
    if not ok: raise RuntimeError(f'cannot read frame {idx}: {video}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@torch.inference_mode()
def dino(a, b):
    x = processor(images=[Image.fromarray(a), Image.fromarray(b)], return_tensors='pt')
    x = {k: v.to('cuda') for k, v in x.items()}
    f = model(**x).last_hidden_state[:, 0]
    f = torch.nn.functional.normalize(f, dim=-1)
    return float((f[0] * f[1]).sum().item())

def find_video(outdir):
    paths = sorted(p for p in outdir.glob('*.mp4') if not p.name.endswith('.tmp.mp4'))
    return paths[0] if paths else None

def load_json(path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None

def validate_runtime_contract(video, job):
    """Reject stale/transient-fake-quant outputs before metric aggregation."""
    runtime = load_json(video.with_name(video.stem + '_runtime.json'))
    memory = load_json(video.with_name(video.stem + '_memory_measurement.json'))
    if not isinstance(runtime, dict) or not isinstance(memory, dict):
        return False, 'MISSING_RUNTIME_OR_MEMORY_METADATA', runtime, memory
    group = int(job['group'])
    mode = {12: 'group12_corrected', 13: 'group13_corrected',
            14: 'group14_corrected', 15: 'group15_corrected'}[group]
    expected_owner = 'INT8/FP8_E4M3' if group in {12, 14} else 'NVFP4/NVFP4'
    checks = {
        'GROUP_RUNTIME_MODE': mode,
        'CPU_COMPRESSED_HISTORY_ACTIVE': 'NO',
        'CPU_HISTORICAL_BF16_ARCHIVE_ACTIVE': 'YES',
        'PERSISTENT_STORAGE_MODE': 'LOWBIT_STORAGE_BF16_COMPUTE',
        'GPU_PERSISTENT_KV_OWNER': expected_owner,
        'FINAL_ATTENTION_DTYPE': 'bfloat16',
        'NATIVE_LOWBIT_KERNEL_USED': 'NO',
        'Q_PREV': 'NO',
        'FLASH_FETCH': 'NO',
        'NEXT_LAYER_PREFETCH': 'NO',
        'HOT_CACHE': 'NO',
        'LONG_LIVE_REUSE': 'NO',
    }
    bad = [f'{k}={runtime.get(k)!r}, expected {v!r}' for k, v in checks.items() if runtime.get(k) != v]
    if bad:
        return False, 'CONTRACT_MISMATCH: ' + '; '.join(bad), runtime, memory
    return True, 'OK', runtime, memory

def collect_numbers(outdir):
    vals = {}
    for p in sorted(outdir.rglob('*.json')):
        try: obj=json.loads(p.read_text())
        except Exception: continue
        stack=[obj]
        while stack:
            x=stack.pop()
            if isinstance(x, dict):
                for k,v in x.items():
                    if isinstance(v,(int,float)) and any(s in k.lower() for s in ('gpu_kv','cpu_kv','draft','latency','memory','kv_bytes','compression')):
                        vals.setdefault(k,v)
                    elif isinstance(v,(dict,list)): stack.append(v)
            elif isinstance(x,list): stack.extend(x)
    return vals

rows=[]
for job in manifest['jobs']:
    label, case = job['label'], job['case']
    video = find_video(Path(job['output']))
    row={'label':label,'group':job['group'],'case':case,'retained_ratio':job.get('sparse_ratio','NA'),
         'video':str(video) if video else 'NOT_AVAILABLE','first_frame':0,'return_frame':237,
         'DINO':'NOT_AVAILABLE','SSIM':'NOT_AVAILABLE','PSNR':'NOT_AVAILABLE','LPIPS':'NOT_AVAILABLE',
         'status':'NOT_AVAILABLE'}
    runtime_meta = None
    memory_meta = None
    if video:
        contract_ok, contract_reason, runtime_meta, memory_meta = validate_runtime_contract(video, job)
        row['contract_status'] = contract_reason
        if not contract_ok:
            row['status'] = contract_reason
            row.update({f'meta_{k}':v for k,v in (runtime_meta or {}).items() if isinstance(v,(int,float,str))})
            rows.append(row)
            continue
        cap=cv2.VideoCapture(str(video)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=float(cap.get(cv2.CAP_PROP_FPS) or 0)
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release()
        row.update({'frames':n,'fps':fps,'width':w,'height':h})
        if n==474 and abs(fps-16)<.01 and (w,h)==(832,480):
            a,b=frame(video,0),frame(video,237)
            row['DINO']=dino(a,b); row['SSIM']=float(structural_similarity(a,b,channel_axis=2,data_range=255)); row['PSNR']=float(peak_signal_noise_ratio(a,b,data_range=255)); row['status']='OK'
        else: row['status']='PROTOCOL_MISMATCH'
    row.update({f'meta_{k}':v for k,v in collect_numbers(Path(job['output'])).items()})
    rows.append(row)

# Hard gate: every canonical experiment must have exactly the same ten cases
# as the manifest before any aggregate summary is considered valid.  In
# particular, never silently reduce Group14/15 to case_01-only evaluation.
expected = {}
for job in manifest['jobs']:
    expected.setdefault(job['label'], set()).add(job['case'])
bad=[]
for label, cases in expected.items():
    actual=[r for r in rows if r['label']==label]
    actual_cases={r['case'] for r in actual}
    if len(cases) != 10 or actual_cases != cases or len(actual) != 10:
        bad.append(f'{label}: expected_cases={sorted(cases)}, actual_cases={sorted(actual_cases)}, rows={len(actual)}')
    invalid=[r['case'] for r in actual if r.get('status')!='OK']
    if invalid:
        bad.append(f'{label}: invalid_cases={sorted(invalid)}')
if bad:
    raise RuntimeError('CANONICAL10_EVALUATION_GATE_FAILED; no aggregate summary written: ' + ' | '.join(bad))

with (OUT/'group14_15_corrected_per_case.csv').open('w',newline='') as f:
    keys=sorted({k for r in rows for k in r}); w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)

def mean(label,key):
    x=[float(r[key]) for r in rows if r['label']==label and r.get('status')=='OK' and r.get(key) not in (None,'NOT_AVAILABLE')]
    return sum(x)/len(x) if x else 'NOT_AVAILABLE'
summary=[]
for label in sorted({r['label'] for r in rows}):
    group=next(r['group'] for r in rows if r['label']==label)
    ratio=next((r['retained_ratio'] for r in rows if r['label']==label),'NA')
    summary.append({'label':label,'group':group,'window':12,'retained_ratio':ratio,'cases':sum(r['label']==label for r in rows),'valid_cases':sum(r['label']==label and r['status']=='OK' for r in rows),'DINO':mean(label,'DINO'),'SSIM':mean(label,'SSIM'),'PSNR':mean(label,'PSNR'),'LPIPS':'NOT_AVAILABLE'})
with (OUT/'group14_15_corrected_10case.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
(OUT/'group14_15_corrected_provenance.json').write_text(json.dumps({'manifest_sha256':hashlib.sha256(Path('/data/zxl/LongLive-RAG-profile/prompts10.txt').read_bytes()).hexdigest(),'seed':0,'protocol':'frame 0 vs frame 237; 474 frames; 16 FPS; 832x480; DINOv2-small CLS; raw RGB SSIM/PSNR; LPIPS NOT_AVAILABLE','backend':'BF16 final attention; fake quant only; no native Anemoi kernel; corrected sparsity semantics'},indent=2)+'\n')
print('WROTE',OUT/'group14_15_corrected_10case.csv')
