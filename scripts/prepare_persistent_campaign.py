#!/usr/bin/env python3
import json, hashlib, re
from pathlib import Path

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
OLD_CFG = ROOT / 'configs/group12_15_corrected_campaign'
CFG = ROOT / 'configs/group12_15_persistent_campaign'
OUT = ROOT / 'results/group12_15_persistent_campaign'
PROMPTS = Path('/data/zxl/LongLive-RAG-profile/prompts10.txt')
CFG.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
manifest = {'manifest': str(PROMPTS), 'sha256': hashlib.sha256(PROMPTS.read_bytes()).hexdigest(), 'seed': 0, 'jobs': []}
for p in sorted(OLD_CFG.glob('group*_case*.yaml')):
    text = p.read_text()
    label = p.stem.rsplit('_case',1)[0]
    case = 'case' + p.stem.rsplit('_case',1)[1]
    old_out = next(line.split(': ',1)[1].strip() for line in text.splitlines() if line.startswith('output_folder:'))
    new_out = OUT / label / case
    text = text.replace(f'output_folder: {old_out}', f'output_folder: {new_out}')
    # The runtime consumes group_sparse_ratio as the retained interaction
    # fraction.  Campaign labels encode the user-facing removal fraction.
    # Normalize the snapshot here so legacy source configs cannot silently
    # invert the corrected Group14/15 semantics.
    sparse_ratio = None
    if 'sparse' in label:
        removed = int(label.rsplit('sparse', 1)[1])
        sparse_ratio = 1.0 - removed / 100.0
        text = re.sub(
            r'(?m)^(\s*group_sparse_ratio:\s*)[^\n]+$',
            rf'\g<1>{sparse_ratio:.2f}', text)
        text = re.sub(
            r'(?m)^ratio_semantics:\s*[^\n]+$',
            'ratio_semantics: retained_interaction_ratio', text)
    # Wan config counts latent frames; 120 latent frames decode to the frozen
    # 474-frame video protocol. Do not put decoded frame count in this field.
    text = text.replace('num_output_frames: 474', 'num_output_frames: 120')
    # Explicitly persist the corrected owner contract in every snapshot.
    text += '\nretrieval_query_mode: current_q\nq_prev: false\nflash_fetch: false\nnext_layer_prefetch: false\nhot_cache: false\nlonglive_reuse: false\n\n'
    dst = CFG / p.name; dst.write_text(text)
    try: group = int(label.split('_')[0].replace('group',''))
    except ValueError: continue
    ratio = sparse_ratio
    manifest['jobs'].append({'group':group,'case':case,'label':label,'sparse_ratio':ratio,'config':str(dst),'output':str(new_out)})
(OUT/'campaign_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'jobs':len(manifest['jobs']),'labels':sorted({j['label'] for j in manifest['jobs']}),'root':str(OUT)},indent=2))
