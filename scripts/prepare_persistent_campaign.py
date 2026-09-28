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
ACTIVE_REMOVED = (30, 50, 70, 40)
sources = []
sources.extend(sorted(OLD_CFG.glob('group12_case*.yaml')))
sources.extend(sorted(OLD_CFG.glob('group13_case*.yaml')))
for group in (14, 15):
    for removed in ACTIVE_REMOVED:
        # The sparse30 file is the canonical template for the group; only
        # the generated label and retained ratio change for the new sweep.
        sources.extend(OLD_CFG.glob(f'group{group}_sparse30_case*.yaml'))
        break
for p in sorted(sources):
    text = p.read_text()
    base_label = p.stem.rsplit('_case',1)[0]
    case = 'case' + p.stem.rsplit('_case',1)[1]
    removed_values = ACTIVE_REMOVED if base_label.startswith(('group14_sparse', 'group15_sparse')) else (None,)
    for removed in removed_values:
        label = base_label if removed is None else base_label.rsplit('_sparse', 1)[0] + f'_sparse{removed:02d}'
        old_out = next(line.split(': ',1)[1].strip() for line in text.splitlines() if line.startswith('output_folder:'))
        new_out = OUT / label / case
        snapshot = text.replace(f'output_folder: {old_out}', f'output_folder: {new_out}')
        sparse_ratio = None if removed is None else 1.0 - removed / 100.0
        # The runtime consumes group_sparse_ratio as the retained interaction
        # fraction. Normalize every active snapshot explicitly.
        if removed is not None:
            snapshot = re.sub(
                r'(?m)^(\s*group_sparse_ratio:\s*)[^\n]+$',
                rf'\g<1>{sparse_ratio:.2f}', snapshot)
            snapshot = re.sub(
                r'(?m)^ratio_semantics:\s*[^\n]+$',
                'ratio_semantics: retained_interaction_ratio', snapshot)
        snapshot = snapshot.replace('num_output_frames: 474', 'num_output_frames: 120')
        snapshot += '\nretrieval_query_mode: current_q\nq_prev: false\nflash_fetch: false\nnext_layer_prefetch: false\nhot_cache: false\nlonglive_reuse: false\n\n'
        dst = CFG / f'{label}_case{p.stem.rsplit("_case",1)[1]}.yaml'; dst.write_text(snapshot)
        try: group = int(label.split('_')[0].replace('group',''))
        except ValueError: continue
        manifest['jobs'].append({'group':group,'case':case,'label':label,'sparse_ratio':sparse_ratio,'config':str(dst),'output':str(new_out)})
(OUT/'campaign_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'jobs':len(manifest['jobs']),'labels':sorted({j['label'] for j in manifest['jobs']}),'root':str(OUT)},indent=2))
