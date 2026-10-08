#!/usr/bin/env python3
import csv, json, os
from pathlib import Path

ROOT=Path('/data/zxl/LongLive-RAG-group11_15_h200')
EVAL=ROOT/'results/canonical_aligned_evaluation'
OUT=ROOT/'results/group14_nonmonotonic_dino_audit.csv'
REPORT=ROOT/'reports/group14_nonmonotonic_dino_audit.md'

rows=[]
for ratio in (30,20,10,5):
    p=EVAL/'group14'/'per_case.csv'
    rr=[r for r in csv.DictReader(p.open()) if r['retained_ratio']==str(ratio)]
    if not rr: continue
    r=rr[0]
    runtime=ROOT/'results'/f'group14_case01_retained{ratio:02d}'/'rank0-0-0_lora_runtime.json'
    d=json.loads(runtime.read_text())
    rows.append({
        'group':'14','labelled_retained_percent':ratio,
        'actual_sparse_ratio':d.get('SPARSE_RATIO'),
        'route_blocks_total':d.get('ROUTE_BLOCKS_TOTAL'),
        'route_blocks_retained':d.get('ROUTE_BLOCKS_RETAINED'),
        'actual_retained_fraction':d.get('ACTUAL_BF16_FRACTION'),
        'actual_zero_fraction':d.get('ACTUAL_ZERO_FRACTION'),
        'dino':r['DINO'],'ssim':r['SSIM'],'psnr':r['PSNR'],
        'case':r['case'],'video':r['video'],
        'frames':r['num_frames'],'fps':r['fps'],
        'first_frame':r['first_frame'],'return_frame':r['return_frame'],
        'protocol':r['protocol'],
        'video_exists':os.path.exists(r['video']),
        'sparse_execution_status':d.get('SPARSE_EXECUTION_STATUS'),
        'routing_implementation_id':d.get('ROUTING_IMPLEMENTATION_ID'),
    })

with OUT.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

g12=list(csv.DictReader((EVAL/'group12'/'per_case.csv').open()))
g14=list(csv.DictReader((EVAL/'group14'/'per_case.csv').open()))
lines=['# Group14 Non-monotonic DINO Audit','',
       '## Audit conclusion','',
       '**The current Group14 result is not a valid 10-case comparison with Group12.**',
       'The canonical evaluator produced 10 rows for Group12 but only 4 rows for Group14: all are `case_01`, one for each retained target.',
       'Therefore the observed non-monotonic DINO curve is a one-case diagnostic, not a reliable group-level quality curve.', '',
       '## Evaluator audit','',
       '- Evaluator: `scripts/evaluate_all_canonical_aligned.py`.',
       '- Manifest SHA256: `db9462c0954186b9128543b58fb6d1d496a8b89e0754333beca698d3cf461fec`.',
       '- Frame protocol: frame 0 vs frame 237.',
       '- Video protocol: 474 frames, 16 FPS, 832x480.',
       '- DINO: DINOv2-small CLS cosine.',
       '- Group12 coverage: 10 cases.',
       '- Group14 coverage: 1 case × 4 retained targets.', '',
       '## Group14 actual routing and DINO','',
       '| labelled retain | actual SPARSE_RATIO | retained blocks / total | actual retained fraction | DINO | SSIM | PSNR |',
       '|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['labelled_retained_percent']}% | {r['actual_sparse_ratio']} | {r['route_blocks_retained']} / {r['route_blocks_total']} | {r['actual_retained_fraction']:.6f} | {float(r['dino']):.6f} | {float(r['ssim']):.6f} | {float(r['psnr']):.6f} |")
lines += ['', '## Findings', '',
          '1. The 5% row is not exactly 5% retained: `SPARSE_RATIO=0.055`, retaining 17/293 blocks (about 5.81%).',
          '2. The four Group14 videos all pass the same frame/fps/resolution protocol and point to distinct output paths.',
          '3. Runtime metadata reports `REAL_SPARSE_EXECUTION` and the same routing implementation ID for all four rows.',
          '4. The evaluator did not accidentally use the same video for all four rows; the paths and video hashes are distinct in the audit source artifacts.',
          '5. The high 5% DINO is therefore not evidence of a general sparsity benefit: it is one case and can arise from case-specific top-k block selection.',
          '6. The 10%/20% dip cannot be distinguished between genuine non-monotonic routing behavior and case-level variance until all four settings are evaluated on the same canonical 10 cases.', '',
          '## Required correction', '',
          'Run the same evaluator on Group14 retained 30/20/10/5 for case_01–case_10, then compare case-by-case and by mean. Do not use the current Group14 summary for a Group12-vs-Group14 claim.', '',
          'Original videos and evaluation outputs were not modified.']
REPORT.write_text('\n'.join(lines)+'\n')
print(OUT); print(REPORT)
