#!/usr/bin/env python3
import csv
import json
import os
import subprocess
import time
from pathlib import Path
import cv2

ROOT = Path('/data/zxl/LongLive-RAG-group11_15_h200')
PYTHON = '/data/zxl/SolarWM_wan5b_direct/.conda-wan/bin/python'
EVAL_PYTHON = '/data/zxl/SolarWM/.env_vbench310/bin/python'
EVAL_SCRIPT = ROOT / 'scripts/evaluate_group_sparse_case.py'
OUT = ROOT / 'results'
STATE = OUT / 'group14_15_sparse_screening.state.json'
CSV_PATH = OUT / 'group14_15_sparse_screening.csv'
REPORT = ROOT / 'reports/group14_15_sparse_screening.md'
rows = []

if CSV_PATH.exists():
    with CSV_PATH.open() as handle:
        rows = list(csv.DictReader(handle))


def write_state(stage, group=None, ratio=None):
    STATE.write_text(json.dumps({
        'stage': stage,
        'group': group,
        'retained_ratio': ratio,
        'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'completed_rows': len(rows),
    }, indent=2) + '\n')


def save_row(row):
    global rows
    rows = [item for item in rows if not (
        item.get('group') == str(row['group'])
        and item.get('retained_ratio') == str(row['retained_ratio'])
    )]
    rows.append({key: str(value) for key, value in row.items()})
    fields = [
        'group', 'retained_ratio', 'status', 'seed', 'frames',
        'CPU_KV_BYTES', 'GPU_KV_BYTES', 'GPU_DRAFT_BYTES',
        'CPU_DRAFT_BYTES', 'full_history_bf16_shadow_bytes',
        'DRAFT_H2D_BYTES', 'effective_retained_ratio', 'DINO', 'SSIM',
        'PSNR', 'LPIPS', 'runtime_s', 'output_path',
    ]
    with CSV_PATH.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in rows:
            writer.writerow({key: item.get(key, 'NOT_AVAILABLE') for key in fields})


def write_report():
    lines = [
        '# Group14/15 sparse screening',
        '',
        'GPU0 only; case01; W12; seed=0. retained_ratio means the fraction '
        'of routed interactions retained and computed in BF16. Skipped '
        'interactions are not computed. Order: 30%, 20%, 10%, 5%.',
        '',
        '| group | retained ratio | status | actual retained | DINO | SSIM | PSNR |',
        '|---:|---:|---|---:|---:|---:|---:|',
    ]
    for group in (14, 15):
        for percent in (30, 20, 10, 5):
            matches = [
                item for item in rows
                if item.get('group') == str(group)
                and item.get('retained_ratio') == str(percent)
            ]
            if matches:
                item = matches[0]
                lines.append(
                    '| %s | %s%% | %s | %s | %s | %s | %s |' % (
                        group, percent, item.get('status'),
                        item.get('effective_retained_ratio', 'NA'),
                        item.get('DINO', 'NA'), item.get('SSIM', 'NA'),
                        item.get('PSNR', 'NA'),
                    )
                )
            else:
                lines.append('| %s | %s%% | PENDING | — | — | — | — |' % (group, percent))
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text('\n'.join(lines) + '\n')


for group in (14, 15):
    for percent in (30, 20, 10, 5):
        ratio = percent / 100.0
        tag = f'group{group}_case01_retained{percent:02d}'
        config = ROOT / 'configs' / f'{tag}.yaml'
        output = OUT / tag
        video = output / 'rank0-0-0_lora.mp4'
        write_state('running', group, percent)

        if not video.exists():
            output.mkdir(parents=True, exist_ok=True)
            with (output / 'inference.log').open('w') as log:
                subprocess.run(['nvidia-smi'], stdout=log, stderr=subprocess.STDOUT, check=False)
                started = time.monotonic()
                env = os.environ.copy()
                env['CUDA_VISIBLE_DEVICES'] = '0'
                env['PYTHONPATH'] = str(ROOT)
                completed = subprocess.run(
                    [PYTHON, '-u', str(ROOT / 'inference.py'), '--config_path', str(config)],
                    env=env, stdout=log, stderr=subprocess.STDOUT,
                )
                elapsed = time.monotonic() - started
            if completed.returncode != 0 or not video.exists():
                save_row({
                    'group': group, 'retained_ratio': percent,
                    'status': 'FAILED_INFERENCE', 'seed': 0,
                    'output_path': output, 'runtime_s': f'{elapsed:.3f}',
                })
                write_report()
                write_state('failed', group, percent)
                break

        cap = cv2.VideoCapture(str(video))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        memories = list(output.glob('*memory_measurement.json'))
        runtimes = list(output.glob('*runtime.json'))
        if not memories or not runtimes or frames <= 0:
            save_row({
                'group': group, 'retained_ratio': percent,
                'status': 'INVALID_METADATA', 'seed': 0, 'frames': frames,
                'output_path': output,
            })
            write_report()
            write_state('failed', group, percent)
            break

        memory = json.loads(memories[0].read_text())
        runtime = json.loads(runtimes[0].read_text())
        actual = runtime.get('ACTUAL_BF16_FRACTION')
        if actual is None and runtime.get('runtime_trace'):
            actual = runtime['runtime_trace'][-1].get('ACTUAL_BF16_FRACTION')
        if actual is None and 'RETAINED_INTERACTIONS' in runtime:
            actual = float(runtime['RETAINED_INTERACTIONS']) / max(float(runtime.get('TOTAL_INTERACTIONS', 1)), 1)
        try:
            valid = abs(float(actual) - ratio) <= 0.05
        except (TypeError, ValueError):
            valid = False
        if not valid:
            save_row({
                'group': group, 'retained_ratio': percent,
                'status': 'INVALID_RATIO', 'seed': 0, 'frames': frames,
                'effective_retained_ratio': actual, 'output_path': output,
            })
            write_report()
            write_state('failed', group, percent)
            break

        quality = output / 'quality.json'
        subprocess.run([
            EVAL_PYTHON, '-u', str(EVAL_SCRIPT), '--video', str(video),
            '--output', str(quality), '--group', str(group), '--ratio', str(percent),
        ], env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0'}, check=True)
        metrics = json.loads(quality.read_text())
        save_row({
            'group': group, 'retained_ratio': percent, 'status': 'COMPLETED',
            'seed': 0, 'frames': frames,
            'CPU_KV_BYTES': memory.get('CPU_KV_TOTAL_BYTES', 'NOT_AVAILABLE'),
            'GPU_KV_BYTES': memory.get('GPU_KV_MEASURED_BYTES', 'NOT_AVAILABLE'),
            'GPU_DRAFT_BYTES': memory.get('GPU_DRAFT_PERSISTENT_BYTES', 'NOT_AVAILABLE'),
            'CPU_DRAFT_BYTES': memory.get('CPU_DRAFT_PERSISTENT_BYTES', 'NOT_AVAILABLE'),
            'full_history_bf16_shadow_bytes': memory.get('FULL_HISTORY_BF16_SHADOW_BYTES', 'NOT_AVAILABLE'),
            'DRAFT_H2D_BYTES': memory.get('DRAFT_H2D_BYTES', 'NOT_AVAILABLE'),
            'effective_retained_ratio': actual,
            'DINO': metrics['DINO'], 'SSIM': metrics['SSIM'], 'PSNR': metrics['PSNR'],
            'LPIPS': 'NOT_AVAILABLE', 'runtime_s': 'RECORDED_IN_INFERENCE_LOG',
            'output_path': output,
        })
        write_report()
        write_state('completed_case', group, percent)

write_report()
write_state('finished')
print('SCREENING_FINISHED')
