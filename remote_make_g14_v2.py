from pathlib import Path
root=Path('/data/zxl/LongLive-RAG-group11_15_h200')
for mode in ('async_overlap','serial_reference'):
    s=(root/'configs/g14_multichunk_integration_async.yaml').read_text()
    s=s.replace('output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/group14_multichunk_integration_async',
                f'output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/group14_multichunk_integration_{mode}_v2')
    s=s.replace('materialization_mode: async_overlap', f'materialization_mode: {mode}')
    (root/f'configs/g14_multichunk_integration_{mode}_v2.yaml').write_text(s)

