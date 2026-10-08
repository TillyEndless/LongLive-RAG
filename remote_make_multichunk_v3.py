from pathlib import Path
root=Path('/data/zxl/LongLive-RAG-group11_15_h200')
for group, src in [(14,'configs/g14_multichunk_integration_async_overlap_v2.yaml'),(15,'configs/g15_multichunk_integration_async.yaml')]:
    base=(root/src).read_text()
    for mode in ('async_overlap','serial_reference'):
        s=base.replace('output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/group14_multichunk_integration_async_overlap_v2',
                      f'output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/group{group}_multichunk_integration_{mode}_v3')
        s=s.replace('output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/group15_multichunk_integration_async',
                    f'output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/group{group}_multichunk_integration_{mode}_v3')
        s=s.replace('materialization_mode: async_overlap', f'materialization_mode: {mode}')
        (root/f'configs/g{group}_multichunk_integration_{mode}_v3.yaml').write_text(s)

