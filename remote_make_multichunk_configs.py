from pathlib import Path
root = Path('/data/zxl/LongLive-RAG-group11_15_h200')
for group, src, mode, ks, vs in [
    (14, 'configs/g14_currentq_int8_fp8_promotion.yaml', 'group14_corrected', 'int8', 'fp8_e4m3'),
    (15, 'configs/g15_currentq_nvfp4_promotion.yaml', 'group15_corrected', 'nvfp4', 'nvfp4'),
]:
    text = (root / src).read_text()
    text = text.replace('kv_promotion_ratio: 0.2', 'kv_promotion_ratio: 0.5')
    text = text.replace('data_path: /data/zxl/LongLive-RAG-profile/prompts10.txt',
                        'data_path: /data/zxl/strict_latency_case01_20260928/case_01.txt')
    text = text.replace('num_output_frames: 120', 'num_output_frames: 120')
    text = text.replace('output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/group14_promotion_contract_smoke',
                        'output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/group14_multichunk_integration_async')
    text = text.replace('output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/group15_promotion_contract_smoke',
                        'output_folder: /data/zxl/LongLive-RAG-group11_15_h200/results/group15_multichunk_integration_async')
    if group == 15:
        text = text.replace('group14_multichunk_integration_async', 'group15_multichunk_integration_async')
    (root / f'configs/g{group}_multichunk_integration_async.yaml').write_text(text)
    serial = text.replace('materialization_mode: async_overlap', 'materialization_mode: serial_reference')
    serial = serial.replace('multichunk_integration_async', 'multichunk_integration_serial')
    (root / f'configs/g{group}_multichunk_integration_serial.yaml').write_text(serial)

