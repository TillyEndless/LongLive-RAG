# Phase 2.18ap unified evaluation

Evaluator: /data/zxl/DraftMap-RAG/scripts/evaluate_canonical3_case.py
Evaluator SHA256: 85e642015bb185009a7f68648735ead32dd3c85485a66c8c1de301c9f66e10e5
Protocol: 474 frames, 16 FPS, 832x480, internal frame 0 vs frame 237. LPIPS used local AlexNet weights.

Both videos passed decode/protocol checks and the same evaluator. No ground-truth reference was available, so these are not GT quality scores.

Official paired rows:
- Baseline: DINO 0.8445296288, SSIM 0.1743881702, PSNR 9.0320051550, LPIPS 0.5627127886.
- W1: DINO 0.8707666993, SSIM 0.1732239530, PSNR 8.9870800390, LPIPS 0.5634891391.

W1 remains unsynchronized exploratory; these metrics do not prove synchronization correctness.
