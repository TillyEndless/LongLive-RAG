# Phase 2.18ao real inference

Baseline: results/baseline_smoke/case_01; E2E 119.1131180562079 s; transformer 106.3547869194299 s.
Corrected W1: results/w1_corrected/case_01; E2E 113.39737264066935 s; transformer 100.50624381937087 s.
Both videos decode to 474 frames at 832x480 and 16 FPS.

W1 completed without CUDA error but is unsynchronized exploratory code. Numerical parity, quality metrics, and Nsight overlap were NOT RUN.
