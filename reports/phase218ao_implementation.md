# Phase 2.18ao implementation

Worktree: /data/zxl/LongLive-RAG-group11_1-w1-phase218ao
HEAD: 410948eb7e328513e7751ac2d44193115363cee4

Target: W20/R16/S1/F120, retrieval_backend=draftmap_online, group11_fetch_mode=w1_zero_acquire.

Classification: W1_STREAMING_UNSYNCHRONIZED_EXPLORATORY. One contiguous final GPU K/V buffer, independent fetch stream, one warmup CUDA event, remaining reverse-order submissions without a pre-attention full-producer wait, and post-attention join. Stock wan.modules.attention.attention is used; no A1/A2 gate, Resolver, Pointer Table, or new Attention kernel.

Source: wan/modules/causal_model_latentmem.py, helper near line 430; warmup near 488-491; remaining submissions near 493-498; stock attention near 1082; final join near 1083-1087.
