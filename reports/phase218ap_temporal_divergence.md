# Phase 2.18ap temporal divergence

Corresponding-frame comparison covered all 474 frames.

Segment means:
1-100: DINO 0.9967, SSIM 0.9488, PSNR 34.0843, LPIPS 0.0185
101-200: DINO 0.9961, SSIM 0.9323, PSNR inf (identical/zero MSE in some frames), LPIPS 0.0259
201-300: DINO 0.6148, SSIM 0.5810, PSNR 20.9556, LPIPS 0.4121
301-400: DINO 0.9009, SSIM 0.4249, PSNR 13.0128, LPIPS 0.3649
401-474: DINO 0.8872, SSIM 0.3664, PSNR 12.0111, LPIPS 0.4035

The first major sustained decline is in segment 201-300. This demonstrates output divergence, not its cause. The records contain no comparable Attention output/LSE trace, so premature KV reads are NOT_MEASURED as a causal explanation.
