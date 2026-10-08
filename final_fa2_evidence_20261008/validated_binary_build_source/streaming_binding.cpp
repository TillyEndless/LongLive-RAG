#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <vector>
#include "src/flash.h"
#include <cutlass/numeric_types.h>

namespace flash {
void launch_streaming_tile_producer(
    const void* src_k, const void* src_v,
    void* dst_k, void* dst_v,
    int* ready, int* counter,
    int seqlen_k, int heads, int head_dim,
    int tile_tokens, int producer_blocks,
    unsigned int delay_ns, cudaStream_t stream);
}

std::vector<at::Tensor> streaming_fwd(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor frontier,
    double softmax_scale, int64_t streaming_first_tile, int64_t streaming_last_tile,
    int64_t streaming_group_tiles) {

    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && frontier.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 && v.scalar_type() == at::kBFloat16,
                "prototype supports BF16 only");
    TORCH_CHECK(frontier.scalar_type() == at::kInt, "frontier must be int32");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4, "q/k/v must be [B,L,H,D]");
    TORCH_CHECK(q.size(0) == 1 && k.size(0) == 1 && v.size(0) == 1, "prototype currently supports B=1");
    TORCH_CHECK(q.size(3) == 128 && k.size(3) == 128 && v.size(3) == 128, "head_dim must be 128");
    TORCH_CHECK(q.size(2) == k.size(2) && k.size(2) == v.size(2), "prototype requires equal Q/KV heads");
    TORCH_CHECK(k.size(1) == v.size(1), "K/V length mismatch");
    TORCH_CHECK(q.stride(-1) == 1 && k.stride(-1) == 1 && v.stride(-1) == 1, "last dim must be contiguous");
    TORCH_CHECK(frontier.dim() == 1 && frontier.size(0) == 1, "frontier must be [batch] for B=1 prototype");
    TORCH_CHECK(frontier.is_contiguous(), "frontier must be contiguous");
    TORCH_CHECK(streaming_group_tiles >= 1 && (streaming_group_tiles & (streaming_group_tiles - 1)) == 0,
                "streaming_group_tiles must be power-of-two");

    c10::cuda::CUDAGuard guard(q.device());

    const int b = 1;
    const int sq = q.size(1);
    const int sk = k.size(1);
    const int h = q.size(2);
    const int d = 128;

    auto out = torch::empty_like(q);
    auto lse = torch::empty({b, h, sq}, q.options().dtype(torch::kFloat32));

    flash::Flash_fwd_params p{};
    p.q_ptr = q.data_ptr();
    p.k_ptr = k.data_ptr();
    p.v_ptr = v.data_ptr();
    p.o_ptr = out.data_ptr();
    p.softmax_lse_ptr = lse.data_ptr();

    p.q_batch_stride = q.stride(0);
    p.k_batch_stride = k.stride(0);
    p.v_batch_stride = v.stride(0);
    p.o_batch_stride = out.stride(0);
    p.q_row_stride = q.stride(1);
    p.k_row_stride = k.stride(1);
    p.v_row_stride = v.stride(1);
    p.o_row_stride = out.stride(1);
    p.q_head_stride = q.stride(2);
    p.k_head_stride = k.stride(2);
    p.v_head_stride = v.stride(2);
    p.o_head_stride = out.stride(2);

    p.b = b;
    p.h = h;
    p.h_k = h;
    p.h_h_k_ratio = 1;
    p.seqlen_q = sq;
    p.seqlen_k = sk;
    p.seqlen_q_rounded = ((sq + 127) / 128) * 128;
    p.seqlen_k_rounded = ((sk + 127) / 128) * 128;
    p.d = d;
    p.d_rounded = d;
    p.total_q = sq;

    p.scale_softmax = static_cast<float>(softmax_scale);
    p.scale_softmax_log2 = static_cast<float>(softmax_scale * M_LOG2E);
    p.p_dropout = 1.0f;
    p.p_dropout_in_uint8_t = 255;
    p.rp_dropout = 1.0f;
    p.scale_softmax_rp_dropout = p.scale_softmax;
    p.window_size_left = -1;
    p.window_size_right = -1;
    p.softcap = 0.0f;
    p.is_bf16 = true;
    p.is_causal = false;
    p.is_seqlens_k_cumulative = true;
    p.unpadded_lse = false;
    p.seqlenq_ngroups_swapped = false;

    p.streaming_ready_ptr = nullptr;
    p.streaming_ready_batch_stride = 0;
    p.streaming_ready_value = 1;
    p.streaming_first_tile = static_cast<int>(streaming_first_tile);
    p.streaming_last_tile = static_cast<int>(streaming_last_tile);
    p.streaming_frontier_ptr = frontier.data_ptr<int>();
    p.streaming_frontier_batch_stride = frontier.stride(0);
    p.streaming_group_tiles = static_cast<int>(streaming_group_tiles);

    auto stream = at::cuda::getCurrentCUDAStream(q.get_device()).stream();
    flash::run_mha_fwd_<cutlass::bfloat16_t, 128, false>(p, stream);
    return {out, lse};
}

void launch_producer(
    at::Tensor src_k, at::Tensor src_v,
    at::Tensor dst_k, at::Tensor dst_v,
    at::Tensor ready, at::Tensor counter,
    int64_t tile_tokens, int64_t producer_blocks, int64_t delay_ns) {
    TORCH_CHECK(src_k.is_cuda() && src_v.is_cuda() && dst_k.is_cuda() && dst_v.is_cuda(), "K/V must be CUDA");
    TORCH_CHECK(src_k.scalar_type() == at::kBFloat16 && src_v.scalar_type() == at::kBFloat16, "source BF16 required");
    TORCH_CHECK(dst_k.scalar_type() == at::kBFloat16 && dst_v.scalar_type() == at::kBFloat16, "dest BF16 required");
    TORCH_CHECK(src_k.sizes() == dst_k.sizes() && src_v.sizes() == dst_v.sizes(), "source/dest mismatch");
    TORCH_CHECK(src_k.dim() == 4 && src_k.size(0) == 1 && src_k.size(3) == 128, "prototype expects [1,L,H,128]");
    TORCH_CHECK(src_k.is_contiguous() && src_v.is_contiguous() && dst_k.is_contiguous() && dst_v.is_contiguous(), "K/V must be contiguous");
    TORCH_CHECK(ready.scalar_type() == at::kInt && counter.scalar_type() == at::kInt, "ready/counter int32 required");
    TORCH_CHECK(ready.is_contiguous() && counter.is_contiguous(), "ready/counter contiguous required");
    c10::cuda::CUDAGuard guard(src_k.device());
    auto stream = at::cuda::getCurrentCUDAStream(src_k.get_device()).stream();
    flash::launch_streaming_tile_producer(
        src_k.data_ptr(), src_v.data_ptr(), dst_k.data_ptr(), dst_v.data_ptr(),
        ready.data_ptr<int>(), counter.data_ptr<int>(),
        src_k.size(1), src_k.size(2), src_k.size(3),
        static_cast<int>(tile_tokens), static_cast<int>(producer_blocks),
        static_cast<unsigned int>(delay_ns), stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("streaming_fwd", &streaming_fwd, "FA2 forward with per-KV-tile readiness");
    m.def("launch_producer", &launch_producer, "Asynchronous reverse-order tiled K/V producer");
}
