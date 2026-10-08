from pathlib import Path
E=Path("/data/zxl/fa2_grouped_frontier_20261007")
R=E/"flash-attention-v2.8.3-streaming"

p=R/"csrc/flash_attn/src/flash.h"
s=p.read_text()
old="""    int streaming_ready_value;
    int streaming_first_tile;
    int streaming_last_tile;
"""
new="""    int streaming_ready_value;
    int streaming_first_tile;
    int streaming_last_tile;
    int * __restrict__ streaming_frontier_ptr;
    index_t streaming_frontier_batch_stride;
    int streaming_group_tiles;
"""
assert old in s
p.write_text(s.replace(old,new,1))

p=R/"csrc/flash_attn/src/flash_fwd_kernel.h"
s=p.read_text()
a=s.index("template<typename Params>\ninline __device__ void streaming_wait_kv_tile")
b=s.index("\n\ntemplate<typename Kernel_traits",a)
helper="""template<typename Params>
inline __device__ void streaming_wait_group_frontier(
        const Params &params, const int bidb, const int n_block) {
    if (params.streaming_frontier_ptr == nullptr) { return; }
    if (n_block < params.streaming_first_tile || n_block > params.streaming_last_tile) { return; }

    const int g = params.streaming_group_tiles;
    const int delta = params.streaming_last_tile - n_block;
    if ((delta & (g - 1)) != 0) { return; }

    int need = n_block - g + 1;
    if (need < params.streaming_first_tile) { need = params.streaming_first_tile; }

    if (threadIdx.x == 0) {
        int *ptr = params.streaming_frontier_ptr + bidb * params.streaming_frontier_batch_stride;
        cuda::atomic_ref<int, cuda::thread_scope_device> frontier(*ptr);
        int observed;
        do {
            observed = frontier.load(cuda::memory_order_acquire);
            if (observed > need) { __nanosleep(64); }
        } while (observed > need);
    }
    __syncthreads();
}
"""
s=s[:a]+helper+s[b:]
assert s.count("streaming_wait_kv_tile(params, bidb, n_block);")==1
s=s.replace("streaming_wait_kv_tile(params, bidb, n_block);",
            "streaming_wait_group_frontier(params, bidb, n_block);",1)
assert s.count("streaming_wait_kv_tile(params, bidb, n_block - 1);")==2
s=s.replace("streaming_wait_kv_tile(params, bidb, n_block - 1);",
            "streaming_wait_group_frontier(params, bidb, n_block - 1);")
p.write_text(s)

p=E/"streaming_binding.cpp"
s=p.read_text()
prefix,rest=s.split("\nvoid launch_producer(",1)
old_sig="""std::vector<at::Tensor> streaming_fwd(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor ready,
    double softmax_scale, int64_t streaming_first_tile, int64_t streaming_last_tile) {"""
new_sig="""std::vector<at::Tensor> streaming_fwd(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor frontier,
    double softmax_scale, int64_t streaming_first_tile, int64_t streaming_last_tile,
    int64_t streaming_group_tiles) {"""
assert old_sig in prefix
prefix=prefix.replace(old_sig,new_sig,1)
prefix=prefix.replace("q.is_cuda() && k.is_cuda() && v.is_cuda() && ready.is_cuda()",
                      "q.is_cuda() && k.is_cuda() && v.is_cuda() && frontier.is_cuda()",1)
prefix=prefix.replace('TORCH_CHECK(ready.scalar_type() == at::kInt, "ready must be int32");',
                      'TORCH_CHECK(frontier.scalar_type() == at::kInt, "frontier must be int32");',1)
old_chk='''    TORCH_CHECK(ready.dim() == 2 && ready.size(0) == 1, "ready must be [1, num_tiles]");
    TORCH_CHECK(ready.is_contiguous(), "ready must be contiguous");'''
new_chk='''    TORCH_CHECK(frontier.dim() == 1 && frontier.size(0) == 1, "frontier must be [batch] for B=1 prototype");
    TORCH_CHECK(frontier.is_contiguous(), "frontier must be contiguous");
    TORCH_CHECK(streaming_group_tiles >= 1 && (streaming_group_tiles & (streaming_group_tiles - 1)) == 0,
                "streaming_group_tiles must be power-of-two");'''
assert old_chk in prefix
prefix=prefix.replace(old_chk,new_chk,1)
old_params='''    p.streaming_ready_ptr = ready.data_ptr<int>();
    p.streaming_ready_batch_stride = ready.stride(0);
    p.streaming_ready_value = 1;
    p.streaming_first_tile = static_cast<int>(streaming_first_tile);
    p.streaming_last_tile = static_cast<int>(streaming_last_tile);'''
new_params='''    p.streaming_ready_ptr = nullptr;
    p.streaming_ready_batch_stride = 0;
    p.streaming_ready_value = 1;
    p.streaming_first_tile = static_cast<int>(streaming_first_tile);
    p.streaming_last_tile = static_cast<int>(streaming_last_tile);
    p.streaming_frontier_ptr = frontier.data_ptr<int>();
    p.streaming_frontier_batch_stride = frontier.stride(0);
    p.streaming_group_tiles = static_cast<int>(streaming_group_tiles);'''
assert old_params in prefix
prefix=prefix.replace(old_params,new_params,1)
p.write_text(prefix+"\nvoid launch_producer("+rest)
print("patched grouped frontier")
