#include <cuda_runtime.h>
#include <cstdint>
#include "namespace_config.h"

namespace FLASH_NAMESPACE {

__global__ void streaming_tile_producer_kernel(
    const uint4* __restrict__ src_k,
    const uint4* __restrict__ src_v,
    uint4* __restrict__ dst_k,
    uint4* __restrict__ dst_v,
    int* __restrict__ ready,
    int* __restrict__ counter,
    size_t elems_per_token_u4,
    int seqlen_k,
    int tile_tokens,
    int num_tiles,
    unsigned int delay_ns) {

    while (true) {
        __shared__ int tile;
        if (threadIdx.x == 0) {
            const int ordinal = atomicAdd(counter, 1);
            tile = num_tiles - 1 - ordinal;  // match FA2 reverse K traversal
        }
        __syncthreads();
        if (tile < 0) { return; }

        const int token0 = tile * tile_tokens;
        const int ntok = min(tile_tokens, seqlen_k - token0);
        const size_t begin = static_cast<size_t>(token0) * elems_per_token_u4;
        const size_t count = static_cast<size_t>(ntok) * elems_per_token_u4;

        for (size_t j = threadIdx.x; j < count; j += blockDim.x) {
            dst_k[begin + j] = src_k[begin + j];
            dst_v[begin + j] = src_v[begin + j];
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            if (delay_ns) { __nanosleep(delay_ns); }
            __threadfence();
            atomicExch(&ready[tile], 1);
        }
        __syncthreads();
    }
}

void launch_streaming_tile_producer(
    const void* src_k, const void* src_v,
    void* dst_k, void* dst_v,
    int* ready, int* counter,
    int seqlen_k, int heads, int head_dim,
    int tile_tokens, int producer_blocks,
    unsigned int delay_ns, cudaStream_t stream) {

    const size_t bytes_per_token = static_cast<size_t>(heads) * head_dim * 2;
    const size_t elems_per_token_u4 = bytes_per_token / sizeof(uint4);
    const int num_tiles = (seqlen_k + tile_tokens - 1) / tile_tokens;
    streaming_tile_producer_kernel<<<producer_blocks, 256, 0, stream>>>(
        reinterpret_cast<const uint4*>(src_k),
        reinterpret_cast<const uint4*>(src_v),
        reinterpret_cast<uint4*>(dst_k),
        reinterpret_cast<uint4*>(dst_v),
        ready, counter, elems_per_token_u4, seqlen_k, tile_tokens, num_tiles, delay_ns);
}

} // namespace FLASH_NAMESPACE
