#include <cuda_runtime.h>
#include <cuda/atomic>

__global__ void publish_frontier_kernel(int* ptr, int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        __threadfence();
        cuda::atomic_ref<int, cuda::thread_scope_device> a(*ptr);
        a.store(value, cuda::memory_order_release);
    }
}

extern "C" int ll_publish_frontier(int* ptr, int value, void* stream) {
    publish_frontier_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(ptr, value);
    return (int)cudaGetLastError();
}
