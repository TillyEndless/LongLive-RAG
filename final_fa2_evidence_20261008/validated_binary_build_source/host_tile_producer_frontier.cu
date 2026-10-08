#include <cuda_runtime.h>
#include <cuda/atomic>
#include <cstddef>
#include <cstdint>

static void* hk=nullptr; static void* hv=nullptr;
static void* hdk=nullptr; static void* hdv=nullptr;
static size_t gbytes=0;

__device__ __forceinline__ void try_advance_frontier(
    int* ready, int* frontier, int first_tile, int last_tile) {
  cuda::atomic_ref<int, cuda::thread_scope_device> f(*frontier);
  while (true) {
    int cur = f.load(cuda::memory_order_acquire);
    if (cur <= first_tile) return;
    int next = cur - 1;
    cuda::atomic_ref<int, cuda::thread_scope_device> r(ready[next]);
    if (r.load(cuda::memory_order_acquire) == 0) return;
    int expected = cur;
    if (f.compare_exchange_strong(expected, next,
          cuda::memory_order_acq_rel, cuda::memory_order_acquire)) {
      continue;
    }
  }
}

__global__ void host_range_tile_producer_frontier(
    const uint4* __restrict__ sk, const uint4* __restrict__ sv,
    uint4* __restrict__ dk, uint4* __restrict__ dv,
    int* __restrict__ ready, int* __restrict__ counter,
    int* __restrict__ frontier,
    size_t vecs_per_token, int seqlen_k, int tile_tokens,
    int range_begin, int range_end, int first_tile, int last_tile,
    unsigned int delay_ns) {
  while (true) {
    __shared__ int tile;
    if (threadIdx.x==0) {
      int ord=atomicAdd(counter,1);
      tile=last_tile-ord;
    }
    __syncthreads();
    if (tile < first_tile) return;

    int lo=max(tile*tile_tokens, range_begin);
    int hi=min(min((tile+1)*tile_tokens,seqlen_k), range_end);
    size_t begin=(size_t)lo*vecs_per_token;
    size_t count=(size_t)(hi-lo)*vecs_per_token;
    for(size_t i=threadIdx.x;i<count;i+=blockDim.x) {
      dk[begin+i]=sk[begin+i];
      dv[begin+i]=sv[begin+i];
    }
    __syncthreads();

    if(threadIdx.x==0) {
      if(delay_ns) __nanosleep(delay_ns);
      // Publish tile only after both K and V writes are globally visible.
      __threadfence();
      cuda::atomic_ref<int, cuda::thread_scope_device> r(ready[tile]);
      r.store(1, cuda::memory_order_release);
      try_advance_frontier(ready, frontier, first_tile, last_tile);
    }
    __syncthreads();
  }
}

extern "C" int htp_prepare(const void* src_k,const void* src_v,size_t bytes,int dev) {
  if(hk){cudaFreeHost(hk);hk=nullptr;hdk=nullptr;}
  if(hv){cudaFreeHost(hv);hv=nullptr;hdv=nullptr;}
  cudaError_t e=cudaSetDevice(dev); if(e) return (int)e;
  e=cudaHostAlloc(&hk,bytes,cudaHostAllocMapped|cudaHostAllocPortable); if(e) return (int)e;
  e=cudaHostAlloc(&hv,bytes,cudaHostAllocMapped|cudaHostAllocPortable); if(e) return (int)e;
  e=cudaHostGetDevicePointer(&hdk,hk,0); if(e) return (int)e;
  e=cudaHostGetDevicePointer(&hdv,hv,0); if(e) return (int)e;
  e=cudaMemcpy(hk,src_k,bytes,cudaMemcpyDeviceToHost); if(e) return (int)e;
  e=cudaMemcpy(hv,src_v,bytes,cudaMemcpyDeviceToHost); if(e) return (int)e;
  gbytes=bytes; return 0;
}

extern "C" int htp_launch_range_frontier(
    void* dst_k,void* dst_v,int* ready,int* counter,int* frontier,
    int seqlen_k,int heads,int d,int tile_tokens,int blocks,
    int range_begin,int range_end,unsigned int delay_ns,void* stream) {
  if(!hk||!hv) return -1001;
  if(range_begin<0||range_end>seqlen_k||range_begin>=range_end) return -1004;
  size_t bytes_per_token=(size_t)heads*d*2;
  if(bytes_per_token%sizeof(uint4)) return -1002;
  int first=range_begin/tile_tokens, last=(range_end-1)/tile_tokens;
  host_range_tile_producer_frontier<<<blocks,256,0,(cudaStream_t)stream>>>(
    (const uint4*)hdk,(const uint4*)hdv,(uint4*)dst_k,(uint4*)dst_v,
    ready,counter,frontier,bytes_per_token/sizeof(uint4),seqlen_k,tile_tokens,
    range_begin,range_end,first,last,delay_ns);
  return (int)cudaGetLastError();
}
