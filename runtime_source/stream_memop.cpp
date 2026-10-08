#include <cuda.h>
#include <cstdint>

extern "C" int ll_stream_write_u32(void* stream, void* addr, uint32_t value) {
    CUresult r = cuStreamWriteValue32(
        reinterpret_cast<CUstream>(stream),
        static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(addr)),
        static_cast<cuuint32_t>(value),
        CU_STREAM_WRITE_VALUE_DEFAULT);
    return static_cast<int>(r);
}
