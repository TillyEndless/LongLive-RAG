from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from pathlib import Path
E=Path("/data/zxl/fa2_grouped_frontier_20261007")
R=E/"flash-attention-v2.8.3-streaming"
setup(
    name="fa2_streaming_consumer_ext",
    ext_modules=[CUDAExtension(
        name="fa2_streaming_consumer_ext",
        sources=[
            str(E/"streaming_binding.cpp"),
            str(E/"streaming_producer.cu"),
            str(R/"csrc/flash_attn/src/flash_fwd_hdim128_bf16_sm80.cu"),
        ],
        include_dirs=[
            str(R/"csrc/flash_attn"),
            str(R/"csrc/flash_attn/src"),
            str(R/"csrc/cutlass/include"),
        ],
        extra_compile_args={
            "cxx":["-O3","-std=c++17"],
            "nvcc":[
                "-O3","-std=c++17","--use_fast_math",
                "--expt-relaxed-constexpr","--expt-extended-lambda",
                "-U__CUDA_NO_HALF_OPERATORS__","-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_HALF2_OPERATORS__","-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                "-DFLASHATTENTION_DISABLE_DROPOUT",
                "-DFLASHATTENTION_DISABLE_ALIBI",
                "-DFLASHATTENTION_DISABLE_SOFTCAP",
                "-DFLASHATTENTION_DISABLE_LOCAL",
                "-gencode=arch=compute_90,code=sm_90",
            ],
        },
    )],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=False)}
)
