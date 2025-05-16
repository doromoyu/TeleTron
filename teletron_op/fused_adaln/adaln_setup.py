# Copyright (c) 2025 TeleAI-infra Team. All rights reserved.
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="fused_adaln",
    include_dirs=["include"],
    ext_modules=[
        CUDAExtension(
            "fused_adaln",
            sources=[
                "adaln_ops.cpp",
                "adaln_forward.cu",
                "adaln_backward.cu",
                "scale_shift_backward.cu"
            ],
            extra_compile_args={
                "cxx": ["-O3"], 
                'nvcc': [
                    '-O3', 
                    '-DENABLE_BF16', 
                    '--use_fast_math', 
                    '-gencode=arch=compute_90,code=sm_90', 
                    '-gencode=arch=compute_90,code=compute_90' 
                ]
            }
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)