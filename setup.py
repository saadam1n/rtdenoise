from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

import os

# Define the target CUDA architectures
cuda_architectures = [
    #'3.5',  # Kepler
    #'5.0',  # Maxwell
    '6.0',  # Pascal
    '7.0',  # Volta
    '7.5',  # Turing
    '8.0',  # Ampere
    '8.6',  # Ampere
    '8.9',  # Ada
    '9.0',  # Hopper
]

# Set the TORCH_CUDA_ARCH_LIST environment variable
os.environ['TORCH_CUDA_ARCH_LIST'] = ';'.join(cuda_architectures)

setup(
    name='rtdenoise',
    version='0.1.0',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    ext_modules=[
        CUDAExtension(
            name='rtdenoise._C',
            sources=[
                'src/rtdenoise/kernels/cpu/kernel_attn.cpp',
                'src/rtdenoise/kernels/cpu/upscale_attn.cpp',
                
                'src/rtdenoise/kernels/cuda/kernel_attn.cu',  
                'src/rtdenoise/kernels/cuda/upscale_attn.cu',
            ],
            extra_compile_args={
                'cxx': ['-DPy_LIMITED_API=0x03090000', '-O3'], 
                'nvcc': ['-DPy_LIMITED_API=0x03090000', '-O3']
            },
            py_limited_api=True
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp39"}}
)