from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name='rtdenoise',
    version='0.1.0',
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    ext_modules=[
        CUDAExtension(
            name='rtdenoise._C',
            sources=[
                'src/rtdenoise/kernels/cuda/kernel_attn.cu',  
                'src/rtdenoise/kernels/cpu/kernel_attn.cpp'  
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