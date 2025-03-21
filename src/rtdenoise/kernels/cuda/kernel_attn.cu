#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <stdlib.h>
#include <iostream>


namespace rtdenoise {

    at::Tensor kernel_attn_cuda(at::Tensor qk, at::Tensor v, int64_t kernel_size) {
        std::cerr << "ERROR: CUDA kernel for kernel attention has not been implemented yet!" << std::endl;
        exit(-1);
    }

    TORCH_LIBRARY_IMPL(rtdenoise, CUDA, m) {
        m.impl("kernel_attn", &kernel_attn_cuda);
    }

}
