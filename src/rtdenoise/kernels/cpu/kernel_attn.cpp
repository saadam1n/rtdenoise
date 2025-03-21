#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm> // for std::max

// Shamelessly copied from https://pytorch.org/tutorials/advanced/cpp_custom_ops.html#cpp-custom-ops-tutorial 
// We need to do this so Python can interact with C++ code.
extern "C" {
    /* Creates a dummy empty _C module that can be imported from Python.
      The import from Python will load the .so consisting of this file
      in this extension, so that the TORCH_LIBRARY static initializers
      below are run. */
    PyObject* PyInit__C(void)
    {
        static struct PyModuleDef module_def = {
            PyModuleDef_HEAD_INIT,
            "_C",   /* name of module */
            NULL,   /* module documentation, may be NULL */
            -1,     /* size of per-interpreter state of the module,
                      or -1 if the module keeps state in global variables. */
            NULL,   /* methods */
        };
        return PyModule_Create(&module_def);
    }
  }

namespace rtdenoise {

    // if we go with arbitrary window sizes we may have to allocate memory for the softmax array
    // we don't want that. we will use flash attention to compute attention on the fly
    template<typename T>
    void kernel_attn_cpu_pixel(int C, int H, int W, int kernel_size, int y, int x, const T* qk, const T* v, T* a) {
        T accum_v[3] {0, 0, 0};
        
        int hks = kernel_size / 2;
        T scale_factor = 1 / sqrt(C);

        int num_seen = 0;

        T logit_max = -9999;
        T running_sum = 0;

        for(int i = y - hks; i <= y + hks; i++) {
            for(int j = x - hks; j <= x + hks; j++) {

                bool is_inside = (i >= 0 && j >= 0 && i < H && j < W);

                T logit = 0;
                if(is_inside) {
                    for(int c = 0; c < C; c++) {
                        logit += qk[c * H * W + i * W + j] * qk[c * H * W + y * W + x];
                    }
                }

                // scale logits by sqrt dim
                logit *= scale_factor;

                T new_logit_max = std::max(logit, logit_max);
                T weight = exp(logit - new_logit_max);
                T sum_modulation = exp(logit_max - new_logit_max);

                for(int k = 0; k < 3; k++) {
                    T v_read = (is_inside ? v[k * H * W + i * W + j] : 0);

                    accum_v[k] = (sum_modulation * accum_v[k] + weight * v_read);
                }

                running_sum = sum_modulation * running_sum + weight;
                logit_max = new_logit_max;

            }
        }

        for(int k = 0; k < 3; k++) {
            a[k * H * W + y * W + x] = accum_v[k] / running_sum; 
        }
    }

    template<typename T>
    void kernel_attn_cpu_impl(int N, int C, int H, int W, int kernel_size, const T* qk, const T* v, T* a) {
        // calculate half kernel size

        for(int n = 0; n < N; n++) {
            for(int y = 0; y < H; y++) {
                for(int x = 0; x < W; x++) {

                    kernel_attn_cpu_pixel(
                        C, H, W, kernel_size, y, x,
                        &qk[n * C * H * W],
                        &v[n * 3 * H * W],
                        &a[n * 3 * H * W]
                    );

                }
            }
        }
    }

    at::Tensor kernel_attn_cpu(at::Tensor qk, at::Tensor v, int64_t kernel_size) {
        // check if the sizes are valid
        TORCH_CHECK(qk.size(0) == v.size(0))
        TORCH_CHECK(3          == v.size(1))
        TORCH_CHECK(qk.size(2) == v.size(2))
        TORCH_CHECK(qk.size(3) == v.size(3))
        TORCH_CHECK(kernel_size % 2 == 1);

        // check if everything if a float
        TORCH_CHECK(qk.dtype() == at::kFloat);
        TORCH_CHECK(v.dtype() == at::kFloat);

        // ensure everything is on CPU
        TORCH_INTERNAL_ASSERT(qk.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(v.device().type() == at::DeviceType::CPU);

        // create input and output tensors
        // we need contig tensors so our kernel doesn't have to handle
        // non-contigious blocks of memory
        at::Tensor qk_contig = qk.contiguous();
        at::Tensor v_contig = v.contiguous();
        at::Tensor result = torch::empty(v_contig.sizes(), v_contig.options());

        const float* qk_ptr = qk_contig.data_ptr<float>();
        const float* v_ptr = v_contig.data_ptr<float>();
        float* a_ptr = result.data_ptr<float>();

        kernel_attn_cpu_impl<float>(
            qk.size(0), 
            qk.size(1), 
            qk.size(2), 
            qk.size(3), 
            kernel_size, 
            qk_ptr, 
            v_ptr, 
            a_ptr
        );

        return result;
    }

    TORCH_LIBRARY(rtdenoise, m) {
        m.def("kernel_attn(Tensor a, Tensor b, int kernel_size) -> Tensor");
    } 

    TORCH_LIBRARY_IMPL(rtdenoise, CPU, m) {
        m.impl("kernel_attn", &kernel_attn_cpu);
    }

}