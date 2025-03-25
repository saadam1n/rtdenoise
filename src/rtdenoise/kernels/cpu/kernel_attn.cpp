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
    void kernel_attn_cpu_pixel(
        int C, int H, int W, int kernel_size, int y, int x, 
        const T* qk, const T* v, T* a,
        T* L, T* m
    ) {
        T accum_v[3] {0, 0, 0};
        
        int hks = kernel_size / 2;
        T scale_factor = 1 / sqrt(C);

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

        if(L) {
            L[y * W + x] = running_sum;
        }

        if(m) {
            m[y * W + x] = logit_max;
        }

    }

    template<typename T>
    void kernel_attn_cpu_impl(
        int N, int C, int H, int W, int kernel_size, 
        const T* qk, const T* v, T* a,
        T* L, T* m
    ) {

        for(int n = 0; n < N; n++) {
            for(int y = 0; y < H; y++) {
                for(int x = 0; x < W; x++) {

                    kernel_attn_cpu_pixel(
                        C, H, W, kernel_size, y, x,
                        &qk[n * C * H * W],
                        &v[n * 3 * H * W],
                        &a[n * 3 * H * W],
                        L ? &L[n * H * W] : nullptr,
                        m ? &m[n * H * W] : nullptr
                    );

                }
            }
        }

    }

    template<typename T>
    void kernel_attn_bwd_cpu_pixel(
        int C, int H, int W, int kernel_size, int y, int x,
        const T* qk, const T* v, const T* a,
        const T* L, const T* m,
        const T* dLda, T* dLdqk, T* dLdv
    ) {

        T local_dLda[3] {0, 0, 0};
        T D = 0;
        for(int k = 0; k < 3; k++) {
            local_dLda[k] = dLda[k * H * W + y * W + x];
            D += local_dLda[k] * a[k * H * W + y * W + x];
        }

        T running_sum = L[y * W + x];
        T logit_max = m[y * W + x];

        int hks = kernel_size / 2;
        T scale_factor = 1 / sqrt(C);

        for(int i = y - hks; i <= y + hks; i++) {
            for(int j = x - hks; j <= x + hks; j++) {

                bool is_inside = (i >= 0 && j >= 0 && i < H && j < W);

                T dLdweight = -D;
                T logit = 0;

                if(is_inside) {
                    for(int k = 0; k < 3; k++) {
                        dLdweight += local_dLda[k] * v[k * H * W + i * W + j];
                    }

                    for(int c = 0; c < C; c++) {
                        logit += qk[c * H * W + i * W + j] * qk[c * H * W + y * W + x];
                    }
                }

                // scale logits by sqrt dim
                logit *= scale_factor;
                T weight = exp(logit - logit_max) / running_sum;

                T dLdlogit = dLdweight * weight * scale_factor;

                // compute query and key derivatives
                for(int c = 0; c < C; c++) {
                    T k_read = (is_inside ? qk[c * H * W + i * W + j] : 0);

                    dLdqk[c * H * W + y * W + x] += dLdlogit * k_read;

                    if(is_inside) {
                        dLdqk[c * H * W + i * W + j] += dLdlogit * qk[c * H * W + y * W + x];
                    }
                }

                if(is_inside) {
                    
                    for(int k = 0; k < 3; k++) {
                        dLdv[k * H * W + i * W + j] += weight * local_dLda[k];
                    }
                }

            }
        }

    }

    template<typename T>
    void kernel_attn_bwd_cpu_impl(
        int N, int C, int H, int W, int kernel_size, 
        const T* qk, const T* v, const T* a,
        const T* L, const T* m,
        const T* dLda, T* dLdqk, T* dLdv
    ) {
 
        for(int n = 0; n < N; n++) {
            for(int y = 0; y < H; y++) {
                for(int x = 0; x < W; x++) {

                    kernel_attn_bwd_cpu_pixel(
                        C, H, W, kernel_size, y, x,
                        &qk[n * C * H * W],
                        &v[n * 3 * H * W],
                        &a[n * 3 * H * W],
                        &L[n * H * W],
                        &m[n * H * W],
                        &dLda[n * 3 * H * W],
                        &dLdqk[n * C * H * W],
                        &dLdv[n * 3 * H * W]
                    );

                }
            }
        }

    }

    at::Tensor kernel_attn_cpu(
        at::Tensor qk, at::Tensor v, int64_t kernel_size, 
        c10::optional<at::Tensor> L, c10::optional<at::Tensor> m
    ) {
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
        float* L_ptr = L.has_value() ? L.value().data_ptr<float>() : nullptr;
        float* m_ptr = m.has_value() ? m.value().data_ptr<float>() : nullptr;


        kernel_attn_cpu_impl<float>(
            qk.size(0), 
            qk.size(1), 
            qk.size(2), 
            qk.size(3), 
            kernel_size, 
            qk_ptr, 
            v_ptr, 
            a_ptr,
            L_ptr,
            m_ptr
        );

        return result;
    }

    void kernel_attn_bwd_cpu(
        at::Tensor qk, at::Tensor v, int64_t kernel_size, 
        at::Tensor L, at::Tensor m, at::Tensor a, 
        at::Tensor dLda, at::Tensor dLdqk, at::Tensor dLdv
    ) {
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
        at::Tensor L_contig = L.contiguous();
        at::Tensor m_contig = m.contiguous();
        at::Tensor a_contig = a.contiguous();
        at::Tensor dLda_contig = dLda.contiguous();

        const float* qk_ptr = qk_contig.data_ptr<float>();
        const float* v_ptr = v_contig.data_ptr<float>();
        float* L_ptr = L_contig.data_ptr<float>();
        float* m_ptr = m_contig.data_ptr<float>();
        float* a_ptr = a_contig.data_ptr<float>();
        float* dLda_ptr = dLda_contig.data_ptr<float>();
        float* dLdqk_ptr = dLdqk.data_ptr<float>();
        float* dLdv_ptr = dLdv.data_ptr<float>();

        kernel_attn_bwd_cpu_impl<float>(
            qk.size(0), 
            qk.size(1), 
            qk.size(2), 
            qk.size(3), 
            kernel_size, 
            qk_ptr, 
            v_ptr, 
            a_ptr,
            L_ptr,
            m_ptr,
            dLda_ptr,
            dLdqk_ptr,
            dLdv_ptr
        );
    }

    TORCH_LIBRARY(rtdenoise, m) {
        m.def(
            "kernel_attn(Tensor qk, Tensor v, int kernel_size, Tensor? L, Tensor? m) -> Tensor"
        );

        m.def(
            "kernel_attn_bwd(Tensor qk, Tensor v, int kernel_size, Tensor L, Tensor m, Tensor a, Tensor dLda, Tensor dLdqk, Tensor dLdv) -> ()"
        );
    } 

    TORCH_LIBRARY_IMPL(rtdenoise, CPU, m) {
        m.impl("kernel_attn", &kernel_attn_cpu);
        m.impl("kernel_attn_bwd", &kernel_attn_bwd_cpu);
    }

}