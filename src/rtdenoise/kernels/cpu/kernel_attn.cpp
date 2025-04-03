#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm> // for std::max
#include <cmath> // for std::isnan

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

    template<typename T>
    struct KernelAttnResultCPU {
        T accum_a[3] {0, 0, 0};
        T running_sum = 0;
        T logit_max = -9999;
    };

    // if we go with arbitrary window sizes we may have to allocate memory for the softmax array
    // we don't want that. we will use flash attention to compute attention on the fly
    template<typename T>
    void kernel_attn_cpu_pixel(
        int C, int H, int W, 
        int kernel_size, 
        int skip_center,
        int y, int x,  
        const T* q, 
        const T* k, 
        const T* v, 
        KernelAttnResultCPU<T>& res
    ) {
        int hks = kernel_size / 2;
        T scale_factor = 1 / sqrt(C);

        for(int i = y - hks; i <= y + hks; i++) {
            for(int j = x - hks; j <= x + hks; j++) {

                if(skip_center && i == y && j == x) {
                    continue;
                }

                bool is_inside = (i >= 0 && j >= 0 && i < H && j < W);

                T logit = 0;
                if(is_inside) {
                    for(int c = 0; c < C; c++) {
                        logit += q[c * H * W + y * W + x] * k[c * H * W + i * W + j];
                    }
                }

                // scale logits by sqrt dim
                logit *= scale_factor;

                T new_logit_max = std::max(logit, res.logit_max);
                T weight = exp(logit - new_logit_max);
                T sum_modulation = exp(res.logit_max - new_logit_max);

                for(int k = 0; k < 3; k++) {
                    T v_read = (is_inside ? v[k * H * W + i * W + j] : 0);

                    res.accum_a[k] = (sum_modulation * res.accum_a[k] + weight * v_read);
                }

                res.running_sum = sum_modulation * res.running_sum + weight;
                res.logit_max = new_logit_max;

            }
        }



    }

    template<typename T>
    void kernel_attn_cpu_impl(
        int N, int C, int H, int W, 
        int kernel_size, 
        int skip_center,
        const T* qk0, const T* v0, 
        const T* qk1, const T* v1, 
        const T* qk2, const T* v2, 
        T* a,
        T* L, T* m
    ) {
        const T* klist[] = {qk0, qk1, qk2};
        const T* vlist[] = {v0, v1, v2};

        for(int n = 0; n < N; n++) {
            for(int y = 0; y < H; y++) {
                for(int x = 0; x < W; x++) {

                    KernelAttnResultCPU<T> res;

                    for(int b = 0; b < 3; b++) {
                        if(!klist[b]) {
                            continue;
                        }

                        kernel_attn_cpu_pixel(
                            C, H, W, 
                            kernel_size, 
                            b == 0 ? skip_center : 0,
                            y, x,
                            &qk0[n * C * H * W],
                            &klist[b][n * C * H * W],
                            &vlist[b][n * 3 * H * W],
                            res
                        );
                    }

                    for(int k = 0; k < 3; k++) {
                        a[n * 3 * H * W + k * H * W + y * W + x] = res.accum_a[k] / res.running_sum; 
                    }
            
                    if(L) {
                        L[n * H * W + y * W + x] = res.running_sum;
                    }
            
                    if(m) {
                        m[n * H * W + y * W + x] = res.logit_max;
                    }

                }
            }
        }

    }

    template<typename T>
    void kernel_attn_bwd_cpu_pixel(
        int C, int H, int W, 
        int kernel_size, 
        int skip_center,
        int y, int x,
        const T* q, const T* k, const T* v, 
        KernelAttnResultCPU<T>& res,
        const T* dLda, 
        T* dLdq, 
        T* dLdk, 
        T* dLdv
    ) {

        T local_dLda[3] {0, 0, 0};
        T D = 0;
        for(int k = 0; k < 3; k++) {
            local_dLda[k] = dLda[k * H * W + y * W + x];
            D += local_dLda[k] * res.accum_a[k];
        }

        int hks = kernel_size / 2;
        T scale_factor = 1 / sqrt(C);

        for(int i = y - hks; i <= y + hks; i++) {
            for(int j = x - hks; j <= x + hks; j++) {

                if(skip_center && i == y && j == x) {
                    continue;
                }

                bool is_inside = (i >= 0 && j >= 0 && i < H && j < W);

                T dLdweight = -D;
                T logit = 0;

                if(is_inside) {
                    for(int k = 0; k < 3; k++) {
                        dLdweight += local_dLda[k] * v[k * H * W + i * W + j];
                    }

                    for(int c = 0; c < C; c++) {
                        logit += q[c * H * W + y * W + x] * k[c * H * W + i * W + j];
                    }
                }

                // scale logits by sqrt dim
                logit *= scale_factor;
                T weight = exp(logit - res.logit_max) / res.running_sum;

                T dLdlogit = dLdweight * weight * scale_factor;

                // compute query and key derivatives
                for(int c = 0; c < C; c++) {
                    T k_read = (is_inside ? k[c * H * W + i * W + j] : 0);

                    dLdq[c * H * W + y * W + x] += dLdlogit * k_read;

                    if(is_inside) {
                        dLdk[c * H * W + i * W + j] += dLdlogit * q[c * H * W + y * W + x];
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
        int N, int C, int H, int W, 
        int kernel_size, int skip_center,
        const T* qk0, const T* v0, 
        const T* qk1, const T* v1, 
        const T* qk2, const T* v2, 
        const T* L, const T* m,
        const T* a,
        const T* dLda, 
        T* dLdqk0, T* dLdv0,
        T* dLdqk1, T* dLdv1,
        T* dLdqk2, T* dLdv2
    ) {
        const T* klist[] = {qk0, qk1, qk2};
        const T* vlist[] = {v0, v1, v2};

        T* dLklist[] = {dLdqk0, dLdqk1, dLdqk2};
        T* dLvlist[] = {dLdv0, dLdv1, dLdv2};


        for(int n = 0; n < N; n++) {
            for(int y = 0; y < H; y++) {
                for(int x = 0; x < W; x++) {

                    KernelAttnResultCPU<T> res;

                    for(int k = 0; k < 3; k++) {
                        res.accum_a[k] = a[n * 3 * H * W + k * H * W + y * W + x];
                    }
                    res.running_sum = L[n * H * W + y * W + x];
                    res.logit_max = m[n * H * W + y * W + x];

                    for(int b = 0; b < 3; b++) {
                        if(!klist[b]) {
                            continue;
                        }

                        kernel_attn_bwd_cpu_pixel<T>(
                            C, H, W, 
                            kernel_size, 
                            b == 0 ? skip_center : 0,
                            y, x,
                            &qk0[n * C * H * W],
                            &klist[b][n * C * H * W],
                            &vlist[b][n * 3 * H * W],
                            res,
                            &dLda[n * 3 * H * W], 
                            &dLdqk0[n * C * H * W],
                            &dLklist[b][n * C * H * W],
                            &dLvlist[b][n * 3 * H * W]
                        );
                    }



                }
            }
        }

    }

    at::Tensor kernel_attn_cpu(
        at::Tensor qk0, at::Tensor v0, 
        c10::optional<at::Tensor> qk1, c10::optional<at::Tensor> v1,
        c10::optional<at::Tensor> qk2, c10::optional<at::Tensor> v2,
        int64_t kernel_size, int64_t skip_center,
        c10::optional<at::Tensor> L, c10::optional<at::Tensor> m
    ) {
        // check if the sizes are valid
        TORCH_CHECK(qk0.size(0) == v0.size(0))
        TORCH_CHECK(3           == v0.size(1))
        TORCH_CHECK(qk0.size(2) == v0.size(2))
        TORCH_CHECK(qk0.size(3) == v0.size(3))
        TORCH_CHECK(qk1.has_value() ? qk1.value().sizes() == qk0.sizes() : true);
        TORCH_CHECK(qk2.has_value() ? qk2.value().sizes() == qk0.sizes() : true);
        TORCH_CHECK(v1.has_value() ? v1.value().sizes() == v0.sizes() : true);
        TORCH_CHECK(v2.has_value() ? v2.value().sizes() == v0.sizes() : true);

        TORCH_CHECK(kernel_size % 2 == 1);

        // check if everything if a float
        TORCH_CHECK(qk0.dtype() == at::kFloat);
        TORCH_CHECK(qk1.has_value() ? qk1.value().dtype() == at::kFloat : true);
        TORCH_CHECK(qk2.has_value() ? qk2.value().dtype() == at::kFloat : true);

        TORCH_CHECK(v0.dtype() == at::kFloat);
        TORCH_CHECK(v1.has_value() ? v1.value().dtype() == at::kFloat : true);
        TORCH_CHECK(v2.has_value() ? v2.value().dtype() == at::kFloat : true);

        // ensure everything is on CPU
        TORCH_INTERNAL_ASSERT(qk0.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(qk1.has_value() ? qk1.value().device().type() == at::DeviceType::CPU : true);
        TORCH_INTERNAL_ASSERT(qk2.has_value() ? qk2.value().device().type() == at::DeviceType::CPU : true);

        TORCH_INTERNAL_ASSERT(v0.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(v1.has_value() ? v1.value().device().type() == at::DeviceType::CPU : true);
        TORCH_INTERNAL_ASSERT(v2.has_value() ? v2.value().device().type() == at::DeviceType::CPU : true);

        // create input and output tensors
        // we need contig tensors so our kernel doesn't have to handle
        // non-contigious blocks of memory
        at::Tensor qk0_contig = qk0.contiguous();
        at::Tensor qk1_contig = qk1.has_value() ? qk1.value().contiguous() : at::Tensor();
        at::Tensor qk2_contig = qk2.has_value() ? qk2.value().contiguous() : at::Tensor();

        at::Tensor v0_contig = v0.contiguous();
        at::Tensor v1_contig = v1.has_value() ? v1.value().contiguous() : at::Tensor();
        at::Tensor v2_contig = v2.has_value() ? v2.value().contiguous() : at::Tensor();


        at::Tensor result = torch::empty(v0_contig.sizes(), v0_contig.options());

        const float* qk0_ptr = qk0_contig.data_ptr<float>();
        const float* qk1_ptr = qk1_contig.defined() ? qk1_contig.data_ptr<float>() : nullptr;
        const float* qk2_ptr = qk2_contig.defined() ? qk2_contig.data_ptr<float>() : nullptr;

        const float* v0_ptr = v0_contig.data_ptr<float>();
        const float* v1_ptr = v1_contig.defined() ? v1_contig.data_ptr<float>() : nullptr;
        const float* v2_ptr = v2_contig.defined() ? v2_contig.data_ptr<float>() : nullptr;
        float* a_ptr = result.data_ptr<float>();
        float* L_ptr = L.has_value() ? L.value().data_ptr<float>() : nullptr;
        float* m_ptr = m.has_value() ? m.value().data_ptr<float>() : nullptr;


        kernel_attn_cpu_impl<float>(
            qk0.size(0), 
            qk0.size(1), 
            qk0.size(2), 
            qk0.size(3), 
            kernel_size, 
            skip_center,
            qk0_ptr, 
            v0_ptr, 
            qk1_ptr, 
            v1_ptr, 
            qk2_ptr, 
            v2_ptr, 
            a_ptr,
            L_ptr,
            m_ptr
        );

        return result;
    }

    void kernel_attn_bwd_cpu(
        at::Tensor qk0, at::Tensor v0, 
        c10::optional<at::Tensor> qk1, c10::optional<at::Tensor> v1,
        c10::optional<at::Tensor> qk2, c10::optional<at::Tensor> v2,
        int64_t kernel_size, int64_t skip_center,
        at::Tensor L, at::Tensor m,
        at::Tensor a,
        at::Tensor dLda, 
        at::Tensor dLdqk0, at::Tensor dLdv0,
        c10::optional<at::Tensor> dLdqk1, c10::optional<at::Tensor> dLdv1,
        c10::optional<at::Tensor> dLdqk2, c10::optional<at::Tensor> dLdv2
    ) {
        // check if the sizes are valid
        TORCH_CHECK(qk0.size(0) == v0.size(0))
        TORCH_CHECK(3           == v0.size(1))
        TORCH_CHECK(qk0.size(2) == v0.size(2))
        TORCH_CHECK(qk0.size(3) == v0.size(3))
        TORCH_CHECK(qk1.has_value() ? qk1.value().sizes() == qk0.sizes() : true);
        TORCH_CHECK(qk2.has_value() ? qk2.value().sizes() == qk0.sizes() : true);
        TORCH_CHECK(v1.has_value() ? v1.value().sizes() == v0.sizes() : true);
        TORCH_CHECK(v2.has_value() ? v2.value().sizes() == v0.sizes() : true);

        TORCH_CHECK(kernel_size % 2 == 1);

        // check if everything if a float
        TORCH_CHECK(qk0.dtype() == at::kFloat);
        TORCH_CHECK(qk1.has_value() ? qk1.value().dtype() == at::kFloat : true);
        TORCH_CHECK(qk2.has_value() ? qk2.value().dtype() == at::kFloat : true);

        TORCH_CHECK(v0.dtype() == at::kFloat);
        TORCH_CHECK(v1.has_value() ? v1.value().dtype() == at::kFloat : true);
        TORCH_CHECK(v2.has_value() ? v2.value().dtype() == at::kFloat : true);

        // ensure everything is on CPU
        TORCH_INTERNAL_ASSERT(qk0.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(qk1.has_value() ? qk1.value().device().type() == at::DeviceType::CPU : true);
        TORCH_INTERNAL_ASSERT(qk2.has_value() ? qk2.value().device().type() == at::DeviceType::CPU : true);

        TORCH_INTERNAL_ASSERT(v0.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(v1.has_value() ? v1.value().device().type() == at::DeviceType::CPU : true);
        TORCH_INTERNAL_ASSERT(v2.has_value() ? v2.value().device().type() == at::DeviceType::CPU : true);

        // create input and output tensors
        // we need contig tensors so our kernel doesn't have to handle
        // non-contigious blocks of memory
        at::Tensor qk0_contig = qk0.contiguous();
        at::Tensor qk1_contig = qk1.has_value() ? qk1.value().contiguous() : at::Tensor();
        at::Tensor qk2_contig = qk2.has_value() ? qk2.value().contiguous() : at::Tensor();

        at::Tensor v0_contig = v0.contiguous();
        at::Tensor v1_contig = v1.has_value() ? v1.value().contiguous() : at::Tensor();
        at::Tensor v2_contig = v2.has_value() ? v2.value().contiguous() : at::Tensor();

        at::Tensor a_contig = a.contiguous();

        at::Tensor dLda_contig = dLda.contiguous();

        const float* qk0_ptr = qk0_contig.data_ptr<float>();
        const float* qk1_ptr = qk1_contig.defined() ? qk1_contig.data_ptr<float>() : nullptr;
        const float* qk2_ptr = qk2_contig.defined() ? qk2_contig.data_ptr<float>() : nullptr;

        const float* v0_ptr = v0_contig.data_ptr<float>();
        const float* v1_ptr = v1_contig.defined() ? v1_contig.data_ptr<float>() : nullptr;
        const float* v2_ptr = v2_contig.defined() ? v2_contig.data_ptr<float>() : nullptr;
        float* a_ptr = a_contig.data_ptr<float>();
        float* L_ptr = L.data_ptr<float>();
        float* m_ptr = m.data_ptr<float>();

        float* dLda_ptr = dLda_contig.data_ptr<float>();

        float* dLdqk0_ptr = dLdqk0.data_ptr<float>();
        float* dLdqk1_ptr = dLdqk1.has_value() ? dLdqk1.value().data_ptr<float>() : nullptr;
        float* dLdqk2_ptr = dLdqk1.has_value() ? dLdqk2.value().data_ptr<float>() : nullptr;

        float* dLdv0_ptr = dLdv0.data_ptr<float>();
        float* dLdv1_ptr = dLdv1.has_value() ? dLdv1.value().data_ptr<float>() : nullptr;
        float* dLdv2_ptr = dLdv2.has_value() ? dLdv2.value().data_ptr<float>() : nullptr;

        kernel_attn_bwd_cpu_impl<float>(
            qk0.size(0), 
            qk0.size(1), 
            qk0.size(2), 
            qk0.size(3), 
            kernel_size, 
            skip_center,
            qk0_ptr, 
            v0_ptr, 
            qk1_ptr, 
            v1_ptr, 
            qk2_ptr, 
            v2_ptr, 
            L_ptr,
            m_ptr,
            a_ptr,
            dLda_ptr,
            dLdqk0_ptr, 
            dLdv0_ptr,
            dLdqk1_ptr, 
            dLdv1_ptr,
            dLdqk2_ptr, 
            dLdv2_ptr
        );
    }

    TORCH_LIBRARY(rtdenoise, m) {
        m.def(
            "kernel_attn("
                "Tensor qk0, Tensor v0, "
                "Tensor? qk1, Tensor? v1, "
                "Tensor? qk2, Tensor? v2, "
                "int kernel_size, int skip_center,"
                "Tensor? L, Tensor? m"
            ") -> Tensor"
        );

        m.def(
            "kernel_attn_bwd("
                "Tensor qk0, Tensor v0, "
                "Tensor? qk1, Tensor? v1, "
                "Tensor? qk2, Tensor? v2, "
                "int kernel_size, int skip_center,"
                "Tensor L, Tensor m, "
                "Tensor a, "
                "Tensor dLda, "
                "Tensor dLdqk0, Tensor dLdv0,"
                "Tensor? dLdqk1, Tensor? dLdv1,"
                "Tensor? dLdqk2, Tensor? dLdv2"
            ") -> ()"
        );
    } 

    TORCH_LIBRARY_IMPL(rtdenoise, CPU, m) {
        m.impl("kernel_attn", &kernel_attn_cpu);
        m.impl("kernel_attn_bwd", &kernel_attn_bwd_cpu);
    }

}