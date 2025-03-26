#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>

#include <stdlib.h>
#include <iostream>


namespace rtdenoise {

    template<typename T>
    struct KernelAttnResultGPU {
        T accum_a[3] {0, 0, 0};
        T running_sum = 0;
        T logit_max = -9999;
    };


    template<typename T>
    __device__ void kernel_attn_cuda_pixel(
        int C, int H, int W, 
        int kernel_size, 
        int y, int x,  
        const T* q, 
        const T* k, 
        const T* v, 
        KernelAttnResultGPU<T>& res
    ) {
        int hks = kernel_size / 2;
        T scale_factor = 1 / sqrt(C);

        for(int i = y - hks; i <= y + hks; i++) {
            for(int j = x - hks; j <= x + hks; j++) {

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
    __global__  void kernel_attn_cuda_impl(
        int N, int C, int H, int W, 
        int kernel_size, 
        const T* qk0, const T* v0, 
        const T* qk1, const T* v1, 
        const T* qk2, const T* v2, 
        T* a,
        T* L, T* m
    ) {
        // convert thread indices to image indices
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int n = blockIdx.z * blockDim.z + threadIdx.z;

        if(y >= H || x >= W || n >= N) {
            return;
        }

        const T* klist[] = {qk0, qk1, qk2};
        const T* vlist[] = {v0, v1, v2};
        
        KernelAttnResultGPU<T> res;

        for(int b = 0; b < 3; b++) {
            if(!klist[b]) {
                continue;
            }

            kernel_attn_cuda_pixel(
                C, H, W, 
                kernel_size, 
                y, x,
                &qk0[n * C * H * W],
                &klist[b][n * C * H * W],
                &vlist[b][n * 3 * H * W],
                res
            );
        }

        // these writes do not need atomic adds
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



    at::Tensor kernel_attn_cuda(
        at::Tensor qk0, at::Tensor v0, 
        c10::optional<at::Tensor> qk1, c10::optional<at::Tensor> v1,
        c10::optional<at::Tensor> qk2, c10::optional<at::Tensor> v2,
        int64_t kernel_size, 
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

        // ensure everything is on CUDA
        TORCH_INTERNAL_ASSERT(qk0.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(qk1.has_value() ? qk1.value().device().type() == at::DeviceType::CUDA : true);
        TORCH_INTERNAL_ASSERT(qk2.has_value() ? qk2.value().device().type() == at::DeviceType::CUDA : true);

        TORCH_INTERNAL_ASSERT(v0.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(v1.has_value() ? v1.value().device().type() == at::DeviceType::CUDA : true);
        TORCH_INTERNAL_ASSERT(v2.has_value() ? v2.value().device().type() == at::DeviceType::CUDA : true);

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

        int N = qk0.size(0);
        int C = qk0.size(1);
        int H = qk0.size(2);
        int W = qk0.size(3);

        dim3 num_blocks((H + 31) / 32, (W + 31) / 32, N);
        dim3 block_size(32, 32, 1);

        kernel_attn_cuda_impl<float><<<num_blocks, block_size>>>(
            qk0.size(0), 
            qk0.size(1), 
            qk0.size(2), 
            qk0.size(3), 
            kernel_size, 
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


    template<typename T>
    __device__ void kernel_attn_bwd_cuda_pixel(
        int C, int H, int W, int kernel_size, int y, int x,
        const T* q, const T* k, const T* v, 
        KernelAttnResultGPU<T>& res,
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

                // compute q, k, and v derivatives
                // these writes all need atomic adds
                for(int c = 0; c < C; c++) {
                    T k_read = (is_inside ? k[c * H * W + i * W + j] : 0);

                    atomicAdd(&dLdq[c * H * W + y * W + x], dLdlogit * k_read);

                    if(is_inside) {
                        atomicAdd(&dLdk[c * H * W + i * W + j], dLdlogit * q[c * H * W + y * W + x]);
                    }
                }

                if(is_inside) {
                    
                    for(int k = 0; k < 3; k++) {
                        atomicAdd(&dLdv[k * H * W + i * W + j], weight * local_dLda[k]);
                    }
                }

            }
        }

    }


    template<typename T>
    __global__ void kernel_attn_bwd_cuda_impl(
        int N, int C, int H, int W, int kernel_size, 
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
        // convert thread indices to image indices
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int n = blockIdx.z * blockDim.z + threadIdx.z;

        if(y >= H || x >= W || n >= N) {
            return;
        }

        const T* klist[] = {qk0, qk1, qk2};
        const T* vlist[] = {v0, v1, v2};

        T* dLklist[] = {dLdqk0, dLdqk1, dLdqk2};
        T* dLvlist[] = {dLdv0, dLdv1, dLdv2};


        KernelAttnResultGPU<T> res;

        for(int k = 0; k < 3; k++) {
            res.accum_a[k] = a[n * 3 * H * W + k * H * W + y * W + x];
        }
        res.running_sum = L[n * H * W + y * W + x];
        res.logit_max = m[n * H * W + y * W + x];

        for(int b = 0; b < 3; b++) {
            if(!klist[b]) {
                continue;
            }

            kernel_attn_bwd_cuda_pixel<T>(
                C, H, W, 
                kernel_size, 
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

    void kernel_attn_bwd_cuda(
        at::Tensor qk0, at::Tensor v0, 
        c10::optional<at::Tensor> qk1, c10::optional<at::Tensor> v1,
        c10::optional<at::Tensor> qk2, c10::optional<at::Tensor> v2,
        int64_t kernel_size, 
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

        // ensure everything is on CUDA
        TORCH_INTERNAL_ASSERT(qk0.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(qk1.has_value() ? qk1.value().device().type() == at::DeviceType::CUDA : true);
        TORCH_INTERNAL_ASSERT(qk2.has_value() ? qk2.value().device().type() == at::DeviceType::CUDA : true);

        TORCH_INTERNAL_ASSERT(v0.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(v1.has_value() ? v1.value().device().type() == at::DeviceType::CUDA : true);
        TORCH_INTERNAL_ASSERT(v2.has_value() ? v2.value().device().type() == at::DeviceType::CUDA : true);

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

        int N = qk0.size(0);
        int C = qk0.size(1);
        int H = qk0.size(2);
        int W = qk0.size(3);

        dim3 num_blocks((H + 31) / 32, (W + 31) / 32, N);
        dim3 block_size(32, 32, 1);

        kernel_attn_bwd_cuda_impl<float><<<num_blocks, block_size>>>(
            qk0.size(0), 
            qk0.size(1), 
            qk0.size(2), 
            qk0.size(3), 
            kernel_size, 
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

    TORCH_LIBRARY_IMPL(rtdenoise, CUDA, m) {
        m.impl("kernel_attn", &kernel_attn_cuda);
        m.impl("kernel_attn_bwd", &kernel_attn_bwd_cuda);
    }

}
