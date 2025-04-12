#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm> // for std::max
#include <cmath> // for std::isnan

namespace rtdenoise {

    template<typename T>
    struct UpscaleAttnResultCUDA {
        T accum_o[3] {0, 0, 0};
        T running_sum = 0;
        T logit_max = -9999;
    };

    template<typename T>
    __device__
    T calc_bias(const int scale_power, const int y, const int x, int i, int j) {
        T bias_factor = -3;

        // calculate center position of this pixel
        i = (i << scale_power) + (1 << (scale_power - 1));
        j = (j << scale_power) + (1 << (scale_power - 1));

        // try example to verify math
        // full res is 1024
        // downsampled is 32
        // each pixel occupies 32 pixels
        // scale power is 5
        // suppose we are at 16 16
        // we are then sampling pixel 0 0 
        // offten by 2^4
        // get 16 16
        // right in the center

        int offset = max(abs(y - i), abs(x - j));
        T soff = (T)offset / (1 << scale_power);

        T bias = bias_factor * soff;

        return bias;
    }

    template<typename T> 
    __device__
    UpscaleAttnResultCUDA<T> upscale_attn_cuda_pixel(
        const int C, 
        const int HU,
        const int WU,
        const int HD,
        const int WD,
        const int kernel_size,
        const int scale_power,
        const int y,
        const int x,
        const T* q,
        const T* k,
        const T* v,
        const T* b
    ) {
        UpscaleAttnResultCUDA<T> res;

        int hks = kernel_size / 2;
        T scale_factor = 1 / sqrt(C);

        // calculate where in the downsampled image this pixel is
        int yd = y >> scale_power;
        int xd = x >> scale_power;
        
        int bidx = 0;
        for(int i = yd - hks; i <= yd + hks; i++) {
            for(int j = xd - hks; j <= xd + hks; j++) {

                bool is_inside = (i >= 0 && j >= 0 && i < HD && j < WD);

                T logit = 0;
                if(is_inside) {
                    for(int c = 0; c < C; c++) {
                        logit += q[c * HU * WU + y * WU + x] * k[c * HD * WD + i * WD + j];
                    }
                }

                // scale logits by sqrt dim
                logit = logit * scale_factor + b[bidx++ * HU * WU + y * WU + x];

                T new_logit_max = std::max(logit, res.logit_max);
                T weight = exp(logit - new_logit_max);
                T sum_modulation = exp(res.logit_max - new_logit_max);

                for(int k = 0; k < 3; k++) {
                    T v_read = (is_inside ? v[k * HD * WD + i * WD + j] : 0);

                    res.accum_o[k] = (sum_modulation * res.accum_o[k] + weight * v_read);
                }

                res.running_sum = sum_modulation * res.running_sum + weight;
                res.logit_max = new_logit_max;

            }
        }

        for(int c = 0; c < 3; c++) {
            res.accum_o[c] /= res.running_sum;
        }

        return res;
    }


    template<typename T>
    __global__
    void upscale_attn_cuda_impl(
        const int N,
        const int C, 
        const int HU,
        const int WU,
        const int HD,
        const int WD,
        const int kernel_size,
        const int scale_power,
        const T* q,
        const T* k,
        const T* v,
        const T* b, 
        T* o,
        T* L,
        T* m
    ) {

        // convert thread indices to image indices
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int n = blockIdx.z * blockDim.z + threadIdx.z;

        if(y >= HU || x >= WU || n >= N) {
            return;
        }

        auto res = upscale_attn_cuda_pixel<T>(
            C,
            HU,
            WU,
            HD,
            WD,
            kernel_size,
            scale_power,
            y,
            x,
            &q[n * C * HU * WU],
            &k[n * C * HD * WD],
            &v[n * 3 * HD * WD],
            &b[n * 9 * HU * WU]
        );

        for(int c = 0; c < 3; c++) {
            o[n * 3 * HU * WU + c * HU * WU + y * WU + x] = res.accum_o[c];
        }

        L[n * HU * WU + y * WU + x] = res.running_sum;
        m[n * HU * WU + y * WU + x] = res.logit_max;


    }

    at::Tensor upscale_attn_cuda(
        at::Tensor q,
        at::Tensor k,
        at::Tensor v,
        at::Tensor b,
        int64_t kernel_size,
        int64_t scale_power,
        at::Tensor L,
        at::Tensor m
    ) {

        // dimension checks
        TORCH_CHECK(k.size(0) == v.size(0));

        TORCH_CHECK(q.size(1) == k.size(1));
        TORCH_CHECK(3         == v.size(1));
        TORCH_CHECK(9         == b.size(1));

        TORCH_CHECK(k.size(2) == v.size(2));
        TORCH_CHECK(k.size(3) == v.size(3));

        TORCH_CHECK(q.size(2) == b.size(2));
        TORCH_CHECK(q.size(3) == b.size(3));

        // parameter checks
        TORCH_CHECK(kernel_size % 2 == 1);
        //TORCH_CHECK(scale_power > 0);

        TORCH_CHECK(q.size(2) >> scale_power == k.size(2));
        TORCH_CHECK(q.size(3) >> scale_power == k.size(3));
        
        TORCH_CHECK(L.size(0) == q.size(0));
        TORCH_CHECK(L.size(1) == 1        );
        TORCH_CHECK(L.size(2) == q.size(2));
        TORCH_CHECK(L.size(3) == q.size(3));

        TORCH_CHECK(m.size(0) == q.size(0));
        TORCH_CHECK(m.size(1) == 1        );
        TORCH_CHECK(m.size(2) == q.size(2));
        TORCH_CHECK(m.size(3) == q.size(3));


        // assert everything is on the CUDA
        TORCH_INTERNAL_ASSERT(q.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(k.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(v.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);

        auto qc = q.contiguous();
        auto kc = k.contiguous();
        auto vc = v.contiguous();
        auto bc = b.contiguous();

        auto o = torch::empty({q.size(0), 3, q.size(2), q.size(3)}, q.options()); 

        int N = q.size(0);
        int C = q.size(1);
        int HU = q.size(2);
        int WU = q.size(3);

        dim3 num_blocks((WU + 31) / 32, (HU + 31) / 32, N);
        dim3 block_size(32, 32, 1);

        upscale_attn_cuda_impl<float><<<num_blocks, block_size>>>(
            q.size(0),
            q.size(1),
            q.size(2),
            q.size(3),
            k.size(2),
            k.size(3),
            kernel_size,
            scale_power,
            qc.data_ptr<float>(),
            kc.data_ptr<float>(),
            vc.data_ptr<float>(),
            bc.data_ptr<float>(),
            o.data_ptr<float>(),
            L.data_ptr<float>(),
            m.data_ptr<float>()
        );

        return o;
    }

    template<typename T> 
    __device__
    void upscale_attn_bwd_cuda_pixel(
        const int C, 
        const int HU,
        const int WU,
        const int HD,
        const int WD,
        const int kernel_size,
        const int scale_power,
        const int y,
        const int x,
        const T* q,
        const T* k,
        const T* v,
        const T* b,
        const T* dLdo,
        T* dLdq,
        T* dLdk,
        T* dLdv,
        T* dLdb,
        UpscaleAttnResultCUDA<T> res
    ) {
        

        int hks = kernel_size / 2;
        T scale_factor = 1 / sqrt(C);

        // calculate where in the downsampled image this pixel is
        int yd = y >> scale_power;
        int xd = x >> scale_power;
        
        T local_dLda[3] {0, 0, 0};
        T D = 0;
        for(int k = 0; k < 3; k++) {
            local_dLda[k] = dLdo[k * HU * WU + y * WU + x];
            D += local_dLda[k] * res.accum_o[k];
        }

        int bidx = 0;
        for(int i = yd - hks; i <= yd + hks; i++) {
            for(int j = xd - hks; j <= xd + hks; j++) {

                bool is_inside = (i >= 0 && j >= 0 && i < HD && j < WD);

                T dLdweight = -D;
                T logit = 0;

                if(is_inside) {
                    for(int k = 0; k < 3; k++) {
                        dLdweight += local_dLda[k] * v[k * HD * WD + i * WD + j];
                    }

                    for(int c = 0; c < C; c++) {
                        logit += q[c * HU * WU + y * WU + x] * k[c * HD * WD + i * WD + j];
                    }
                }

                // scale logits by sqrt dim
                int bic = bidx++;
                logit = logit * scale_factor + b[bic * HU * WU + y * WU + x];

                T weight = exp(logit - res.logit_max) / res.running_sum;

                T dLdlogit = dLdweight * weight;

                dLdb[bic * HU * WU + y * WU + x] = dLdlogit;

                dLdlogit *= scale_factor;

                // compute query and key derivatives
                for(int c = 0; c < C; c++) {
                    T k_read = (is_inside ? k[c * HD * WD + i * WD + j] : 0);

                    dLdq[c * HU * WU + y * WU + x] += dLdlogit * k_read;

                    if(is_inside) {
                        atomicAdd(&dLdk[c * HD * WD + i * WD + j], dLdlogit * q[c * HU * WU + y * WU + x]);
                    }

                }

                if(is_inside) {
                    
                    for(int k = 0; k < 3; k++) {
                        atomicAdd(&dLdv[k * HD * WD + i * WD + j], weight * local_dLda[k]);
                    }
                }

            }
        }
    }

    template<typename T>
    __global__
    void upscale_attn_bwd_cuda_impl(
        const int N,
        const int C, 
        const int HU,
        const int WU,
        const int HD,
        const int WD,
        const int kernel_size,
        const int scale_power,
        const T* q,
        const T* k,
        const T* v,
        const T* b, 
        const T* o,
        const T* L,
        const T* m,
        const T* dLdo,
        T* dLdq,
        T* dLdk,
        T* dLdv,
        T* dLdb
    ) {
        // convert thread indices to image indices
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int n = blockIdx.z * blockDim.z + threadIdx.z;

        if(y >= HU || x >= WU || n >= N) {
            return;
        }

        UpscaleAttnResultCUDA<T> res;

        // load res into memory
        for(int c = 0; c < 3; c++) {
            res.accum_o[c] = o[n * 3 * HU * WU + c * HU * WU + y * WU + x];
        }

        res.running_sum = L[n * HU * WU + y * WU + x];
        res.logit_max = m[n * HU * WU + y * WU + x];

        upscale_attn_bwd_cuda_pixel<T>(
            C,
            HU,
            WU,
            HD,
            WD,
            kernel_size,
            scale_power,
            y,
            x,
            &q[n * C * HU * WU],
            &k[n * C * HD * WD],
            &v[n * 3 * HD * WD],
            &b[n * 9 * HU * WU],
            &dLdo[n * 3 * HU * WU],
            &dLdq[n * C * HU * WU],
            &dLdk[n * C * HD * HD],
            &dLdv[n * 3 * HD * HD],
            &dLdb[n * 9 * HU * WU],
            res
        );



    }

    at::Tensor upscale_attn_bwd_cuda(
        at::Tensor q,
        at::Tensor k,
        at::Tensor v,
        at::Tensor b,
        int64_t kernel_size,
        int64_t scale_power,
        at::Tensor o,
        at::Tensor L,
        at::Tensor m,
        at::Tensor dLdo,
        at::Tensor dLdq,
        at::Tensor dLdk,
        at::Tensor dLdv,
        at::Tensor dLdb
    ) {

        // dimension checks
        TORCH_CHECK(k.size(0) == v.size(0));

        TORCH_CHECK(q.size(1) == k.size(1));
        TORCH_CHECK(3         == v.size(1));
        TORCH_CHECK(9         == b.size(1));

        TORCH_CHECK(k.size(2) == v.size(2));
        TORCH_CHECK(k.size(3) == v.size(3));

        TORCH_CHECK(q.size(2) == b.size(2));
        TORCH_CHECK(q.size(3) == b.size(3));

        // parameter checks
        TORCH_CHECK(kernel_size % 2 == 1);
        //TORCH_CHECK(scale_power > 0);

        TORCH_CHECK(q.size(2) >> scale_power == k.size(2));
        TORCH_CHECK(q.size(3) >> scale_power == k.size(3));
        

        // assert everything is on the CUDA
        TORCH_INTERNAL_ASSERT(q.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(k.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(v.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);

        TORCH_INTERNAL_ASSERT(o.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(L.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(m.device().type() == at::DeviceType::CUDA);

        auto qc = q.contiguous();
        auto kc = k.contiguous();
        auto vc = v.contiguous();
        auto bc = b.contiguous();
        auto oc = o.contiguous();
        auto Lc = L.contiguous();
        auto mc = m.contiguous();
        auto dLdoc = dLdo.contiguous();


        int N = q.size(0);
        int C = q.size(1);
        int HU = q.size(2);
        int WU = q.size(3);

        dim3 num_blocks((WU + 31) / 32, (HU + 31) / 32, N);
        dim3 block_size(32, 32, 1);

        upscale_attn_bwd_cuda_impl<float><<<num_blocks, block_size>>>(
            q.size(0),
            q.size(1),
            q.size(2),
            q.size(3),
            k.size(2),
            k.size(3),
            kernel_size,
            scale_power,
            qc.data_ptr<float>(),
            kc.data_ptr<float>(),
            vc.data_ptr<float>(),
            bc.data_ptr<float>(),
            oc.data_ptr<float>(),
            Lc.data_ptr<float>(),
            mc.data_ptr<float>(),
            dLdoc.data_ptr<float>(),
            dLdq.data_ptr<float>(),
            dLdk.data_ptr<float>(),
            dLdv.data_ptr<float>(),
            dLdb.data_ptr<float>()
        );

        return o;
    }

    TORCH_LIBRARY_IMPL(rtdenoise, CUDA, m) {
        m.impl("upscale_attn", &upscale_attn_cuda);
        m.impl("upscale_attn_bwd", &upscale_attn_bwd_cuda);
    }

}