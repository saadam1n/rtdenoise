#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <algorithm> // for std::max
#include <cmath> // for std::isnan

namespace rtdenoise {

    template<typename T>
    struct UpscaleAttnResultCPU {
        T accum_o[3] {0, 0, 0};
        T running_sum = 0;
        T logit_max = -9999;
    };

    template<typename T> 
    UpscaleAttnResultCPU<T> upscale_attn_cpu_pixel(
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
        UpscaleAttnResultCPU<T> res;

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

        return res;
    }


    template<typename T>
    void upscale_attn_cpu_impl(
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
        T* o
    ) {
        for(int n = 0; n < N; n++) {
            for(int y = 0; y < HU; y++) {
                for(int x = 0; x < WU; x++) {

                    auto res = upscale_attn_cpu_pixel<T>(
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
                        o[n * 3 * HU * WU + c * HU * WU + y * WU + x] = res.accum_o[c] / res.running_sum;
                    }

                }
            }
        }
    }

    at::Tensor upscale_attn_cpu(
        at::Tensor q,
        at::Tensor k,
        at::Tensor v,
        at::Tensor b,
        int64_t kernel_size,
        int64_t scale_power
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
        TORCH_CHECK(scale_power > 0);

        TORCH_CHECK(q.size(2) >> scale_power == k.size(2));
        TORCH_CHECK(q.size(3) >> scale_power == k.size(3));
        

        // assert everything is on the CPU
        TORCH_INTERNAL_ASSERT(q.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(k.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(v.device().type() == at::DeviceType::CPU);
        TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CPU);

        auto qc = q.contiguous();
        auto kc = k.contiguous();
        auto vc = v.contiguous();
        auto bc = b.contiguous();

        auto o = torch::empty({q.size(0), 3, q.size(2), q.size(3)}); 

        upscale_attn_cpu_impl(
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
            o.data_ptr<float>()
        );

        return o;
    }

    TORCH_LIBRARY_IMPL(rtdenoise, CPU, m) {
        m.impl("upscale_attn", &upscale_attn_cpu);
    }

}