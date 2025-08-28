#pragma once
#include <cuda_runtime_api.h>
void launch_nhwc_to_nchw_norm(const unsigned char* d_srcU8NHWC,
                                    int H, int W, size_t srcStepBytes,
                                    float* d_dstNCHW,
                                    bool swapRB,           // true if src is BGR and you want RGB
                                    float scale,           // usually 1.f/255.f
                                    const float mean[3],   // per-channel mean in [0..1]
                                    const float stdv[3],   // per-channel std in [0..1]
                                    cudaStream_t stream = nullptr);