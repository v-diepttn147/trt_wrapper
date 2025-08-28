#include "trt_wrapper/preprocess_cuda.h"

__global__ void nhwc_to_nchw_norm_kernel(const unsigned char* __restrict__ src,
                                               int H, int W, size_t stepBytes,
                                               float* __restrict__ dst,
                                               int rIdx, int gIdx, int bIdx,
                                               float scale,     // 1/255
                                               float m0, float m1, float m2,
                                               float s0, float s1, float s2)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // W
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // H
    if (x >= W || y >= H) return;

    const unsigned char* row = src + y * stepBytes;
    int base = x * 3;
    float r = static_cast<float>(row[base + rIdx]) * scale;
    float g = static_cast<float>(row[base + gIdx]) * scale;
    float b = static_cast<float>(row[base + bIdx]) * scale;

    int hw = H * W;
    int o  = y * W + x;
    dst[0 * hw + o] = (r - m0) / s0;
    dst[1 * hw + o] = (g - m1) / s1;
    dst[2 * hw + o] = (b - m2) / s2;

}

void launch_nhwc_to_nchw_norm(const unsigned char* d_srcU8NHWC,
                                    int H, int W, size_t srcStepBytes,
                                    float* d_dstNCHW,
                                    bool swapRB,
                                    float scale,
                                    const float mean[3],
                                    const float stdv[3],
                                    cudaStream_t stream)
{
    // channel indices in the NHWC source buffer
    int rIdx = swapRB ? 2 : 0; // src is BGR -> RGB if swapRB=true
    int gIdx = 1;
    int bIdx = swapRB ? 0 : 2;

    dim3 block(32, 16);
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y);

    nhwc_to_nchw_norm_kernel<<<grid, block, 0, stream>>>(
        d_srcU8NHWC, H, W, srcStepBytes,
        d_dstNCHW,
        rIdx, gIdx, bIdx,
        scale,
        mean[0], mean[1], mean[2],
        stdv[0], stdv[1], stdv[2]);
}
