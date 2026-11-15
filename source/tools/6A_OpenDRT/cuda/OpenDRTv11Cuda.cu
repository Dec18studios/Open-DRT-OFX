// OpenDRTv11Cuda.cu - CUDA backend for the second OpenDRT model ("model 1").
// Not wired into the plugin yet.
//
// Compiles the CPU core header with __CUDACC__ active so every LDT_FUNC becomes
// __device__ inline: the device path is the core itself, no twin. The host
// entry uploads the already precomputed params block to constant memory,
// launches a 16x16 grid and returns (the OFX host owns stream sync). Row pitch
// is honored in float4 units so padded rows work, matching the Metal path.
// y is the raw buffer row index, as in model 0.

#include "../core/OpenDRTv11Algorithm.h"

#include <cuda_runtime.h>

// 376 bytes, far under the 64 KB constant bank. Distinct from model 0 c_Params.
__constant__ OpenDRTv11Params c_OpenDRTv11Params;

__global__ void OpenDRTv11Kernel(int p_Width, int p_Height,
                                 int p_SrcPitch, int p_DstPitch,
                                 const float* __restrict__ p_Input,
                                 float* __restrict__ p_Output)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p_Width || y >= p_Height) return;

    const int srcIdx = (y * p_SrcPitch + x) * 4;
    const int dstIdx = (y * p_DstPitch + x) * 4;

    const float3 in_ = make_float3(p_Input[srcIdx + 0],
                                   p_Input[srcIdx + 1],
                                   p_Input[srcIdx + 2]);
    const float a = p_Input[srcIdx + 3];

    const float3 out = opendrt_v11_processPixel(in_, x, y, p_Width, p_Height, &c_OpenDRTv11Params);

    p_Output[dstIdx + 0] = out.x;
    p_Output[dstIdx + 1] = out.y;
    p_Output[dstIdx + 2] = out.z;
    p_Output[dstIdx + 3] = a;
}

void RunOpenDRTv11CudaKernel(void* p_Stream, int p_Width, int p_Height,
                             int p_SrcRowBytes, int p_DstRowBytes,
                             const OpenDRTv11Params* p_Params,
                             const float* p_Input, float* p_Output)
{
    if (p_Params == nullptr || p_Input == nullptr || p_Output == nullptr) return;
    if (p_Width <= 0 || p_Height <= 0) return;

    cudaStream_t stream = static_cast<cudaStream_t>(p_Stream);

    cudaMemcpyToSymbolAsync(c_OpenDRTv11Params, p_Params, sizeof(OpenDRTv11Params),
                            0, cudaMemcpyHostToDevice, stream);

    const int srcPitch = p_SrcRowBytes / 16;
    const int dstPitch = p_DstRowBytes / 16;

    dim3 threads(16, 16, 1);
    dim3 blocks((p_Width  + threads.x - 1) / threads.x,
                (p_Height + threads.y - 1) / threads.y,
                1);

    OpenDRTv11Kernel<<<blocks, threads, 0, stream>>>(
        p_Width, p_Height, srcPitch, dstPitch, p_Input, p_Output);
}
