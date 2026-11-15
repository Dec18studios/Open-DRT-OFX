// TesseractCuda.cu - CUDA backend for the Tesseract view transform (6A).
//
// Compiles the shared core header (core/TesseractAlgorithm.h) directly for the device, so
// the CUDA path has no hand-maintained twin: the per-pixel function is the very
// text the CPU path runs, with the LookDevPlatform.h float3 operators and
// TESS_EXP10F = exp10f under __CUDACC__. The params block is uploaded to a
// file-local __constant__ symbol (c_TessParams, distinct from the other 6A CUDA
// files) with an async copy on the host stream; the launch is queued on the
// same stream and NOT synchronised here (the host owns stream sync).
//
// Pitches arrive as rowBytes and are turned into RGBA pixel strides
// (rowBytes / 16); buffers are indexed (y * pitch + x) * 4 floats. Alpha passes
// through. The caller has already run the host precompute. C++17.

#include "../core/TesseractAlgorithm.h"

#include <cuda_runtime.h>

static __constant__ TesseractParams c_TessParams;

__global__ void TesseractKernel(int p_Width, int p_Height,
                                int p_SrcPitch, int p_DstPitch,
                                const float* __restrict__ p_Input,
                                float* __restrict__ p_Output)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p_Width || y >= p_Height) return;

    const int srcIdx = (y * p_SrcPitch + x) * 4;
    const int dstIdx = (y * p_DstPitch + x) * 4;

    float3 in_ = make_float3(p_Input[srcIdx + 0],
                             p_Input[srcIdx + 1],
                             p_Input[srcIdx + 2]);
    float  a   = p_Input[srcIdx + 3];

    float3 out = tesseract_processPixel(in_, x, y, p_Width, p_Height, &c_TessParams);

    p_Output[dstIdx + 0] = out.x;
    p_Output[dstIdx + 1] = out.y;
    p_Output[dstIdx + 2] = out.z;
    p_Output[dstIdx + 3] = a;
}

void RunTesseractCudaKernel(void* p_Stream, int p_Width, int p_Height,
                            int p_SrcRowBytes, int p_DstRowBytes,
                            const TesseractParams* p_Params,
                            const float* p_Input, float* p_Output)
{
    if (p_Params == nullptr || p_Input == nullptr || p_Output == nullptr) return;
    if (p_Width <= 0 || p_Height <= 0) return;

    cudaStream_t stream = static_cast<cudaStream_t>(p_Stream);

    cudaMemcpyToSymbolAsync(c_TessParams, p_Params, sizeof(TesseractParams),
                            0, cudaMemcpyHostToDevice, stream);

    const int srcPitch = p_SrcRowBytes / 16;
    const int dstPitch = p_DstRowBytes / 16;

    dim3 threads(16, 16, 1);
    dim3 blocks((p_Width  + threads.x - 1) / threads.x,
                (p_Height + threads.y - 1) / threads.y,
                1);

    TesseractKernel<<<blocks, threads, 0, stream>>>(
        p_Width, p_Height, srcPitch, dstPitch, p_Input, p_Output);
}
