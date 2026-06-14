// OpenDRTCuda.cu — CUDA backend for "6A. OpenDRT".
//
// Compiles the shared algorithm header with __CUDACC__ active so every LDT_FUNC
// becomes __device__ inline. The host entry uploads the (already-precomputed)
// params struct to device constant memory, launches the grid, and returns (the
// OFX host manages stream sync). Row pitch is honored explicitly (in float4
// units) so Resolve's padded rows are handled — matching the Metal path.
//
// NOTE: the host (OpenDRT.cpp render()) has ALREADY called opendrt_precompute()
// on p_Params before this entry — the display-encoding remap and ts_* constants
// are filled. This file only uploads + dispatches; it never re-derives anything.
// Do NOT copy the canonical CudaKernel.cu host logic (it is the known-broken file
// that never linked, left tonescale all-zero, and dropped two feature blocks).

#include "../core/OpenDRTAlgorithm.h"

#include <cuda_runtime.h>

// ── Constant memory params (~1.6 KB OpenDRTParams, well under the 64 KB cap) ────
__constant__ OpenDRTParams c_Params;

// ── Per-pixel kernel ──────────────────────────────────────────────────────────
__global__ void OpenDRTKernel(int p_Width, int p_Height,
                              int p_SrcPitch, int p_DstPitch,
                              const float* __restrict__ p_Input,
                              float* __restrict__ p_Output)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= p_Width || y >= p_Height) return;

    // Pitches are in float4 units (rowBytes/16); ×4 to index the float* arrays.
    const int srcIdx = (y * p_SrcPitch + x) * 4;
    const int dstIdx = (y * p_DstPitch + x) * 4;

    float3 in_ = make_float3(p_Input[srcIdx + 0],
                             p_Input[srcIdx + 1],
                             p_Input[srcIdx + 2]);
    float  a   = p_Input[srcIdx + 3];

    float3 out = opendrt_processPixel(in_, x, y, p_Width, p_Height, &c_Params);

    p_Output[dstIdx + 0] = out.x;
    p_Output[dstIdx + 1] = out.y;
    p_Output[dstIdx + 2] = out.z;
    p_Output[dstIdx + 3] = a;
}

// ── Host entry point ──────────────────────────────────────────────────────────
void RunOpenDRTCudaKernel(void* p_Stream, int p_Width, int p_Height,
                          int p_SrcRowBytes, int p_DstRowBytes,
                          const OpenDRTParams* p_Params,
                          const float* p_Input, float* p_Output)
{
    cudaStream_t stream = static_cast<cudaStream_t>(p_Stream);

    cudaMemcpyToSymbolAsync(c_Params, p_Params, sizeof(OpenDRTParams),
                            0, cudaMemcpyHostToDevice, stream);

    // rowBytes -> float4 stride (16 bytes per float4).
    const int srcPitch = p_SrcRowBytes / 16;
    const int dstPitch = p_DstRowBytes / 16;

    dim3 threads(16, 16, 1);
    dim3 blocks((p_Width  + threads.x - 1) / threads.x,
                (p_Height + threads.y - 1) / threads.y,
                1);

    OpenDRTKernel<<<blocks, threads, 0, stream>>>(
        p_Width, p_Height, srcPitch, dstPitch, p_Input, p_Output);
}
