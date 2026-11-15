// ChromagnonOpenCL.cpp - OpenCL host entry point for the Chromagnon view transform (6A).
//
// Binds the embedded ChromagnonOpenCLKernels.cl (CMake turns it into the generated
// header ChromagnonOpenCLKernelSource.h) to the shared runtime in
// shared/LdtOpenCL.cpp, which owns the ICD loader, the per-queue program cache,
// staging and the passthrough. If the kernel cannot run, the same per-pixel
// function the CPU path uses renders the frame on the host instead, with the
// kernel's coordinates. Never throws into the OFX render action.
//
// Includes only the core header (not OpenDRT.h, which pulls in the OFX SDK).

#include "../core/ChromagnonAlgorithm.h"
#include "LdtOpenCL.h"
#include "../AltDrtRunners.h"

#if LDT_OPENCL_ENABLED

#include "ChromagnonOpenCLKernelSource.h"

static_assert(sizeof(ChromagnonParams) == 464, "update ChromagnonOpenCLKernels.cl too");

bool RunChromagnonOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height,
                               int p_SrcRowBytes, int p_DstRowBytes,
                               const ChromagnonParams* p_Params,
                               const float* p_Input, float* p_Output,
                               int p_SrcOffX, int p_SrcOffY, int p_SrcW, int p_SrcH)
{
    if (p_CmdQ == nullptr || p_Params == nullptr || p_Output == nullptr) return false;

    // The params block goes up exactly as the caller resolved it (same bytes
    // the Metal and CUDA hosts get).
    if (ldt::opencl::RunPixelKernel(p_CmdQ, "Chromagnon", ChromagnonOpenCLKernelSource(), "ChromagnonKernel",
                                    p_Params, sizeof(ChromagnonParams), p_Width, p_Height,
                                    p_SrcRowBytes, p_DstRowBytes, p_Input, p_Output,
                                    p_SrcOffX, p_SrcOffY, p_SrcW, p_SrcH)) {
        return true;
    }

    const ChromagnonParams params = *p_Params;
    const int pw = p_Width;
    const int ph = p_Height;
    return ldt::opencl::CpuFallback(
        p_CmdQ, p_Width, p_Height, p_SrcRowBytes, p_DstRowBytes, p_Input, p_Output,
        p_SrcOffX, p_SrcOffY, p_SrcW, p_SrcH,
        [&](const float* in, float* out, int x, int y) {
            const float3 px = make_float3(in[0], in[1], in[2]);
            const float3 r = chromagnon_processPixel(px, x, y, pw, ph, &params);
            out[0] = r.x;
            out[1] = r.y;
            out[2] = r.z;
            out[3] = in[3];
        });
}

#endif  // LDT_OPENCL_ENABLED
