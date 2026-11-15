// OpenDRTOpenCL.cpp - OpenCL host entry point for "6A. OpenDRT".
//
// Binds the embedded OpenDRTOpenCLKernels.cl (CMake turns it into the generated
// header OpenDRTOpenCLKernelSource.h) to the shared runtime in
// shared/LdtOpenCL.cpp, which owns the ICD loader, the per-queue program cache,
// staging and the passthrough. If the kernel cannot run, the same per-pixel
// function the CPU processor uses renders the frame on the host instead, with
// the kernel's coordinates. Never throws into the OFX render action.

#include "../OpenDRT.h"
#include "LdtOpenCL.h"

#if LDT_OPENCL_ENABLED

#include "OpenDRTOpenCLKernelSource.h"

static_assert(sizeof(OpenDRTParams) == 968, "update OpenDRTOpenCLKernels.cl too");

bool RunOpenDRTOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height,
                            int p_SrcRowBytes, int p_DstRowBytes,
                            const OpenDRTParams* p_Params,
                            const float* p_Input, float* p_Output,
                            int p_SrcOffX, int p_SrcOffY, int p_SrcW, int p_SrcH)
{
    if (p_CmdQ == nullptr || p_Params == nullptr || p_Output == nullptr) return false;

    // The params block goes up exactly as the tool filled it (same bytes the
    // Metal and CUDA hosts get).
    if (ldt::opencl::RunPixelKernel(p_CmdQ, "OpenDRT", OpenDRTOpenCLKernelSource(), "OpenDRTKernel",
                                    p_Params, sizeof(OpenDRTParams), p_Width, p_Height,
                                    p_SrcRowBytes, p_DstRowBytes, p_Input, p_Output,
                                    p_SrcOffX, p_SrcOffY, p_SrcW, p_SrcH)) {
        return true;
    }

    const OpenDRTParams params = *p_Params;
    const int pw = p_Width;
    const int ph = p_Height;
    return ldt::opencl::CpuFallback(
        p_CmdQ, p_Width, p_Height, p_SrcRowBytes, p_DstRowBytes, p_Input, p_Output,
        p_SrcOffX, p_SrcOffY, p_SrcW, p_SrcH,
        [&](const float* in, float* out, int x, int y) {
            const float3 px = make_float3(in[0], in[1], in[2]);
            const float3 r = opendrt_processPixel(px, x, y, pw, ph, &params);
            out[0] = r.x;
            out[1] = r.y;
            out[2] = r.z;
            out[3] = in[3];
        });
}

#endif  // LDT_OPENCL_ENABLED
