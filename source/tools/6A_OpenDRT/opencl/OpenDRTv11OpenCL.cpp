// OpenDRTv11OpenCL.cpp - OpenCL host entry point for the second OpenDRT model
// ("model 1"). Not wired into the plugin yet.
//
// Binds the embedded OpenDRTv11OpenCLKernels.cl (CMake turns it into the
// generated header OpenDRTv11OpenCLKernelSource.h, symbol OpenDRTv11) to the
// shared runtime in shared/LdtOpenCL.cpp, which owns the ICD loader, the
// per-queue program cache, staging and the passthrough. If the kernel cannot
// run, the CPU core renders the frame on the host with the kernel's
// coordinates. Never throws into the OFX render action.

#include "../core/OpenDRTv11Algorithm.h"
#include "../OpenDRTv11Runners.h"
#include "LdtOpenCL.h"

#if LDT_OPENCL_ENABLED

#include "OpenDRTv11OpenCLKernelSource.h"

static_assert(sizeof(OpenDRTv11Params) == 94 * 4, "update OpenDRTv11OpenCLKernels.cl too");

bool RunOpenDRTv11OpenCLKernel(void* p_CmdQ, int p_Width, int p_Height,
                               int p_SrcRowBytes, int p_DstRowBytes,
                               const OpenDRTv11Params* p_Params,
                               const float* p_Input, float* p_Output,
                               int p_SrcOffX, int p_SrcOffY, int p_SrcW, int p_SrcH)
{
    if (p_CmdQ == nullptr || p_Params == nullptr || p_Output == nullptr) return false;

    if (ldt::opencl::RunPixelKernel(p_CmdQ, "OpenDRTv11", OpenDRTv11OpenCLKernelSource(), "OpenDRTv11Kernel",
                                    p_Params, sizeof(OpenDRTv11Params), p_Width, p_Height,
                                    p_SrcRowBytes, p_DstRowBytes, p_Input, p_Output,
                                    p_SrcOffX, p_SrcOffY, p_SrcW, p_SrcH)) {
        return true;
    }

    const OpenDRTv11Params params = *p_Params;
    const int pw = p_Width;
    const int ph = p_Height;
    return ldt::opencl::CpuFallback(
        p_CmdQ, p_Width, p_Height, p_SrcRowBytes, p_DstRowBytes, p_Input, p_Output,
        p_SrcOffX, p_SrcOffY, p_SrcW, p_SrcH,
        [&](const float* in, float* out, int x, int y) {
            const float3 px = make_float3(in[0], in[1], in[2]);
            const float3 r = opendrt_v11_processPixel(px, x, y, pw, ph, &params);
            out[0] = r.x;
            out[1] = r.y;
            out[2] = r.z;
            out[3] = in[3];
        });
}

#endif  // LDT_OPENCL_ENABLED
