#pragma once
// OpenDRTv11Runners.h - GPU host entry points for the second OpenDRT model
// ("model 1", core/OpenDRTv11Algorithm.h). Same guards, argument order and
// buffer conventions as the model 0 runners declared in OpenDRT.h. Not wired
// into the plugin yet.
//
// Every runner expects p_Params already filled by opendrt_v11_fill_presets (or
// set field by field) and then passed through opendrt_v11_precompute. y is the
// raw buffer row index, exactly as the model 0 backends pass it.

#include "core/OpenDRTv11Params.h"

#ifdef LDT_HAS_CUDA
// p_Stream is a cudaStream_t; p_Input / p_Output are device pointers at the
// render rect origin. Launches and returns (the host owns stream sync).
void RunOpenDRTv11CudaKernel(void* p_Stream, int p_Width, int p_Height,
                             int p_SrcRowBytes, int p_DstRowBytes,
                             const OpenDRTv11Params* p_Params,
                             const float* p_Input, float* p_Output);
#endif

#ifdef LDT_HAS_OPENCL
// p_CmdQ is a cl_command_queue, p_Input / p_Output are cl_mem handles from
// getPixelData. p_SrcOffX / p_SrcOffY place the render rect inside the source,
// p_SrcW / p_SrcH give the source size (-1: covers the rect at offset 0).
// Returns false only when neither the kernel nor the CPU fallback rendered.
bool RunOpenDRTv11OpenCLKernel(void* p_CmdQ, int p_Width, int p_Height,
                               int p_SrcRowBytes, int p_DstRowBytes,
                               const OpenDRTv11Params* p_Params,
                               const float* p_Input, float* p_Output,
                               int p_SrcOffX = 0, int p_SrcOffY = 0,
                               int p_SrcW = -1, int p_SrcH = -1);
#endif

#ifdef __APPLE__
// p_CmdQ is an id<MTLCommandQueue>; p_Input / p_Output are id<MTLBuffer>
// handles cast to float*. Blocks until the command buffer completes.
void RunOpenDRTv11MetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                              int p_SrcRowBytes, int p_DstRowBytes,
                              const OpenDRTv11Params* p_Params,
                              const float* p_Input, float* p_Output);
#endif
