#pragma once
// AltDrtRunners.h - GPU host entry points for the two alternate 6A display
// transforms, Tesseract (core/TesseractAlgorithm.h) and Chromagnon
// (core/ChromagnonAlgorithm.h). Same guards, argument order and buffer
// conventions as the OpenDRT runners declared in OpenDRT.h. OpenDRT.cpp calls
// them from render() when Model is Tesseract or Chromagnon.
//
// Every runner expects p_Params already filled (tesseract_defaults /
// chromagnon_defaults or field by field) and then passed through
// tesseract_precompute / chromagnon_precompute. y is the raw buffer row index,
// exactly as the OpenDRT backends pass it (neither model reads x or y).

#include "core/TesseractParams.h"
#include "core/ChromagnonParams.h"

#ifdef LDT_HAS_CUDA
// p_Stream is a cudaStream_t; p_Input / p_Output are device pointers at the
// render rect origin. Launches and returns (the host owns stream sync).
void RunTesseractCudaKernel(void* p_Stream, int p_Width, int p_Height,
                            int p_SrcRowBytes, int p_DstRowBytes,
                            const TesseractParams* p_Params,
                            const float* p_Input, float* p_Output);
void RunChromagnonCudaKernel(void* p_Stream, int p_Width, int p_Height,
                             int p_SrcRowBytes, int p_DstRowBytes,
                             const ChromagnonParams* p_Params,
                             const float* p_Input, float* p_Output);
#endif

#ifdef LDT_HAS_OPENCL
// p_CmdQ is a cl_command_queue, p_Input / p_Output are cl_mem handles from
// getPixelData. p_SrcOffX / p_SrcOffY place the render rect inside the source,
// p_SrcW / p_SrcH give the source size (-1: covers the rect at offset 0).
// Returns false only when neither the kernel nor the CPU fallback rendered.
bool RunTesseractOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height,
                              int p_SrcRowBytes, int p_DstRowBytes,
                              const TesseractParams* p_Params,
                              const float* p_Input, float* p_Output,
                              int p_SrcOffX = 0, int p_SrcOffY = 0,
                              int p_SrcW = -1, int p_SrcH = -1);
bool RunChromagnonOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height,
                               int p_SrcRowBytes, int p_DstRowBytes,
                               const ChromagnonParams* p_Params,
                               const float* p_Input, float* p_Output,
                               int p_SrcOffX = 0, int p_SrcOffY = 0,
                               int p_SrcW = -1, int p_SrcH = -1);
#endif

#ifdef __APPLE__
// p_CmdQ is an id<MTLCommandQueue>; p_Input / p_Output are id<MTLBuffer>
// handles cast to float*. Blocks until the command buffer completes.
void RunTesseractMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                             int p_SrcRowBytes, int p_DstRowBytes,
                             const TesseractParams* p_Params,
                             const float* p_Input, float* p_Output);
void RunChromagnonMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                              int p_SrcRowBytes, int p_DstRowBytes,
                              const ChromagnonParams* p_Params,
                              const float* p_Input, float* p_Output);
#endif
