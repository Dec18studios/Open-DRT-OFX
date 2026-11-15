#pragma once
// LdtOpenCL.h - shared OpenCL runtime for the Look Dev Tools suite.
//
// Every tool's OpenCL backend is the same shape: one pointwise kernel, one
// packed parameter block, src and dst as whole OFX images. This header is the
// whole surface the tools see. It deliberately includes no OpenCL header, so
// tool sources compile the same with or without an OpenCL SDK; only
// LdtOpenCL.cpp talks to the API, and it loads the ICD loader at run time with
// dlopen / LoadLibrary (the plugin never links OpenCL).
//
// Device pointers. With pOpenCLCmdQ the host hands over cl_mem handles, not
// addresses, so they can never be offset (getPixelAddress is meaningless) and
// the host must never read or write through them. Tools pass getPixelData()
// of both images, and say where the render rect sits inside the source with
// srcOffX / srcOffY (dst bounds minus src bounds) and how big the source is
// with srcW / srcH (-1 means "covers the render rect at offset 0").
//
// Guard. The backend compiles only with LDT_HAS_OPENCL on a non-Apple target.
// LDT_OPENCL_SYNTAX_CHECK lifts the Apple exclusion so syntax-check-opencl.sh
// can type-check the real code against the macOS OpenCL 1.2 headers; CMake
// never sets either on Apple, so the Mac plugin carries no OpenCL code at all.

#include <cstddef>

#if defined(LDT_HAS_OPENCL) && (!defined(__APPLE__) || defined(LDT_OPENCL_SYNTAX_CHECK))
  #define LDT_OPENCL_ENABLED 1
#else
  #define LDT_OPENCL_ENABLED 0
#endif

#if LDT_OPENCL_ENABLED

#include <algorithm>
#include <thread>
#include <vector>

namespace ldt {
namespace opencl {

// True when an ICD loader with every required entry point was found.
bool Available();

// True when the loader is available and the queue answers clGetCommandQueueInfo.
bool QueueUsable(void* p_CmdQ);

// stderr plus %TEMP%\lookdevtools_opencl.log (Windows) or
// /tmp/lookdevtools_opencl.log (elsewhere). Identical consecutive lines fold.
void Diag(const char* fmt, ...);

// Runs one pointwise kernel over a w x h render rect.
//
//   tool        cache key, one program per (queue, tool)
//   source      full OpenCL C source (the embedded .cl)
//   kernelName  __kernel void name(__constant P*, __global const float4*,
//                                  __global float4*, __constant int4* dims)
//               dims = {srcPitch, dstPitch, w, h} in float4 units
//   params      host copy of the parameter block, uploaded with a blocking write
//
// Returns true only when the kernel ran and the queue drained. On any failure
// the source is copied through to dst (clEnqueueCopyBufferRect) where both
// images overlap, the queue is drained, and false is returned so the caller
// can run its CPU fallback. Never throws.
bool RunPixelKernel(void* p_CmdQ, const char* p_Tool, const char* p_Source, const char* p_KernelName,
                    const void* p_Params, std::size_t p_ParamsBytes,
                    int p_Width, int p_Height, int p_SrcRowBytes, int p_DstRowBytes,
                    const float* p_Input, float* p_Output,
                    int p_SrcOffX = 0, int p_SrcOffY = 0, int p_SrcW = -1, int p_SrcH = -1);

// Device to device copy of the w x h render rect (src at srcOff, dst at 0,0).
// Pixels the source does not cover are left as they are. Drains the queue.
bool CopyImage(void* p_CmdQ, int p_Width, int p_Height, int p_SrcRowBytes, int p_DstRowBytes,
               const float* p_Input, float* p_Output,
               int p_SrcOffX = 0, int p_SrcOffY = 0, int p_SrcW = -1, int p_SrcH = -1);

// Device to host: the w x h rect at (offX, offY) of a device image with
// rowBytes pitch, into tightly packed RGBA rows (w * h * 4 floats). Pixels
// outside the device image (when srcW / srcH say it is smaller) read as
// transparent black. Blocking; drains the queue.
bool ReadImage(void* p_CmdQ, const float* p_DeviceBuf, int p_Width, int p_Height, int p_RowBytes,
               float* p_HostTight, int p_OffX = 0, int p_OffY = 0, int p_SrcW = -1, int p_SrcH = -1);

// Host to device: tightly packed RGBA rows into the w x h rect at 0,0 of a
// device image with rowBytes pitch. Blocking; drains the queue.
bool WriteImage(void* p_CmdQ, float* p_DeviceBuf, int p_Width, int p_Height, int p_RowBytes,
                const float* p_HostTight);

// CPU fallback for a failed kernel. Reads the source rect back, runs
// fn(const float* inRGBA, float* outRGBA, int x, int y) per pixel on all
// cores (x, y are 0 based in the render rect, the same coordinates the
// kernel sees), and writes the result to dst. With no source the input is
// (0, 0, 0, 1), as in the tools' CPU processors. If the source cannot be read
// back, dst keeps whatever RunPixelKernel's passthrough put there. Never throws.
template <class Fn>
bool CpuFallback(void* p_CmdQ, int p_Width, int p_Height, int p_SrcRowBytes, int p_DstRowBytes,
                 const float* p_Input, float* p_Output,
                 int p_SrcOffX, int p_SrcOffY, int p_SrcW, int p_SrcH, Fn&& fn)
{
    if (p_CmdQ == nullptr || p_Output == nullptr || p_Width <= 0 || p_Height <= 0) {
        return false;
    }
    try {
        const std::size_t count = (std::size_t)p_Width * (std::size_t)p_Height * 4;
        std::vector<float> in(count, 0.0f);
        std::vector<float> out(count, 0.0f);
        if (p_Input != nullptr) {
            if (!ReadImage(p_CmdQ, p_Input, p_Width, p_Height, p_SrcRowBytes, in.data(),
                           p_SrcOffX, p_SrcOffY, p_SrcW, p_SrcH)) {
                Diag("LDT OpenCL: CPU fallback could not read the source back; keeping the passthrough.\n");
                return false;
            }
        } else {
            for (std::size_t i = 3; i < count; i += 4) {
                in[i] = 1.0f;
            }
        }

        const int width = p_Width;
        auto rows = [&](int y0, int y1) {
            for (int y = y0; y < y1; ++y) {
                const std::size_t row = (std::size_t)y * (std::size_t)width * 4;
                for (int x = 0; x < width; ++x) {
                    const std::size_t s = row + (std::size_t)x * 4;
                    fn(&in[s], &out[s], x, y);
                }
            }
        };

        unsigned hw = std::thread::hardware_concurrency();
        int threads = (int)std::max(1u, std::min(hw == 0 ? 1u : hw, 64u));
        threads = std::min(threads, p_Height);
        std::vector<std::thread> pool;
        const int band = (p_Height + threads - 1) / threads;
        int next = band;
        try {
            for (int t = 1; t < threads; ++t, next += band) {
                const int y0 = next;
                const int y1 = std::min(p_Height, next + band);
                if (y0 >= y1) {
                    break;
                }
                pool.emplace_back(rows, y0, y1);
            }
        } catch (...) {
            // Could not start a worker: this thread does the rest itself.
        }
        const int done = band * (int)(pool.size() + 1);
        rows(0, std::min(band, p_Height));
        if (done < p_Height) {
            rows(std::max(band, done), p_Height);
        }
        for (std::thread& th : pool) {
            th.join();
        }

        if (!WriteImage(p_CmdQ, p_Output, p_Width, p_Height, p_DstRowBytes, out.data())) {
            Diag("LDT OpenCL: CPU fallback could not write the result back.\n");
            return false;
        }
        Diag("LDT OpenCL: rendered on the CPU fallback (%dx%d).\n", p_Width, p_Height);
        return true;
    } catch (...) {
        Diag("LDT OpenCL: CPU fallback ran out of memory; keeping the passthrough.\n");
        return false;
    }
}

}  // namespace opencl
}  // namespace ldt

#endif  // LDT_OPENCL_ENABLED
