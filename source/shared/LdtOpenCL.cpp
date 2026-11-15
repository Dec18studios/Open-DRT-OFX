// LdtOpenCL.cpp - shared OpenCL runtime for the Look Dev Tools suite.
//
// Metal is the reference. Each tool's device source is a line for line port of
// its MSL kernel (tools/<tool>/opencl/<Tool>OpenCLKernels.cl, embedded at build
// time); this file is the dispatch around it, shared by every tool. It follows
// the Technicolor DRT OpenCL runtime:
//
//  1. The OpenCL ICD loader is opened with dlopen/LoadLibrary and never linked,
//     so a machine without one still loads the plugin.
//  2. Per command queue state (one program per tool, the kernel, its
//     work-group limit, the pooled buffers) is revalidated on every call: the
//     queue's context and device are re-read and a mismatch (a recycled queue
//     handle) drops and rebuilds everything.
//  3. A failed program build prints CL_PROGRAM_BUILD_LOG and is retried a few
//     times with a growing gap instead of being latched off for good.
//  4. On an out-of-order host queue, barriers order our commands after the
//     host's, and the passthrough after a failed dispatch.
//  5. Parameters and dims go up with BLOCKING writes, so no stack source is
//     referenced after the write call returns, and every exit path that
//     queued work ends in clFinish.
//  6. A failed dispatch never leaves dst untouched: the source is copied
//     through with clEnqueueCopyBufferRect and the caller then runs its CPU
//     fallback over it (see CpuFallback in LdtOpenCL.h).
//
// Build options are "-cl-std=CL1.2 -cl-fast-relaxed-math" and nothing else.
// Relaxed math lets the driver contract and reassociate the way Metal's
// default fast math does. The kernels use no native_* calls, and never any
// "-cl-mad-enable=false" spelling (not a valid option; AMD rejects the whole
// build over it). syntax-check-opencl.sh enforces both.

// windows.h (pulled in below for the loader) defines min/max macros that break
// std::min / std::max.
#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include "LdtOpenCL.h"

#if LDT_OPENCL_ENABLED

#define CL_TARGET_OPENCL_VERSION 120
#ifndef CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif
#ifndef CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#endif
#include <CL/cl.h>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <type_traits>

namespace ldt {
namespace opencl {

namespace {

// ---------------------------------------------------------------------------
// Diagnostics. stderr plus a log file, because GUI hosts on Windows discard
// stderr. The file is opened once. Identical consecutive lines are folded into
// a "(previous line repeated N times)" note so a per-frame failure cannot grow
// the file without bound.
// ---------------------------------------------------------------------------

std::mutex g_diagMutex;
FILE* g_diagFile = nullptr;
bool g_diagFileTried = false;
std::string g_diagLast;
unsigned long g_diagRepeats = 0;

void diagWrite(const char* text)
{
    std::fputs(text, stderr);
    if (g_diagFile != nullptr) {
        std::fputs(text, g_diagFile);
        std::fflush(g_diagFile);
    }
}

void diagLine(const std::string& line)
{
    std::lock_guard<std::mutex> lock(g_diagMutex);
    if (!g_diagFileTried) {
        g_diagFileTried = true;
#if defined(_WIN32)
        const char* tmp = std::getenv("TEMP");
        const std::string path = std::string(tmp ? tmp : ".") + "\\lookdevtools_opencl.log";
#else
        const std::string path = "/tmp/lookdevtools_opencl.log";
#endif
        g_diagFile = std::fopen(path.c_str(), "a");
    }

    if (line == g_diagLast) {
        ++g_diagRepeats;
        if (g_diagRepeats % 1000 == 0) {
            char note[96];
            std::snprintf(note, sizeof(note), "(previous line repeated %lu times so far)\n", g_diagRepeats);
            diagWrite(note);
        }
        return;
    }
    if (g_diagRepeats > 0) {
        char note[96];
        std::snprintf(note, sizeof(note), "(previous line repeated %lu times)\n", g_diagRepeats);
        diagWrite(note);
    }
    g_diagRepeats = 0;
    g_diagLast = line;
    diagWrite(line.c_str());
}

// ---------------------------------------------------------------------------
// Runtime loader. Only the entry points this backend calls. The barrier entry
// points are optional (one is needed only on an out-of-order queue, the 1.2
// one is preferred), and so are the rect read and write (without them the CPU
// fallback moves rows one blocking call at a time).
// ---------------------------------------------------------------------------

struct CLApi {
    bool ok = false;

    cl_int (CL_API_CALL *GetCommandQueueInfo)(cl_command_queue, cl_command_queue_info, size_t, void*, size_t*) = nullptr;
    cl_int (CL_API_CALL *GetDeviceInfo)(cl_device_id, cl_device_info, size_t, void*, size_t*) = nullptr;
    cl_program (CL_API_CALL *CreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*) = nullptr;
    cl_int (CL_API_CALL *BuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void (CL_CALLBACK*)(cl_program, void*), void*) = nullptr;
    cl_int (CL_API_CALL *GetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*) = nullptr;
    cl_kernel (CL_API_CALL *CreateKernel)(cl_program, const char*, cl_int*) = nullptr;
    cl_int (CL_API_CALL *GetKernelWorkGroupInfo)(cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*) = nullptr;
    cl_int (CL_API_CALL *SetKernelArg)(cl_kernel, cl_uint, size_t, const void*) = nullptr;
    cl_int (CL_API_CALL *EnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_mem (CL_API_CALL *CreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*) = nullptr;
    cl_int (CL_API_CALL *ReleaseMemObject)(cl_mem) = nullptr;
    cl_int (CL_API_CALL *EnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_int (CL_API_CALL *EnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_int (CL_API_CALL *EnqueueCopyBufferRect)(cl_command_queue, cl_mem, cl_mem, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_int (CL_API_CALL *Finish)(cl_command_queue) = nullptr;
    cl_int (CL_API_CALL *ReleaseProgram)(cl_program) = nullptr;
    cl_int (CL_API_CALL *ReleaseKernel)(cl_kernel) = nullptr;

    // Optional.
    cl_int (CL_API_CALL *EnqueueBarrierWithWaitList)(cl_command_queue, cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_int (CL_API_CALL *EnqueueBarrier)(cl_command_queue) = nullptr;
    cl_int (CL_API_CALL *EnqueueReadBufferRect)(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*) = nullptr;
    cl_int (CL_API_CALL *EnqueueWriteBufferRect)(cl_command_queue, cl_mem, cl_bool, const size_t*, const size_t*, const size_t*, size_t, size_t, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*) = nullptr;
};

// p_Global is set when symbols resolve through the global namespace (the
// returned handle is then RTLD_DEFAULT, which is a null pointer on glibc).
void* openLoaderLibrary(bool& p_Global)
{
    p_Global = false;
#if defined(_WIN32)
    return reinterpret_cast<void*>(::LoadLibraryA("OpenCL.dll"));
#elif defined(__APPLE__)
    // Only reached by the syntax check build; the Mac plugin has no OpenCL code.
    return ::dlopen("/System/Library/Frameworks/OpenCL.framework/OpenCL", RTLD_LAZY | RTLD_LOCAL);
#else
    void* handle = ::dlopen("libOpenCL.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
        handle = ::dlopen("libOpenCL.so", RTLD_LAZY | RTLD_LOCAL);
    }
#if defined(RTLD_DEFAULT)
    // No loader on the search path, but the host may have one loaded already
    // (it is sending OpenCL renders): resolve through the global namespace.
    if (handle == nullptr && ::dlsym(RTLD_DEFAULT, "clGetCommandQueueInfo") != nullptr) {
        p_Global = true;
        handle = RTLD_DEFAULT;
    }
#endif
    return handle;
#endif
}

void* loaderSymbol(void* handle, const char* name)
{
#if defined(_WIN32)
    return reinterpret_cast<void*>(::GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
#else
    return ::dlsym(handle, name);
#endif
}

const CLApi& api()
{
    static CLApi loaded = [] {
        CLApi a {};
        bool global = false;
        void* handle = openLoaderLibrary(global);
        if (handle == nullptr && !global) {
            Diag("LDT OpenCL: no ICD loader found; OpenCL rendering unavailable.\n");
            return a;
        }
        bool complete = true;
        auto bind = [&](auto& slot, const char* name, bool required) {
            void* sym = loaderSymbol(handle, name);
            if (sym == nullptr) {
                if (required) {
                    Diag("LDT OpenCL: ICD loader lacks %s.\n", name);
                    complete = false;
                }
                return;
            }
            slot = reinterpret_cast<std::remove_reference_t<decltype(slot)>>(sym);
        };
        bind(a.GetCommandQueueInfo,        "clGetCommandQueueInfo", true);
        bind(a.GetDeviceInfo,              "clGetDeviceInfo", true);
        bind(a.CreateProgramWithSource,    "clCreateProgramWithSource", true);
        bind(a.BuildProgram,               "clBuildProgram", true);
        bind(a.GetProgramBuildInfo,        "clGetProgramBuildInfo", true);
        bind(a.CreateKernel,               "clCreateKernel", true);
        bind(a.GetKernelWorkGroupInfo,     "clGetKernelWorkGroupInfo", true);
        bind(a.SetKernelArg,               "clSetKernelArg", true);
        bind(a.EnqueueNDRangeKernel,       "clEnqueueNDRangeKernel", true);
        bind(a.CreateBuffer,               "clCreateBuffer", true);
        bind(a.ReleaseMemObject,           "clReleaseMemObject", true);
        bind(a.EnqueueWriteBuffer,         "clEnqueueWriteBuffer", true);
        bind(a.EnqueueReadBuffer,          "clEnqueueReadBuffer", true);
        bind(a.EnqueueCopyBufferRect,      "clEnqueueCopyBufferRect", true);
        bind(a.Finish,                     "clFinish", true);
        bind(a.ReleaseProgram,             "clReleaseProgram", true);
        bind(a.ReleaseKernel,              "clReleaseKernel", true);
        bind(a.EnqueueBarrierWithWaitList, "clEnqueueBarrierWithWaitList", false);
        bind(a.EnqueueBarrier,             "clEnqueueBarrier", false);
        bind(a.EnqueueReadBufferRect,      "clEnqueueReadBufferRect", false);
        bind(a.EnqueueWriteBufferRect,     "clEnqueueWriteBufferRect", false);
        if (!complete) {
            Diag("LDT OpenCL: ICD loader is missing required entry points.\n");
            return a;
        }
        a.ok = true;
        return a;
    }();
    return loaded;
}

// ---------------------------------------------------------------------------
// Per queue state. One program (and its single kernel) per tool, keyed by the
// tool name the caller passes, plus a small buffer pool shared by the tools.
// Renders on one queue are serialised by QueueContext::mutex, and every render
// drains the queue before it unlocks, so the pooled buffers are never in use
// by two renders at once.
// ---------------------------------------------------------------------------

constexpr int kMaxBuildAttempts = 3;

struct ToolProgram {
    cl_program program = nullptr;
    cl_kernel kernel = nullptr;
    std::size_t maxWg = 0;          // CL_KERNEL_WORK_GROUP_SIZE, cached at creation
    bool ready = false;
    int buildFailures = 0;
    std::uint64_t nextBuildSerial = 0;
};

enum BufferId { B_Params = 0, B_Dims, B_Scratch, B_Count };

struct PooledBuffer {
    cl_mem mem = nullptr;
    std::size_t bytes = 0;
};

struct QueueContext {
    cl_context context = nullptr;
    cl_device_id device = nullptr;
    bool outOfOrder = false;
    std::uint64_t serial = 0;
    std::map<std::string, ToolProgram> programs;
    PooledBuffer pool[B_Count] {};
    std::mutex mutex;
};

std::mutex g_registryMutex;
std::map<cl_command_queue, QueueContext*> g_registry;

QueueContext* contextForQueue(cl_command_queue queue)
{
    std::lock_guard<std::mutex> lock(g_registryMutex);
    auto it = g_registry.find(queue);
    if (it != g_registry.end()) {
        return it->second;
    }
    QueueContext* ctx = new QueueContext();
    g_registry[queue] = ctx;
    return ctx;
}

void releaseProgram(const CLApi& cl, ToolProgram& prog)
{
    if (prog.kernel != nullptr) {
        cl.ReleaseKernel(prog.kernel);
        prog.kernel = nullptr;
    }
    if (prog.program != nullptr) {
        cl.ReleaseProgram(prog.program);
        prog.program = nullptr;
    }
    prog.maxWg = 0;
    prog.ready = false;
}

void releaseAll(const CLApi& cl, QueueContext& ctx)
{
    for (auto& kv : ctx.programs) {
        releaseProgram(cl, kv.second);
    }
    ctx.programs.clear();
    for (int i = 0; i < B_Count; ++i) {
        if (ctx.pool[i].mem != nullptr) {
            cl.ReleaseMemObject(ctx.pool[i].mem);
        }
        ctx.pool[i] = PooledBuffer {};
    }
}

// Re-reads the queue's context, device and properties. A command queue handle
// can be freed by the host and the address reused for a queue on a different
// context; cached programs and buffers from the old one must not be touched.
bool validateQueue(const CLApi& cl, cl_command_queue queue, QueueContext& ctx)
{
    cl_context context = nullptr;
    cl_device_id device = nullptr;
    cl_command_queue_properties props = 0;
    if (cl.GetCommandQueueInfo(queue, CL_QUEUE_CONTEXT, sizeof(context), &context, nullptr) != CL_SUCCESS ||
        cl.GetCommandQueueInfo(queue, CL_QUEUE_DEVICE, sizeof(device), &device, nullptr) != CL_SUCCESS ||
        context == nullptr || device == nullptr) {
        Diag("LDT OpenCL: could not read the context or device off the host command queue.\n");
        return false;
    }
    if (cl.GetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(props), &props, nullptr) != CL_SUCCESS) {
        props = 0;
    }
    if (ctx.context != nullptr && (ctx.context != context || ctx.device != device)) {
        Diag("LDT OpenCL: command queue now belongs to a different context or device; rebuilding.\n");
        releaseAll(cl, ctx);
    }
    ctx.context = context;
    ctx.device = device;
    ctx.outOfOrder = (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;
    return true;
}

// Compiles a tool's embedded source for this queue's device. A failure is
// retried after a growing number of calls, up to kMaxBuildAttempts per context.
bool ensureProgram(const CLApi& cl, QueueContext& ctx, ToolProgram& prog,
                   const char* tool, const char* source, const char* kernelName)
{
    if (prog.ready) {
        return true;
    }
    if (prog.buildFailures >= kMaxBuildAttempts || ctx.serial < prog.nextBuildSerial) {
        return false;
    }

    auto fail = [&]() {
        releaseProgram(cl, prog);
        ++prog.buildFailures;
        prog.nextBuildSerial = ctx.serial + (std::uint64_t(16) << prog.buildFailures);
        if (prog.buildFailures >= kMaxBuildAttempts) {
            Diag("LDT OpenCL: %s: giving up after %d build attempts on this context.\n", tool, prog.buildFailures);
        }
        return false;
    };

    if (source == nullptr || kernelName == nullptr) {
        return fail();
    }
    const std::size_t length = std::strlen(source);
    cl_int err = CL_SUCCESS;
    prog.program = cl.CreateProgramWithSource(ctx.context, 1, &source, &length, &err);
    if (prog.program == nullptr || err != CL_SUCCESS) {
        Diag("LDT OpenCL: %s: clCreateProgramWithSource failed (%d).\n", tool, (int)err);
        prog.program = nullptr;
        return fail();
    }

    // Keep in step with syntax-check-opencl.sh, which rejects "=false" here.
    // Fast relaxed math matches Metal's default fast math; see the header note.
    const char* options = "-cl-std=CL1.2 -cl-fast-relaxed-math";
    err = cl.BuildProgram(prog.program, 1, &ctx.device, options, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::size_t logSize = 0;
        cl.GetProgramBuildInfo(prog.program, ctx.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::string log(logSize + 1, '\0');
        if (logSize > 0) {
            cl.GetProgramBuildInfo(prog.program, ctx.device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
        }
        Diag("LDT OpenCL: %s: clBuildProgram failed (%d), attempt %d. Build log follows:\n%s\n",
             tool, (int)err, prog.buildFailures + 1, log.c_str());
        return fail();
    }

    prog.kernel = cl.CreateKernel(prog.program, kernelName, &err);
    if (prog.kernel == nullptr || err != CL_SUCCESS) {
        Diag("LDT OpenCL: %s: clCreateKernel failed for %s (%d).\n", tool, kernelName, (int)err);
        prog.kernel = nullptr;
        return fail();
    }
    std::size_t wg = 0;
    if (cl.GetKernelWorkGroupInfo(prog.kernel, ctx.device, CL_KERNEL_WORK_GROUP_SIZE,
                                  sizeof(wg), &wg, nullptr) != CL_SUCCESS || wg == 0) {
        wg = 1;
    }
    prog.maxWg = wg;

    char deviceName[256] = "unknown device";
    cl.GetDeviceInfo(ctx.device, CL_DEVICE_NAME, sizeof(deviceName) - 1, deviceName, nullptr);
    deviceName[sizeof(deviceName) - 1] = '\0';
    Diag("LDT OpenCL: %s: %s ready on %s (work-group limit %u).\n",
         tool, kernelName, deviceName, (unsigned)prog.maxWg);

    prog.ready = true;
    prog.buildFailures = 0;
    return true;
}

// Returns a pooled buffer of at least `bytes`, growing it if needed.
cl_mem acquire(const CLApi& cl, QueueContext& ctx, BufferId id, std::size_t bytes, cl_mem_flags flags)
{
    PooledBuffer& slot = ctx.pool[id];
    if (slot.mem != nullptr && slot.bytes < bytes) {
        cl.ReleaseMemObject(slot.mem);
        slot = PooledBuffer {};
    }
    if (slot.mem == nullptr) {
        cl_int err = CL_SUCCESS;
        slot.mem = cl.CreateBuffer(ctx.context, flags, bytes, nullptr, &err);
        if (slot.mem == nullptr || err != CL_SUCCESS) {
            Diag("LDT OpenCL: clCreateBuffer of %u bytes failed (%d).\n", (unsigned)bytes, (int)err);
            slot = PooledBuffer {};
            return nullptr;
        }
        slot.bytes = bytes;
    }
    return slot.mem;
}

// Uploads a small block with a blocking write.
cl_mem upload(const CLApi& cl, cl_command_queue queue, QueueContext& ctx,
              BufferId id, const void* data, std::size_t bytes)
{
    cl_mem mem = acquire(cl, ctx, id, bytes, CL_MEM_READ_ONLY);
    if (mem == nullptr) {
        return nullptr;
    }
    const cl_int err = cl.EnqueueWriteBuffer(queue, mem, CL_TRUE, 0, bytes, data, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        Diag("LDT OpenCL: clEnqueueWriteBuffer of %u bytes failed (%d).\n", (unsigned)bytes, (int)err);
        return nullptr;
    }
    return mem;
}

bool queueOutOfOrder(const CLApi& cl, cl_command_queue queue)
{
    cl_command_queue_properties props = 0;
    if (cl.GetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(props), &props, nullptr) != CL_SUCCESS) {
        return false;
    }
    return (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0;
}

// Orders everything queued after it behind everything queued before it. A
// no-op on an in-order queue.
bool barrier(const CLApi& cl, cl_command_queue queue, bool outOfOrder)
{
    if (!outOfOrder) {
        return true;
    }
    if (cl.EnqueueBarrierWithWaitList != nullptr) {
        return cl.EnqueueBarrierWithWaitList(queue, 0, nullptr, nullptr) == CL_SUCCESS;
    }
    if (cl.EnqueueBarrier != nullptr) {
        return cl.EnqueueBarrier(queue) == CL_SUCCESS;
    }
    // Nothing to order with: drain the queue instead.
    return cl.Finish(queue) == CL_SUCCESS;
}

// Shrinks a 16x16 tile until it fits a kernel's cached work-group limit.
void fitTile(std::size_t maxWg, std::size_t& tileX, std::size_t& tileY)
{
    tileX = 16;
    tileY = 16;
    if (maxWg == 0) {
        maxWg = 1;
    }
    while (tileX * tileY > maxWg) {
        if (tileY > 1) {
            tileY /= 2;
        } else if (tileX > 1) {
            tileX /= 2;
        } else {
            break;
        }
    }
}

inline std::size_t roundUp(std::size_t n, std::size_t multiple)
{
    return ((n + multiple - 1) / multiple) * multiple;
}

// Where the w x h render rect (dst at 0,0) can read from the source: the
// columns [x1, x2) and rows [y1, y2) of the rect that the source covers.
struct Overlap {
    int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    bool empty() const { return x2 <= x1 || y2 <= y1; }
    bool full(int w, int h) const { return x1 == 0 && y1 == 0 && x2 == w && y2 == h; }
};

Overlap overlap(int w, int h, int offX, int offY, int srcW, int srcH)
{
    if (srcW < 0 || srcH < 0) {
        // Unknown size: the source is taken to cover the rect from its offset.
        srcW = offX + w;
        srcH = offY + h;
    }
    Overlap o;
    o.x1 = std::max(0, -offX);
    o.y1 = std::max(0, -offY);
    o.x2 = std::min(w, srcW - offX);
    o.y2 = std::min(h, srcH - offY);
    return o;
}

// Last-resort passthrough: a rectangle copy of the part of the render rect the
// source covers. Pixels outside the source are left as they are.
bool copyRect(const CLApi& cl, cl_command_queue queue, int w, int h, int srcRowBytes, int dstRowBytes,
              cl_mem src, cl_mem dst, int offX, int offY, int srcW, int srcH)
{
    const Overlap o = overlap(w, h, offX, offY, srcW, srcH);
    if (o.empty()) {
        return true;
    }
    if (src == dst && offX == 0 && offY == 0 && srcRowBytes == dstRowBytes) {
        return true;   // in place: the source already is the destination
    }
    const std::size_t px = 4 * sizeof(float);
    const std::size_t srcOrigin[3] = { (std::size_t)(o.x1 + offX) * px, (std::size_t)(o.y1 + offY), 0 };
    const std::size_t dstOrigin[3] = { (std::size_t)o.x1 * px, (std::size_t)o.y1, 0 };
    const std::size_t region[3] = { (std::size_t)(o.x2 - o.x1) * px, (std::size_t)(o.y2 - o.y1), 1 };
    const cl_int err = cl.EnqueueCopyBufferRect(queue, src, dst, srcOrigin, dstOrigin, region,
                                                (std::size_t)srcRowBytes, 0,
                                                (std::size_t)dstRowBytes, 0,
                                                0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        Diag("LDT OpenCL: passthrough clEnqueueCopyBufferRect failed (%d).\n", (int)err);
        return false;
    }
    return true;
}

bool finish(const CLApi& cl, cl_command_queue queue, const char* what)
{
    const cl_int err = cl.Finish(queue);
    if (err != CL_SUCCESS) {
        Diag("LDT OpenCL: clFinish failed (%d) after %s.\n", (int)err, what);
        return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

void Diag(const char* fmt, ...)
{
    char stackBuf[1024];
    va_list args;
    va_start(args, fmt);
    const int needed = std::vsnprintf(stackBuf, sizeof(stackBuf), fmt, args);
    va_end(args);
    if (needed < 0) {
        return;
    }
    std::string line;
    try {
        if (static_cast<std::size_t>(needed) < sizeof(stackBuf)) {
            line.assign(stackBuf, static_cast<std::size_t>(needed));
        } else {
            line.resize(static_cast<std::size_t>(needed) + 1);
            va_start(args, fmt);
            std::vsnprintf(&line[0], line.size(), fmt, args);
            va_end(args);
            line.resize(static_cast<std::size_t>(needed));
        }
        diagLine(line);
    } catch (...) {
        // A diagnostic must never fail a render.
    }
}

bool Available()
{
    return api().ok;
}

bool QueueUsable(void* p_CmdQ)
{
    const CLApi& cl = api();
    if (!cl.ok || p_CmdQ == nullptr) {
        return false;
    }
    cl_context context = nullptr;
    return cl.GetCommandQueueInfo(static_cast<cl_command_queue>(p_CmdQ), CL_QUEUE_CONTEXT,
                                  sizeof(context), &context, nullptr) == CL_SUCCESS &&
           context != nullptr;
}

bool RunPixelKernel(void* p_CmdQ, const char* p_Tool, const char* p_Source, const char* p_KernelName,
                    const void* p_Params, std::size_t p_ParamsBytes,
                    int p_Width, int p_Height, int p_SrcRowBytes, int p_DstRowBytes,
                    const float* p_Input, float* p_Output,
                    int p_SrcOffX, int p_SrcOffY, int p_SrcW, int p_SrcH)
{
    const CLApi& cl = api();
    cl_command_queue queue = static_cast<cl_command_queue>(p_CmdQ);
    if (!cl.ok || queue == nullptr || p_Output == nullptr) {
        return false;
    }
    if (p_Width <= 0 || p_Height <= 0) {
        return true;   // nothing to render
    }
    const char* tool = p_Tool != nullptr ? p_Tool : "?";

    try {
        QueueContext& ctx = *contextForQueue(queue);
        std::lock_guard<std::mutex> lock(ctx.mutex);   // the kernel's args are shared state
        ++ctx.serial;

        if (!validateQueue(cl, queue, ctx)) {
            return false;   // no usable queue, so no passthrough is possible either
        }

        cl_mem srcMem = reinterpret_cast<cl_mem>(const_cast<float*>(p_Input));
        cl_mem dstMem = reinterpret_cast<cl_mem>(p_Output);
        const Overlap cover = overlap(p_Width, p_Height, p_SrcOffX, p_SrcOffY, p_SrcW, p_SrcH);
        const bool pitchOk = p_SrcRowBytes > 0 && p_DstRowBytes > 0 &&
                             p_SrcRowBytes % 16 == 0 && p_DstRowBytes % 16 == 0;

        bool rendered = false;
        if (srcMem == nullptr) {
            Diag("LDT OpenCL: %s: no source image.\n", tool);
        } else if (!pitchOk) {
            Diag("LDT OpenCL: %s: row pitch %d / %d is not whole float4 pixels.\n",
                 tool, p_SrcRowBytes, p_DstRowBytes);
        } else if (!cover.full(p_Width, p_Height)) {
            Diag("LDT OpenCL: %s: source does not cover the render rect.\n", tool);
        } else if (barrier(cl, queue, ctx.outOfOrder)) {
            ToolProgram& prog = ctx.programs[tool];
            if (ensureProgram(cl, ctx, prog, tool, p_Source, p_KernelName)) {
                bool ok = true;
                cl_mem kernelSrc = srcMem;
                int srcPitch = p_SrcRowBytes / 16;
                if (p_SrcOffX != 0 || p_SrcOffY != 0) {
                    // A cl_mem cannot be offset: stage the rect into a tight scratch image.
                    const std::size_t tightRow = (std::size_t)p_Width * 16;
                    cl_mem scratch = acquire(cl, ctx, B_Scratch, tightRow * (std::size_t)p_Height, CL_MEM_READ_WRITE);
                    ok = scratch != nullptr &&
                         copyRect(cl, queue, p_Width, p_Height, p_SrcRowBytes, (int)tightRow,
                                  srcMem, scratch, p_SrcOffX, p_SrcOffY, p_SrcW, p_SrcH) &&
                         barrier(cl, queue, ctx.outOfOrder);
                    kernelSrc = scratch;
                    srcPitch = p_Width;
                }
                const cl_int dims[4] = { srcPitch, p_DstRowBytes / 16, p_Width, p_Height };
                cl_mem paramsMem = ok ? upload(cl, queue, ctx, B_Params, p_Params, p_ParamsBytes) : nullptr;
                cl_mem dimsMem = paramsMem != nullptr ? upload(cl, queue, ctx, B_Dims, dims, sizeof(dims)) : nullptr;
                ok = ok && paramsMem != nullptr && dimsMem != nullptr;
                const cl_mem args[4] = { paramsMem, kernelSrc, dstMem, dimsMem };
                for (cl_uint i = 0; ok && i < 4; ++i) {
                    if (cl.SetKernelArg(prog.kernel, i, sizeof(cl_mem), &args[i]) != CL_SUCCESS) {
                        Diag("LDT OpenCL: %s: clSetKernelArg %u failed for %s.\n", tool, (unsigned)i, p_KernelName);
                        ok = false;
                    }
                }
                if (ok) {
                    std::size_t tileX = 16;
                    std::size_t tileY = 16;
                    fitTile(prog.maxWg, tileX, tileY);
                    const std::size_t global[2] = { roundUp((std::size_t)p_Width, tileX),
                                                    roundUp((std::size_t)p_Height, tileY) };
                    const std::size_t local[2] = { tileX, tileY };
                    const cl_int err = cl.EnqueueNDRangeKernel(queue, prog.kernel, 2, nullptr, global, local,
                                                               0, nullptr, nullptr);
                    if (err != CL_SUCCESS) {
                        Diag("LDT OpenCL: %s: clEnqueueNDRangeKernel failed for %s (%d), global %ux%u local %ux%u.\n",
                             tool, p_KernelName, (int)err, (unsigned)global[0], (unsigned)global[1],
                             (unsigned)local[0], (unsigned)local[1]);
                        ok = false;
                    }
                }
                rendered = ok;
            }
        }

        if (!rendered && srcMem != nullptr) {
            Diag("LDT OpenCL: %s: kernel did not run; passing the source through.\n", tool);
            barrier(cl, queue, ctx.outOfOrder);
            copyRect(cl, queue, p_Width, p_Height, p_SrcRowBytes, p_DstRowBytes,
                     srcMem, dstMem, p_SrcOffX, p_SrcOffY, p_SrcW, p_SrcH);
        }

        if (!finish(cl, queue, tool)) {
            return false;
        }
        return rendered;
    } catch (...) {
        // std::map / std::string allocation only. The queue is drained so no
        // queued work outlives the call.
        Diag("LDT OpenCL: %s: out of memory in the dispatcher.\n", tool);
        cl.Finish(queue);
        return false;
    }
}

bool CopyImage(void* p_CmdQ, int p_Width, int p_Height, int p_SrcRowBytes, int p_DstRowBytes,
               const float* p_Input, float* p_Output,
               int p_SrcOffX, int p_SrcOffY, int p_SrcW, int p_SrcH)
{
    const CLApi& cl = api();
    cl_command_queue queue = static_cast<cl_command_queue>(p_CmdQ);
    if (!cl.ok || queue == nullptr || p_Input == nullptr || p_Output == nullptr) {
        return false;
    }
    if (p_Width <= 0 || p_Height <= 0) {
        return true;
    }
    const bool ordered = barrier(cl, queue, queueOutOfOrder(cl, queue));
    const bool copied = ordered &&
        copyRect(cl, queue, p_Width, p_Height, p_SrcRowBytes, p_DstRowBytes,
                 reinterpret_cast<cl_mem>(const_cast<float*>(p_Input)), reinterpret_cast<cl_mem>(p_Output),
                 p_SrcOffX, p_SrcOffY, p_SrcW, p_SrcH);
    const bool drained = finish(cl, queue, "copy");
    return copied && drained;
}

bool ReadImage(void* p_CmdQ, const float* p_DeviceBuf, int p_Width, int p_Height, int p_RowBytes,
               float* p_HostTight, int p_OffX, int p_OffY, int p_SrcW, int p_SrcH)
{
    const CLApi& cl = api();
    cl_command_queue queue = static_cast<cl_command_queue>(p_CmdQ);
    if (!cl.ok || queue == nullptr || p_DeviceBuf == nullptr || p_HostTight == nullptr || p_RowBytes <= 0) {
        return false;
    }
    if (p_Width <= 0 || p_Height <= 0) {
        return true;
    }
    const std::size_t px = 4 * sizeof(float);
    const std::size_t tightRow = (std::size_t)p_Width * px;
    std::memset(p_HostTight, 0, tightRow * (std::size_t)p_Height);   // outside the source: transparent black

    const Overlap o = overlap(p_Width, p_Height, p_OffX, p_OffY, p_SrcW, p_SrcH);
    if (o.empty()) {
        return true;
    }
    cl_mem mem = reinterpret_cast<cl_mem>(const_cast<float*>(p_DeviceBuf));
    bool ok = barrier(cl, queue, queueOutOfOrder(cl, queue));
    if (ok && cl.EnqueueReadBufferRect != nullptr) {
        const std::size_t bufOrigin[3] = { (std::size_t)(o.x1 + p_OffX) * px, (std::size_t)(o.y1 + p_OffY), 0 };
        const std::size_t hostOrigin[3] = { (std::size_t)o.x1 * px, (std::size_t)o.y1, 0 };
        const std::size_t region[3] = { (std::size_t)(o.x2 - o.x1) * px, (std::size_t)(o.y2 - o.y1), 1 };
        const cl_int err = cl.EnqueueReadBufferRect(queue, mem, CL_TRUE, bufOrigin, hostOrigin, region,
                                                    (std::size_t)p_RowBytes, 0, tightRow, 0,
                                                    p_HostTight, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            Diag("LDT OpenCL: clEnqueueReadBufferRect failed (%d).\n", (int)err);
            ok = false;
        }
    } else if (ok) {
        for (int y = o.y1; ok && y < o.y2; ++y) {
            const std::size_t offset = (std::size_t)(y + p_OffY) * (std::size_t)p_RowBytes +
                                       (std::size_t)(o.x1 + p_OffX) * px;
            const cl_int err = cl.EnqueueReadBuffer(queue, mem, CL_TRUE, offset, (std::size_t)(o.x2 - o.x1) * px,
                                                    reinterpret_cast<char*>(p_HostTight) + (std::size_t)y * tightRow +
                                                        (std::size_t)o.x1 * px,
                                                    0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                Diag("LDT OpenCL: clEnqueueReadBuffer failed (%d).\n", (int)err);
                ok = false;
            }
        }
    }
    const bool drained = finish(cl, queue, "readback");
    return ok && drained;
}

bool WriteImage(void* p_CmdQ, float* p_DeviceBuf, int p_Width, int p_Height, int p_RowBytes,
                const float* p_HostTight)
{
    const CLApi& cl = api();
    cl_command_queue queue = static_cast<cl_command_queue>(p_CmdQ);
    if (!cl.ok || queue == nullptr || p_DeviceBuf == nullptr || p_HostTight == nullptr || p_RowBytes <= 0) {
        return false;
    }
    if (p_Width <= 0 || p_Height <= 0) {
        return true;
    }
    const std::size_t px = 4 * sizeof(float);
    const std::size_t tightRow = (std::size_t)p_Width * px;
    if ((std::size_t)p_RowBytes < tightRow) {
        Diag("LDT OpenCL: upload row pitch %d is narrower than %d pixels.\n", p_RowBytes, p_Width);
        return false;
    }
    cl_mem mem = reinterpret_cast<cl_mem>(p_DeviceBuf);
    bool ok = barrier(cl, queue, queueOutOfOrder(cl, queue));
    if (ok && cl.EnqueueWriteBufferRect != nullptr) {
        const std::size_t origin[3] = { 0, 0, 0 };
        const std::size_t region[3] = { tightRow, (std::size_t)p_Height, 1 };
        const cl_int err = cl.EnqueueWriteBufferRect(queue, mem, CL_TRUE, origin, origin, region,
                                                     (std::size_t)p_RowBytes, 0, tightRow, 0,
                                                     p_HostTight, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            Diag("LDT OpenCL: clEnqueueWriteBufferRect failed (%d).\n", (int)err);
            ok = false;
        }
    } else if (ok) {
        for (int y = 0; ok && y < p_Height; ++y) {
            const cl_int err = cl.EnqueueWriteBuffer(queue, mem, CL_TRUE, (std::size_t)y * (std::size_t)p_RowBytes,
                                                     tightRow,
                                                     reinterpret_cast<const char*>(p_HostTight) + (std::size_t)y * tightRow,
                                                     0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                Diag("LDT OpenCL: clEnqueueWriteBuffer failed (%d).\n", (int)err);
                ok = false;
            }
        }
    }
    const bool drained = finish(cl, queue, "upload");
    return ok && drained;
}

}  // namespace opencl
}  // namespace ldt

#endif  // LDT_OPENCL_ENABLED
