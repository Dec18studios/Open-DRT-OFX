#pragma once
// LookDevPlatform.h
// Platform qualifier + cross-platform float3 + math-builtin aliases shared by
// every Look Dev Tools algorithm header.
//
// Lifted verbatim from SplitToneXAlgorithm.h lines 11-113 (STX_ -> LDT_). Keep
// this header-only and dependency-free: it is #included as text inside the Metal
// shader source AND compiled by CUDA (__CUDACC__) and the CPU path.
//
// Maintenance rule: the qualifier LDT_FUNC resolves to `__device__ inline` under
// CUDA and plain `inline` under Metal/CPU, so a single algorithm body compiles on
// all three backends.

// ── Platform qualifiers ───────────────────────────────────────────────────────
#if defined(__CUDACC__)
  #define LDT_FUNC  __device__ inline
  #define LDT_HOST  __host__
#elif defined(__METAL_VERSION__)
  #define LDT_FUNC  inline
  #define LDT_HOST
#else
  #define LDT_FUNC  inline
  #define LDT_HOST
#endif

// ── Cross-platform float3 ─────────────────────────────────────────────────────
#if defined(__METAL_VERSION__)
  // Metal: use native float3
#elif defined(__CUDACC__)
  // CUDA: native float3 / make_float3 are available from cuda_runtime.h.
  // Map the _xxx() aliases used in the algorithm to CUDA's device math builtins.
  #ifndef _fabs
    #define _fabs(x)        fabsf((float)(x))
  #endif
  #ifndef _powf
    #define _powf(x,y)      powf((float)(x),(float)(y))
  #endif
  #ifndef _expf
    #define _expf(x)        expf((float)(x))
  #endif
  #ifndef _exp2f
    #define _exp2f(x)       exp2f((float)(x))
  #endif
  #ifndef _log2f
    #define _log2f(x)       log2f((float)(x))
  #endif
  #ifndef _fmaxf
    #define _fmaxf(x,y)     fmaxf((float)(x),(float)(y))
  #endif
  #ifndef _fminf
    #define _fminf(x,y)     fminf((float)(x),(float)(y))
  #endif
  #ifndef _floorf
    #define _floorf(x)      floorf((float)(x))
  #endif
  #ifndef _fmodf
    #define _fmodf(x,y)     fmodf((float)(x),(float)(y))
  #endif
  #ifndef _fmod
    #define _fmod(x,y)      fmodf((float)(x),(float)(y))
  #endif
  #ifndef _sinf
    #define _sinf(x)        sinf((float)(x))
  #endif
  #ifndef _cosf
    #define _cosf(x)        cosf((float)(x))
  #endif
  #ifndef _tanf
    #define _tanf(x)        tanf((float)(x))
  #endif
  #ifndef _acosf
    #define _acosf(x)       acosf((float)(x))
  #endif
  #ifndef _atan2f
    #define _atan2f(y,x)    atan2f((float)(y),(float)(x))
  #endif
  #ifndef _sqrtf
    #define _sqrtf(x)       sqrtf((float)(x))
  #endif
  #ifndef _copysignf
    #define _copysignf(x,y) copysignf((float)(x),(float)(y))
  #endif
  #ifndef _log10f
    #define _log10f(x)      log10f((float)(x))
  #endif
  #ifndef _saturatef
    #define _saturatef(x)   __saturatef((float)(x))
  #endif
#else
  // Define a guard so a co-included tool header (e.g. SplitToneAlgorithm.h) that
  // also defines a byte-identical float3/make_float3 can skip its own definition.
  #if !defined(LDT_PLATFORM_FLOAT3_DEFINED) && !defined(STX_FLOAT3_DEFINED)
  #define LDT_PLATFORM_FLOAT3_DEFINED
  struct float3 { float x, y, z; };
  LDT_HOST static inline float3 make_float3(float x, float y, float z)
      { float3 r; r.x = x; r.y = y; r.z = z; return r; }
  #endif
  struct float2 { float x, y; };
  LDT_HOST static inline float2 make_float2(float x, float y)
      { float2 r; r.x = x; r.y = y; return r; }
  #include <cmath>
  #include <algorithm>
  #ifndef _fabs
    #define _fabs(x)  std::abs(x)
  #endif
  #ifndef _powf
    #define _powf(x,y) std::pow((float)(x),(float)(y))
  #endif
  #ifndef _expf
    #define _expf(x)  std::exp((float)(x))
  #endif
  #ifndef _exp2f
    #define _exp2f(x) std::exp2((float)(x))
  #endif
  #ifndef _log2f
    #define _log2f(x) std::log2((float)(x))
  #endif
  #ifndef _fmaxf
    // std::fmax (NOT std::max) so the host CPU path matches CUDA/Metal fmaxf:
    // when one argument is NaN the non-NaN argument is returned. std::max would
    // propagate NaN, diverging from the device kernels at every clamp/min/max.
    #define _fmaxf(x,y) std::fmax((float)(x),(float)(y))
  #endif
  #ifndef _fminf
    #define _fminf(x,y) std::fmin((float)(x),(float)(y))
  #endif
  #ifndef _floorf
    #define _floorf(x) std::floor((float)(x))
  #endif
  #ifndef _fmodf
    #define _fmodf(x,y) std::fmod((float)(x),(float)(y))
  #endif
  #ifndef _fmod
    #define _fmod(x,y) std::fmod((float)(x),(float)(y))
  #endif
  #ifndef _sinf
    #define _sinf(x) std::sin((float)(x))
  #endif
  #ifndef _cosf
    #define _cosf(x) std::cos((float)(x))
  #endif
  #ifndef _tanf
    #define _tanf(x) std::tan((float)(x))
  #endif
  #ifndef _acosf
    #define _acosf(x) std::acos((float)(x))
  #endif
  #ifndef _atan2f
    #define _atan2f(y,x) std::atan2((float)(y),(float)(x))
  #endif
  #ifndef _sqrtf
    #define _sqrtf(x) std::sqrt((float)(x))
  #endif
  #ifndef _copysignf
    #define _copysignf(x,y) std::copysign((float)(x),(float)(y))
  #endif
  #ifndef _log10f
    #define _log10f(x) std::log10((float)(x))
  #endif
  #ifndef _saturatef
    // CUDA __saturatef(NaN) -> 0; std::fmin/fmax give the same NaN->0 here.
    #define _saturatef(x) std::fmin(std::fmax((float)(x), 0.0f), 1.0f)
  #endif
#endif

// ── float3 arithmetic operators ───────────────────────────────────────────────
// Metal's float3 has these natively; CUDA's float3 and the host struct do NOT.
// Some algorithm bodies (e.g. the tetrahedral hue engine) do float3 +/-/scalar*
// arithmetic the way DCTL allows. Provide the operators on the non-Metal backends
// so the SAME source compiles on host, CUDA, and Metal.
#if !defined(__METAL_VERSION__)
LDT_FUNC float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
LDT_FUNC float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
LDT_FUNC float3 operator*(float s, float3 a)  { return make_float3(s * a.x, s * a.y, s * a.z); }
LDT_FUNC float3 operator*(float3 a, float s)  { return make_float3(a.x * s, a.y * s, a.z * s); }
#endif
