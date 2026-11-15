// ChromagnonMetal.mm - Metal backend for the Chromagnon view transform (6A).
//
// The MSL kernel below is a hand-maintained twin of the Chromagnon device path
// (core/ChromagnonAlgorithm.h plus the tess_ helpers it reuses from
// core/TesseractAlgorithm.h and the odt_ helpers from core/OpenDRTAlgorithm.h)
// over the suite's float3/math aliases. Forward and inverse both live in the one
// kernel; p->direction selects. It is compiled once per MTLCommandQueue at
// runtime (the host ships no .metallib) and the pipeline state is cached.
// ChromagnonParams is duplicated in MSL with a byte-identical layout (464 bytes,
// ints and floats only); keep the two in lockstep with core/ChromagnonParams.h.
//
// The #define aliases at the top of kChromagnonMetalSrc make the function bodies
// textually identical to the C++/CUDA source so the twin can be diffed by eye.
// Differences from the C++ text: matrix pointers and the params pointer carry
// the MSL `constant` address space, TESS_EXP10F is exp10 (the core header picks
// the same form under __METAL_VERSION__), and the comments are shortened.
//
// The caller has ALREADY run chromagnon_precompute() on the params block; the
// kernel only reads the resolved fields.
//
// GOTCHA: after ANY signature/struct change, extract this string and
// `xcrun -sdk macosx metal -c` it; a host build green proves nothing, a runtime
// MSL compile failure only shows as a passthrough or white frame. No C++
// lambdas in MSL. Buffers arrive as float* handles -> reinterpret_cast, never
// newBufferWithBytesNoCopy, never released here.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstdio>
#include <unordered_map>
#include <mutex>

#include "../core/ChromagnonAlgorithm.h"

static_assert(sizeof(ChromagnonParams) == 464, "ChromagnonParams changed; update kChromagnonMetalSrc");

// Pipeline cache (keyed by MTLCommandQueue pointer). File-scope statics so the
// symbols never collide with the other 6A Metal files in the monolith.
static std::mutex s_CmgPipelineMutex;
static std::unordered_map<void*, id<MTLComputePipelineState>> s_CmgPipelineCache;

// Metal shader source (MSL twin of ChromagnonAlgorithm.h device path).
static const char* kChromagnonMetalSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

#define LDT_FUNC inline

// Math-builtin aliases so the bodies below are textually identical to the
// C++/CUDA source in core/TesseractAlgorithm.h and core/OpenDRTAlgorithm.h.
#define _powf(x,y)       pow((float)(x),(float)(y))
#define _exp2f(x)        exp2((float)(x))
#define _expf(x)         exp((float)(x))
#define _logf(x)         log((float)(x))
#define _log2f(x)        log2((float)(x))
#define _fmaxf(x,y)      fmax((float)(x),(float)(y))
#define _fminf(x,y)      fmin((float)(x),(float)(y))
#define _sqrtf(x)        sqrt((float)(x))
#define make_float3(a,b,c) float3(a,b,c)
#define TESS_EXP10F(x)   exp10((float)(x))

// Byte-identical twin of ChromagnonParams (core/ChromagnonParams.h), 464 bytes.
typedef struct ChromagnonParams {
    int   in_gamut;
    int   in_oetf;
    int   display_gamut;
    int   eotf;
    int   clamp;
    int   direction;
    float Lp;
    float Lg;
    float Lgb;
    float off;
    float con;
    float conh;
    float w0;
    float wtm;
    float wtn;
    float gtm;
    float gtn;
    float pc_p;
    float pc_x0;
    float pc_t0;
    float rs_rnd;
    float gs_rnd;
    float bs_rnd;
    float rr_rnd;
    float gr_rnd;
    float br_rnd;
    float rs_lk;
    float gs_lk;
    float bs_lk;
    float in_to_xyz[9];
    float xyz_to_display[9];
    float display_to_xyz[9];
    float xyz_to_in[9];
    float rnd_mtx[9];
    float rnd_mtx_inv[9];
    float look_mtx[9];
    float look_mtx_inv[9];
    float ds;
    float clamp_max;
    float eotf_p;
    float pc_m;
    float c_s0;
    float c_t0;
    float c_o0;
    float c_m0;
    float ts_m_r;
    float ts_m_g;
    float ts_m_b;
    float ts_s_r;
    float ts_s_g;
    float ts_s_b;
    int   _pad0;
} ChromagnonParams;

static_assert(sizeof(ChromagnonParams) == 464, "ChromagnonParams layout");

// ---- odt_ helpers (verbatim from core/OpenDRTAlgorithm.h) ----------------------
LDT_FUNC float odt_sdivf(float a, float b) {
    return (b == 0.0f) ? 0.0f : a/b;
}
LDT_FUNC float3 odt_sdivf3f(float3 a, float b) {
    return make_float3(odt_sdivf(a.x, b), odt_sdivf(a.y, b), odt_sdivf(a.z, b));
}
LDT_FUNC float odt_spowf(float a, float b) {
    return (a <= 0.0f) ? a : _powf(a, b);
}
LDT_FUNC float3 odt_spowf3(float3 a, float b) {
    return make_float3(odt_spowf(a.x, b), odt_spowf(a.y, b), odt_spowf(a.z, b));
}
LDT_FUNC float odt_clampf(float a, float mn, float mx) {
    return _fminf(_fmaxf(a, mn), mx);
}
LDT_FUNC float3 odt_clampf3(float3 a, float mn, float mx) {
    return make_float3(odt_clampf(a.x, mn, mx), odt_clampf(a.y, mn, mx), odt_clampf(a.z, mn, mx));
}
LDT_FUNC float3 odt_clampminf3(float3 a, float mn) {
    return make_float3(_fmaxf(a.x, mn), _fmaxf(a.y, mn), _fmaxf(a.z, mn));
}
LDT_FUNC float odt_fmaxf3(float3 a) {
    return _fmaxf(a.x, _fmaxf(a.y, a.z));
}
LDT_FUNC float odt_fminf3(float3 a) {
    return _fminf(a.x, _fminf(a.y, a.z));
}
LDT_FUNC float odt_oetf_davinci_intermediate(float x) {
    return x <= 0.02740668f ? x/10.44426855f : _exp2f(x/0.07329248f - 7.0f) - 0.0075f;
}
LDT_FUNC float odt_oetf_filmlight_tlog(float x) {
    return x < 0.075f ? (x-0.075f)/16.184376489665897f : _expf((x - 0.5520126568606655f)/0.09232902596577353f) - 0.0057048244042473785f;
}
LDT_FUNC float odt_oetf_acescct(float x) {
    return x <= 0.155251141552511f ? (x - 0.0729055341958355f)/10.5402377416545f : _exp2f(x*17.52f - 9.72f);
}
LDT_FUNC float odt_oetf_arri_logc4(float x) {
    return x < -0.7774983977293537f ? x*0.3033266726886969f - 0.7774983977293537f : (_exp2f(14.0f*(x - 0.09286412512218964f)/0.9071358748778103f + 6.0f) - 64.0f)/2231.8263090676883f;
}
LDT_FUNC float odt_oetf_red_log3g10(float x) {
    return x < 0.0f ? (x/15.1927f) - 0.01f : (_powf(10.0f, x/0.224282f) - 1.0f)/155.975327f - 0.01f;
}
LDT_FUNC float odt_oetf_fujifilm_flog2(float x) {
    return x < 0.100686685370811f ? (x - 0.092864f)/8.799461f : (_powf(10.0f, ((x - 0.384316f)/0.245281f))/5.555556f - 0.064829f/5.555556f);
}
LDT_FUNC float3 odt_eotf_pq(float3 rgb, int inverse) {
    const float m1 = 2610.0f/16384.0f;
    const float m2 = 2523.0f/32.0f;
    const float c1 = 107.0f/128.0f;
    const float c2 = 2413.0f/128.0f;
    const float c3 = 2392.0f/128.0f;

    if (inverse == 1) {
        rgb = odt_spowf3(rgb, m1);
        rgb = odt_spowf3(make_float3((c1 + c2*rgb.x)/(1.0f + c3*rgb.x),
                                     (c1 + c2*rgb.y)/(1.0f + c3*rgb.y),
                                     (c1 + c2*rgb.z)/(1.0f + c3*rgb.z)), m2);
    } else {
        rgb = odt_spowf3(rgb, 1.0f/m2);
        rgb = odt_spowf3(make_float3((rgb.x - c1)/(c2 - c3*rgb.x),
                                     (rgb.y - c1)/(c2 - c3*rgb.y),
                                     (rgb.z - c1)/(c2 - c3*rgb.z)), 1.0f/m1);
    }
    return rgb;
}

// ---- tess_ helpers reused by Chromagnon (core/TesseractAlgorithm.h) -----------------------------------
LDT_FUNC float3 tess_vdot9(constant float* m, float3 v) {
    return make_float3(m[0]*v.x + m[1]*v.y + m[2]*v.z,
                       m[3]*v.x + m[4]*v.y + m[5]*v.z,
                       m[6]*v.x + m[7]*v.y + m[8]*v.z);
}
LDT_FUNC float tess_oetf_arri_logc3(float x) {
    return x < 5.367655f*0.010591f + 0.092809f ? (x - 0.092809f)/5.367655f : (TESS_EXP10F((x - 0.385537f)/0.247190f) - 0.052272f)/5.555556f;
}
LDT_FUNC float tess_oetf_panasonic_vlog(float x) {
    return x < 0.181f ? (x - 0.125f)/5.6f : TESS_EXP10F((x - 0.598206f)/0.241514f) - 0.00873f;
}
LDT_FUNC float tess_oetf_sony_slog3(float x) {
    return x < 171.2102946929f/1023.0f ? (x*1023.0f - 95.0f)*0.01125f/(171.2102946929f - 95.0f) : (TESS_EXP10F(((x*1023.0f - 420.0f)/261.5f))*(0.18f + 0.01f) - 0.01f);
}
LDT_FUNC float tess_oetf_fujifilm_flog(float x) {
    return x < 0.1005377752f ? (x - 0.092864f)/8.735631f : (TESS_EXP10F(((x - 0.790453f)/0.344676f))/0.555556f - 0.009468f/0.555556f);
}
LDT_FUNC float tess_apply_oetf(float x, int tf) {
    switch (tf) {
        case 1:  return odt_oetf_davinci_intermediate(x);
        case 2:  return odt_oetf_filmlight_tlog(x);
        case 3:  return odt_oetf_acescct(x);
        case 4:  return tess_oetf_arri_logc3(x);
        case 5:  return odt_oetf_arri_logc4(x);
        case 6:  return odt_oetf_red_log3g10(x);
        case 7:  return tess_oetf_panasonic_vlog(x);
        case 8:  return tess_oetf_sony_slog3(x);
        case 9:  return odt_oetf_fujifilm_flog2(x);
        case 10: return tess_oetf_fujifilm_flog(x);
        default: return x;
    }
}
LDT_FUNC float3 tess_linearize(float3 rgb, int tf) {
    if (tf == 0) return rgb;
    rgb.x = tess_apply_oetf(rgb.x, tf);
    rgb.y = tess_apply_oetf(rgb.y, tf);
    rgb.z = tess_apply_oetf(rgb.z, tf);
    return rgb;
}
LDT_FUNC float3 tess_eotf_hlg(float3 rgb, int inverse) {
    const float HLG_Lw = 1000.0f;
    const float HLG_Ls = 5.0f;
    const float h_a = 0.17883277f;
    const float h_b = 1.0f - 4.0f*0.17883277f;
    const float h_c = 0.5f - h_a*_logf(4.0f*h_a);
    const float h_g = 1.2f*odt_spowf(1.111f, _log2f(HLG_Lw/1000.0f))*odt_spowf(0.98f, _log2f(_fmaxf(1e-6f, HLG_Ls)/5.0f));
    if (inverse == 1) {
        float Yd = 0.2627f*rgb.x + 0.6780f*rgb.y + 0.0593f*rgb.z;
        rgb = rgb*odt_spowf(Yd, (1.0f - h_g)/h_g);
        // DEVIATION (core/TesseractAlgorithm.h): fmax(0, x) keeps sqrt finite for x < 0.
        rgb.x = rgb.x <= 1.0f/12.0f ? _sqrtf(3.0f*_fmaxf(0.0f, rgb.x)) : h_a*_logf(12.0f*rgb.x - h_b) + h_c;
        rgb.y = rgb.y <= 1.0f/12.0f ? _sqrtf(3.0f*_fmaxf(0.0f, rgb.y)) : h_a*_logf(12.0f*rgb.y - h_b) + h_c;
        rgb.z = rgb.z <= 1.0f/12.0f ? _sqrtf(3.0f*_fmaxf(0.0f, rgb.z)) : h_a*_logf(12.0f*rgb.z - h_b) + h_c;
    } else {
        rgb.x = rgb.x <= 0.5f ? rgb.x*rgb.x/3.0f : (_expf((rgb.x - h_c)/h_a) + h_b)/12.0f;
        rgb.y = rgb.y <= 0.5f ? rgb.y*rgb.y/3.0f : (_expf((rgb.y - h_c)/h_a) + h_b)/12.0f;
        rgb.z = rgb.z <= 0.5f ? rgb.z*rgb.z/3.0f : (_expf((rgb.z - h_c)/h_a) + h_b)/12.0f;
        float Ys = 0.2627f*rgb.x + 0.6780f*rgb.y + 0.0593f*rgb.z;
        rgb = rgb*odt_spowf(Ys, h_g - 1.0f);
    }
    return rgb;
}
LDT_FUNC float3 tess_display_encode(float3 rgb, int eotf, float eotf_p) {
    if ((eotf > 0) && (eotf < 4)) {
        rgb = odt_spowf3(rgb, 1.0f/eotf_p);
    } else if (eotf == 4) {
        rgb = odt_eotf_pq(rgb, 1);
    } else if (eotf == 5) {
        rgb = tess_eotf_hlg(rgb, 1);
    }
    return rgb;
}
// ---- cmg_ helpers (core/ChromagnonAlgorithm.h) -----------------------------------
LDT_FUNC float3 cmg_xlog(float3 rgb, int inv) {
    const float w0 = 128.0f;
    const float k = -4.0f;
    const float y1 = 0.0f;
    const float o0 = -odt_spowf(2.0f, k);
    const float s0 = (1.0f - y1)/_logf(1.0f - w0/o0);
    const float o1 = 1.0f - s0*_logf(w0 - o0);
    if (inv == 1) {
        rgb.x = _expf((rgb.x - o1)/s0) + o0;
        rgb.y = _expf((rgb.y - o1)/s0) + o0;
        rgb.z = _expf((rgb.z - o1)/s0) + o0;
    } else {
        rgb.x = s0*_logf(rgb.x - o0) + o1;
        rgb.y = s0*_logf(rgb.y - o0) + o1;
        rgb.z = s0*_logf(rgb.z - o0) + o1;
    }
    return rgb;
}
LDT_FUNC float cmg_log1p(float u) {
    if (_fmaxf(u, -u) < 0.05f)
        return u*(1.0f - u*(0.5f - u*(1.0f/3.0f - u*(0.25f - u*(0.2f - u*(1.0f/6.0f - u*(1.0f/7.0f)))))));
    return _logf(1.0f + u);
}
LDT_FUNC float cmg_expm1(float z) {
    if (_fmaxf(z, -z) < 0.05f)
        return z*(1.0f + z*(0.5f + z*(1.0f/6.0f + z*(1.0f/24.0f + z*(1.0f/120.0f + z*(1.0f/720.0f))))));
    return _expf(z) - 1.0f;
}
// DEVIATION (core/ChromagnonAlgorithm.h): cancellation-free forms when
// |m| < 0.01*t0; forward overflow-safe identity when log(q) > 64*p; inverse
// base >= 1e-30 and |gain| <= 1e10 (x <= m keeps the upstream pass-through).
LDT_FUNC float cmg_pc_inv_gain(float b, float p) {
    if (b <= 0.0f) return _fmaxf(b, -1e10f);
    return _fminf(odt_spowf(_fmaxf(b, 1e-30f), -p), 1e10f);
}
LDT_FUNC float cmg_compress_powerptoe(float x, float p, float m, float t0, int inv) {
    if (x > t0) return x;
    float q = (t0 - x)/(t0 - m);
    const bool small_m = _fmaxf(m, -m) < 0.01f*t0;
    if (inv == 1 && small_m) {
        const float d = (m - x)/(t0 - m);
        return (x - t0)*cmg_pc_inv_gain(-cmg_expm1(cmg_log1p(d)/p), p) + t0;
    }
    if (inv == 0 && small_m && q > 1.0f)
        return m - (t0 - m)*cmg_expm1(-p*cmg_log1p(odt_spowf(q, -1.0f/p)));
    if (inv == 1) {
        return (x - t0)*cmg_pc_inv_gain(1.0f - odt_spowf(q, 1.0f/p), p) + t0;
    }
    if (q > 1.0f && _logf(q) > 64.0f*p)
        return (x - t0)*odt_spowf(1.0f + odt_spowf(q, -1.0f/p), -p)/q + t0;
    return (x - t0)*odt_spowf(1.0f + odt_spowf(q, 1.0f/p), -p) + t0;
}
LDT_FUNC float cmg_contrast(float x, float p, float s0, float t0, float o0, float m0, int inv) {
    if (inv == 1)
        return x < o0 ? odt_spowf(x/s0, 1.0f/p) : (x - o0)/m0 + t0;
    else
        return x < t0 ? s0*odt_spowf(x, p) : m0*(x - t0) + o0;
}
LDT_FUNC float cmg_tonescale(float x, float m, float s, int inv) {
    // DEVIATION (core): denominator floored at 1e-6*m (inf at x == m upstream).
    if (inv == 1)
        return x > m ? x : s*x/_fmaxf(m - x, 1e-6f*m);
    else
        return m*x/(x + s);
}
// Inverse-path display decode (includes the core's PQ clamp-at-0 deviation).
LDT_FUNC float3 cmg_display_decode(float3 rgb, int eotf, float eotf_p) {
    if ((eotf > 0) && (eotf < 4)) {
        rgb = odt_spowf3(rgb, eotf_p);
    } else if (eotf == 4) {
        rgb = odt_eotf_pq(rgb, 0);
        rgb = odt_clampminf3(rgb, 0.0f);
    } else if (eotf == 5) {
        rgb = tess_eotf_hlg(rgb, 0);
    }
    return rgb;
}

// ---- Forward -----------------------------------------------------------------------
LDT_FUNC float3 cmg_forward(float3 rgb, constant ChromagnonParams* p) {
    // 6A addition: linearize the input (identity when in_oetf == 0)
    rgb = tess_linearize(rgb, p->in_oetf);

    // Convert into display gamut
    rgb = tess_vdot9(p->in_to_xyz, rgb);
    rgb = tess_vdot9(p->xyz_to_display, rgb);

    // Offset
    rgb.x += p->off;
    rgb.y += p->off;
    rgb.z += p->off;

    // Purity compress
    if (p->pc_p < 0.01f || p->pc_t0 < 0.001f) {
        rgb = odt_clampminf3(rgb, 0.0f);
    } else {
        float mx = _fmaxf(rgb.x, _fmaxf(rgb.y, rgb.z));
        rgb = odt_sdivf3f(rgb, mx);
        rgb.x = cmg_compress_powerptoe(rgb.x, p->pc_p, p->pc_m, p->pc_t0, 0);
        rgb.y = cmg_compress_powerptoe(rgb.y, p->pc_p, p->pc_m, p->pc_t0, 0);
        rgb.z = cmg_compress_powerptoe(rgb.z, p->pc_p, p->pc_m, p->pc_t0, 0);
        rgb = rgb*mx;
    }

    // Rendering matrix
    rgb = tess_vdot9(p->rnd_mtx, rgb);

    // Clamp negatives
    rgb = odt_clampminf3(rgb, 0.0f);

    // Look matrix in log
    rgb = cmg_xlog(rgb, 0);
    rgb = tess_vdot9(p->look_mtx, rgb);
    rgb = cmg_xlog(rgb, 1);

    // Tonescale
    rgb.x = cmg_contrast(rgb.x, p->con, p->c_s0, p->c_t0, p->c_o0, p->c_m0, 0);
    rgb.y = cmg_contrast(rgb.y, p->con, p->c_s0, p->c_t0, p->c_o0, p->c_m0, 0);
    rgb.z = cmg_contrast(rgb.z, p->con, p->c_s0, p->c_t0, p->c_o0, p->c_m0, 0);

    rgb.x = cmg_tonescale(rgb.x, p->ts_m_r, p->ts_s_r, 0);
    rgb.y = cmg_tonescale(rgb.y, p->ts_m_g, p->ts_s_g, 0);
    rgb.z = cmg_tonescale(rgb.z, p->ts_m_b, p->ts_s_b, 0);

    // Display scale
    rgb = rgb*p->ds;

    // Clamp
    if (p->clamp == 1)
        rgb = odt_clampf3(rgb, 0.0f, p->clamp_max);

    // Display encode
    rgb = tess_display_encode(rgb, p->eotf, p->eotf_p);
    return rgb;
}

// ---- Inverse -----------------------------------------------------------------------
LDT_FUNC float3 cmg_inverse(float3 rgb, constant ChromagnonParams* p) {
    rgb = odt_clampf3(rgb, 0.0f, 1.0f);

    // Display decode
    rgb = cmg_display_decode(rgb, p->eotf, p->eotf_p);

    // Inverse display scale
    rgb.x = rgb.x/p->ds;
    rgb.y = rgb.y/p->ds;
    rgb.z = rgb.z/p->ds;

    // Invert tonescale
    rgb.x = cmg_tonescale(rgb.x, p->ts_m_r, p->ts_s_r, 1);
    rgb.y = cmg_tonescale(rgb.y, p->ts_m_g, p->ts_s_g, 1);
    rgb.z = cmg_tonescale(rgb.z, p->ts_m_b, p->ts_s_b, 1);
    rgb.x = cmg_contrast(rgb.x, p->con, p->c_s0, p->c_t0, p->c_o0, p->c_m0, 1);
    rgb.y = cmg_contrast(rgb.y, p->con, p->c_s0, p->c_t0, p->c_o0, p->c_m0, 1);
    rgb.z = cmg_contrast(rgb.z, p->con, p->c_s0, p->c_t0, p->c_o0, p->c_m0, 1);

    // Inverse look matrix in XLog
    rgb = cmg_xlog(rgb, 0);
    rgb = tess_vdot9(p->look_mtx_inv, rgb);
    rgb = cmg_xlog(rgb, 1);

    // Inverse rendering matrix
    rgb = tess_vdot9(p->rnd_mtx_inv, rgb);

    // Inverse purity compress, gated exactly like the forward path (core deviation).
    if (!(p->pc_p < 0.01f || p->pc_t0 < 0.001f)) {
        float mx = _fmaxf(rgb.x, _fmaxf(rgb.y, rgb.z));
        rgb = odt_sdivf3f(rgb, mx);
        rgb.x = cmg_compress_powerptoe(rgb.x, p->pc_p, p->pc_m, p->pc_t0, 1);
        rgb.y = cmg_compress_powerptoe(rgb.y, p->pc_p, p->pc_m, p->pc_t0, 1);
        rgb.z = cmg_compress_powerptoe(rgb.z, p->pc_p, p->pc_m, p->pc_t0, 1);
        rgb = rgb*mx;
    }

    // Inverse offset
    rgb.x -= p->off;
    rgb.y -= p->off;
    rgb.z -= p->off;

    // Display gamut back to scene gamut
    rgb = tess_vdot9(p->display_to_xyz, rgb);
    rgb = tess_vdot9(p->xyz_to_in, rgb);
    return rgb;
}

// ---- Per-pixel entry point (chromagnon_processPixel) -------------------------------
LDT_FUNC float3 chromagnon_processPixel(float3 rgb_in, int x, int y, int w, int h,
                                        constant ChromagnonParams* p) {
    if (p->direction == 1) return cmg_inverse(rgb_in, p);
    return cmg_forward(rgb_in, p);
}

// ---- Kernel ------------------------------------------------------------------------
kernel void ChromagnonKernel(constant ChromagnonParams* p_Params [[buffer(0)]],
                             device const float4*      p_Input  [[buffer(1)]],
                             device float4*            p_Output [[buffer(2)]],
                             constant int4*            p_Dims   [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]])
{
    int srcPitch = p_Dims->x;   // float4 stride (rowBytes/16)
    int dstPitch = p_Dims->y;
    int pw       = p_Dims->z;
    int ph       = p_Dims->w;
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= pw || y >= ph) return;

    float4 src    = p_Input[y * srcPitch + x];
    float3 result = chromagnon_processPixel(src.xyz, x, y, pw, ph, p_Params);
    p_Output[y * dstPitch + x] = float4(result, src.w);
}
)METAL";

// Host entry point.
void RunChromagnonMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                              int p_SrcRowBytes, int p_DstRowBytes,
                              const ChromagnonParams* p_Params,
                              const float* p_Input, float* p_Output)
{
    if (p_CmdQ == nullptr || p_Params == nullptr || p_Input == nullptr || p_Output == nullptr) return;
    if (p_Width <= 0 || p_Height <= 0) return;

    id<MTLCommandQueue> queue  = (__bridge id<MTLCommandQueue>)p_CmdQ;
    id<MTLDevice>       device = queue.device;

    // Pipeline cache lookup.
    id<MTLComputePipelineState> pipelineState;
    {
        std::unique_lock<std::mutex> lock(s_CmgPipelineMutex);
        auto it = s_CmgPipelineCache.find(p_CmdQ);
        if (it == s_CmgPipelineCache.end()) {
            NSError* err = nil;
            MTLCompileOptions* opts = [MTLCompileOptions new];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            // Runtime guard: setMathMode: does not exist on macOS 13/14.
            if (@available(macOS 15.0, *)) {
                opts.mathMode = MTLMathModeFast;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
            }
#else
            opts.fastMathEnabled = YES;
#endif
            id<MTLLibrary> lib = [device newLibraryWithSource:@(kChromagnonMetalSrc)
                                                      options:opts
                                                        error:&err];
            [opts release];
            if (!lib) {
                fprintf(stderr, "[Chromagnon] Metal compile error: %s\n",
                        err.localizedDescription.UTF8String);
                return;
            }
            id<MTLFunction> fn = [lib newFunctionWithName:@"ChromagnonKernel"];
            [lib release];
            if (!fn) {
                fprintf(stderr, "[Chromagnon] Metal function not found\n");
                return;
            }
            pipelineState = [device newComputePipelineStateWithFunction:fn error:&err];
            [fn release];
            if (!pipelineState) {
                fprintf(stderr, "[Chromagnon] Metal pipeline error: %s\n",
                        err.localizedDescription.UTF8String);
                return;
            }
            s_CmgPipelineCache[p_CmdQ] = pipelineState;
        } else {
            pipelineState = it->second;
        }
    }

    // The host passes p_Input / p_Output as id<MTLBuffer> handles: reinterpret,
    // never copy or release them.
    id<MTLBuffer> inBuf  = reinterpret_cast<id<MTLBuffer>>(const_cast<float*>(p_Input));
    id<MTLBuffer> outBuf = reinterpret_cast<id<MTLBuffer>>(p_Output);

    id<MTLBuffer> paramBuf = [device newBufferWithBytes:p_Params
                                                 length:sizeof(ChromagnonParams)
                                                options:MTLResourceStorageModeShared];

    // {srcPitch, dstPitch, width, height}, pitches in float4 units (rowBytes/16).
    int dims[4] = { p_SrcRowBytes / 16, p_DstRowBytes / 16, p_Width, p_Height };
    id<MTLBuffer> dimsBuf = [device newBufferWithBytes:dims
                                                length:sizeof(dims)
                                               options:MTLResourceStorageModeShared];

    id<MTLCommandBuffer>         cmdBuf  = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

    [encoder setComputePipelineState:pipelineState];
    [encoder setBuffer:paramBuf offset:0 atIndex:0];
    [encoder setBuffer:inBuf    offset:0 atIndex:1];
    [encoder setBuffer:outBuf   offset:0 atIndex:2];
    [encoder setBuffer:dimsBuf  offset:0 atIndex:3];

    MTLSize tg   = MTLSizeMake(16, 16, 1);
    MTLSize grid = MTLSizeMake((NSUInteger)p_Width, (NSUInteger)p_Height, 1);
    [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
    [encoder endEncoding];

    [cmdBuf commit];
    [cmdBuf waitUntilCompleted];

    if (cmdBuf.error) {
        fprintf(stderr, "[Chromagnon] Metal execution error: %s\n",
                cmdBuf.error.localizedDescription.UTF8String);
    }

    [paramBuf release];
    [dimsBuf release];
    // inBuf and outBuf are owned by the host; do NOT release them.
}
