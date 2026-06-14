// OpenDRTMetal.mm — Metal backend for "6A. OpenDRT".
//
// The MSL kernel below is a hand-maintained twin of the shared OpenDRT device
// path (core/OpenDRTAlgorithm.h) over the suite's float3/math aliases. It is
// compiled once per MTLCommandQueue at runtime (Resolve ships no .metallib) and
// the pipeline state is cached. OpenDRTParams + OpenDRT_HueCompressionParams are
// duplicated in MSL with a byte-identical layout — keep the two in lockstep.
//
// The #define aliases at the top of kOpenDRTMetalSrc make the function bodies
// TEXTUALLY IDENTICAL to the C++/CUDA source so the twin can be diffed by eye.
// Metal's float3 has native +,-,* operators, so (unlike the host/CUDA path) NO
// float3 operators are declared here. Matrix arrays are passed as thread-space
// arrays (MSL requires an address space on array params).
//
// The host (OpenDRT.cpp render()) has ALREADY called opendrt_precompute() before
// this entry — the display-encoding remap and ts_* constants are filled; the
// kernel only READS them.
//
// GOTCHA (from memory): after ANY signature/struct change, extract this string
// and `xcrun metal -c` it — a host build green proves nothing; Resolve's runtime
// MSL compile fails silently into white frames. No C++ lambdas in MSL. Buffers
// arrive as float* handles -> reinterpret_cast, never newBufferWithBytesNoCopy.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

#include "../core/OpenDRTAlgorithm.h"

// ── Pipeline cache (keyed by MTLCommandQueue pointer) ─────────────────────────
static std::mutex s_PipelineMutex;
static std::unordered_map<void*, id<MTLComputePipelineState>> s_PipelineCache;

// ── Metal shader source (MSL twin of OpenDRTAlgorithm.h device path) ──────────
static const char* kOpenDRTMetalSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

#define LDT_FUNC inline

// Math-builtin aliases so the bodies below are textually identical to the
// C++/CUDA source in core/OpenDRTAlgorithm.h.
#define _powf(x,y)       pow((float)(x),(float)(y))
#define _exp2f(x)        exp2((float)(x))
#define _expf(x)         exp((float)(x))
#define _logf(x)         log((float)(x))
#define _log2f(x)        log2((float)(x))
#define _log10f(x)       log10((float)(x))
#define _fabs(x)         fabs((float)(x))
#define _fmaxf(x,y)      fmax((float)(x),(float)(y))
#define _fminf(x,y)      fmin((float)(x),(float)(y))
#define _saturatef(x)    saturate((float)(x))
#define _sinf(x)         sin((float)(x))
#define _cosf(x)         cos((float)(x))
#define _tanf(x)         tan((float)(x))
#define _acosf(x)        acos((float)(x))
#define _sqrtf(x)        sqrt((float)(x))
#define _atan2f(y,x)     atan2((float)(y),(float)(x))
#define _copysignf(x,y)  copysign((float)(x),(float)(y))
#define _fmod(x,y)       fmod((float)(x),(float)(y))
#define _fmodf(x,y)      fmod((float)(x),(float)(y))
#define make_float2(a,b)   float2(a,b)
#define make_float3(a,b,c) float3(a,b,c)

#define ODT_SQRT3 1.73205080756887729353f
#define ODT_PI    3.14159265358979323846f

// ── Byte-identical twin of OpenDRT_HueCompressionParams ──
struct OpenDRT_HueCompressionParams {
    float3 anchorBaseVectors[6];
    float  anchorRotations[6];
    float  anchorStrengths[6];
    float  anchorFalloffAngles[6];

    int    useSingleAnchorSettings;
    int    gangRGBAnchors;
    int    gangCMYAnchors;
    int    globalRotationEnabled;
    float  globalRotation;
    float  globalStrength;
};

// ── Byte-identical twin of OpenDRTParams (int and float are both 4 bytes) ──
struct OpenDRTParams {
    int inGamut;
    int inOetf;

    float tnLp;
    float tnGb;
    float ptHdr;

    int   clamp;
    float tnLg;
    float tnCon;
    float tnSh;
    float tnToe;
    float tnOff;

    float tnHcon;
    float tnHconPv;
    float tnHconSt;

    float tnLcon;
    float tnLconW;
    float tnLconPc;

    int   cwp;
    float cwpRng;

    float rsSa;
    float rsRw;
    float rsBw;

    float ptR, ptG, ptB;
    float ptRngLow;
    float ptRngHigh;

    float ptmLow;
    float ptmLowSt;
    float ptmHigh;
    float ptmHighSt;

    float brlR, brlG, brlB;
    float brlC, brlM, brlY;
    float brlRng;

    float hsR, hsG, hsB;
    float hsRgbRng;

    float hsC, hsM, hsY;

    float hcR;

    float advHcR;
    float advHcG;
    float advHcB;
    float advHcC;
    float advHcM;
    float advHcY;
    float advHcPower;

    int   filmicMode;
    float filmicDynamicRange;
    int   filmicProjectorSim;
    float filmicSourceStops;
    float filmicTargetStops;
    float filmicStrength;
    int   advHueContrast;
    int   tonescaleMap;
    int   diagnosticsMode;
    int   rgbChipsMode;
    int   betaFeaturesEnable;

    int   displayGamut;
    int   eotf;

    float inputMatrix[9];
    float outputMatrix[9];
    float cwpMatrix[9];
    float xyzToP3Matrix[9];
    float p3ToRec709Matrix[9];

    float oetfParams[8];
    float eotfParams[8];
    int   oetfType;
    int   eotfType;

    float ts_x1;
    float ts_y1;
    float ts_x0;
    float ts_y0;
    float ts_s0;
    float ts_s10;
    float ts_m1;
    float ts_m2;
    float ts_s;
    float ts_dsc;
    float pt_cmp_Lf;
    float s_Lp100;
    float ts_s1;
    float ts_g;
    float ts_c;
    float ts_n;
    float ts_k;
    float ts_ip;
    float ts_cp;
    float ts_w2;
    float ts_t;
    float ts_nd;

    int tnHconEnable;
    int tnLconEnable;
    int ptlEnable;
    int ptmEnable;
    int brlEnable;
    int hsRgbEnable;
    int hsCmyEnable;

    int   hueCompressionEnable;
    float hueAnchorRotations[6];
    float hueAnchorStrengths[6];
    float hueAnchorFalloffAngles[6];
    int   hueUseSingleAnchorSettings;
    int   hueGangRGBAnchors;
    int   hueGangCMYAnchors;
    int   hueGlobalRotationEnabled;
    float hueGlobalRotation;
    float hueGlobalStrength;

    int tnHconUIEnable;
    int tnLconUIEnable;
    int ptlUIEnable;
    int ptmUIEnable;
    int brlUIEnable;
    int hsRgbUIEnable;
    int hsCmyUIEnable;
    int hcUIEnable;

    int tnHconPresetEnable;
    int tnLconPresetEnable;
    int ptlPresetEnable;
    int ptmPresetEnable;
    int brlPresetEnable;
    int hsRgbPresetEnable;
    int hsCmyPresetEnable;
    int hcPresetEnable;

    OpenDRT_HueCompressionParams hueCompression;
};

// ── Shared matrix-multiply helper (CudaKernel.cu vdot) ──
LDT_FUNC float3 odt_vdot(thread const float3 m[3], float3 v) {
    return make_float3(m[0].x*v.x + m[1].x*v.y + m[2].x*v.z,
                       m[0].y*v.x + m[1].y*v.y + m[2].y*v.z,
                       m[0].z*v.x + m[1].z*v.y + m[2].z*v.z);
}

// ── Math helpers (verbatim) ──
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
LDT_FUNC float odt_hypotf3(float3 a) {
    return _sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

LDT_FUNC float odt_compress_toe_cubic(float x, float m, float w, int inv) {
    if (m == 1.0f) return x;
    float x2 = x * x;
    if (inv == 0) {
        return x * (x2 + m * w) / (x2 + w);
    } else {
        float p0 = x2 - 3.0f * m * w;
        float p1 = 2.0f * x2 + 27.0f * w - 9.0f * m * w;
        float p2 = _powf(_sqrtf(x2 * p1 * p1 - 4 * p0 * p0 * p0) / 2.0f + x * p1 / 2.0f, 1.0f / 3.0f);
        return p0 / (3.0f * p2) + p2 / 3.0f + x / 3.0f;
    }
}

LDT_FUNC float odt_compress_hyperbolic_power(float x, float s, float p) {
    return odt_spowf(x / (x + s), p);
}

LDT_FUNC float odt_contrast_high(float x, float p, float pv, float pv_lx, int inv) {
    const float x0 = 0.18f * _powf(2.0f, pv);
    if (x < x0 || p == 1.0f) return x;

    const float o = x0 - x0 / p;
    const float s0 = _powf(x0, 1.0f - p) / p;
    const float x1 = x0 * _powf(2.0f, pv_lx);
    const float k1 = p * s0 * _powf(x1, p) / x1;
    const float y1 = s0 * _powf(x1, p) + o;
    if (inv == 1)
        return x > y1 ? (x - y1) / k1 + x1 : _powf((x - o) / s0, 1.0f / p);
    else
        return x > x1 ? k1 * (x - x1) + y1 : s0 * _powf(x, p) + o;
}

LDT_FUNC float odt_gauss_window(float x, float w) {
    x /= w;
    return _expf(-x * x);
}

LDT_FUNC float odt_hue_offset(float h, float o) {
    return _fmodf(h - o + ODT_PI, 2.0f * ODT_PI) - ODT_PI;
}

LDT_FUNC float odt_compress_toe_quadratic(float x, float toe, int inv) {
    if (toe == 0.0f) return x;
    if (inv == 0) {
        return odt_spowf(x, 2.0f) / (x + toe);
    } else {
        return (x + _sqrtf(x * (4.0f * toe + x))) / 2.0f;
    }
}

LDT_FUNC float odt_complement_power(float x, float p) {
    return 1.0f - odt_spowf(1.0f - x, 1.0f/p);
}

LDT_FUNC float odt_sigmoid_cubic(float x, float s) {
    if (x < 0.0f || x > 1.0f) return 1.0f;
    return 1.0f + s*(1.0f - 3.0f*x*x + 2.0f*x*x*x);
}

LDT_FUNC float odt_softplus(float x, float s, float x0, float y0) {
    if (x > 10.0f*s + y0 || s < 1e-3f) return x;
    float m = 1.0f;
    if (_fabs(y0) > 1e-6f) m = _expf(y0/s);
    m -= _expf(x0/s);
    return s*_logf(_fmaxf(0.0f, m + _expf(x/s)));
}

LDT_FUNC float odt_gamma_contrast(float x, float gamma, float mid_gray) {
    return mid_gray * _powf(_fmaxf(0.0f, x / mid_gray), gamma);
}

// ── MSL-intrinsic equivalents (used by the Hue Anchor Compression block) ──
LDT_FUNC float odt_mix(float x, float y, float a) {
    return x + (y - x) * a;
}
LDT_FUNC float3 odt_mix3(float3 x, float3 y, float a) {
    return make_float3(odt_mix(x.x, y.x, a), odt_mix(x.y, y.y, a), odt_mix(x.z, y.z, a));
}
LDT_FUNC float odt_smoothstep(float e0, float e1, float x) {
    float t = odt_clampf((x - e0) / (e1 - e0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}
LDT_FUNC float odt_dot3(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
LDT_FUNC float odt_length3(float3 a) {
    return _sqrtf(odt_dot3(a, a));
}
LDT_FUNC float3 odt_normalize3(float3 a) {
    float l = odt_length3(a);
    return make_float3(a.x / l, a.y / l, a.z / l);
}
LDT_FUNC float3 odt_cross3(float3 a, float3 b) {
    return make_float3(a.y*b.z - a.z*b.y,
                       a.z*b.x - a.x*b.z,
                       a.x*b.y - a.y*b.x);
}

// ── OETF (linearization) functions (verbatim) ──
LDT_FUNC float odt_oetf_davinci_intermediate(float x) {
    return x <= 0.02740668f ? x/10.44426855f : _exp2f(x/0.07329248f - 7.0f) - 0.0075f;
}
LDT_FUNC float odt_oetf_filmlight_tlog(float x) {
    return x < 0.075f ? (x-0.075f)/16.184376489665897f : _expf((x - 0.5520126568606655f)/0.09232902596577353f) - 0.0057048244042473785f;
}
LDT_FUNC float odt_oetf_acescct(float x) {
    return x <= 0.155251141552511f ? (x - 0.0729055341958355f)/10.5402377416545f : _exp2f(x*17.52f - 9.72f);
}
LDT_FUNC float odt_oetf_arri_logc3(float x) {
    return x < 5.367655f*0.010591f + 0.092809f ? (x - 0.092809f)/5.367655f : (_powf(10.0f, (x - 0.385537f)/0.247190f) - 0.052272f)/5.555556f;
}
LDT_FUNC float odt_oetf_arri_logc4(float x) {
    return x < -0.7774983977293537f ? x*0.3033266726886969f - 0.7774983977293537f : (_exp2f(14.0f*(x - 0.09286412512218964f)/0.9071358748778103f + 6.0f) - 64.0f)/2231.8263090676883f;
}
LDT_FUNC float odt_oetf_red_log3g10(float x) {
    return x < 0.0f ? (x/15.1927f) - 0.01f : (_powf(10.0f, x/0.224282f) - 1.0f)/155.975327f - 0.01f;
}
LDT_FUNC float odt_oetf_panasonic_vlog(float x) {
    return x < 0.181f ? (x - 0.125f)/5.6f : _powf(10.0f, (x - 0.598206f)/0.241514f) - 0.00873f;
}
LDT_FUNC float odt_oetf_sony_slog3(float x) {
    return x < 171.2102946929f/1023.0f ? (x*1023.0f - 95.0f)*0.01125f/(171.2102946929f - 95.0f) : (_powf(10.0f, ((x*1023.0f - 420.0f)/261.5f))*(0.18f + 0.01f) - 0.01f);
}
LDT_FUNC float odt_oetf_fujifilm_flog2(float x) {
    return x < 0.100686685370811f ? (x - 0.092864f)/8.799461f : (_powf(10.0f, ((x - 0.384316f)/0.245281f))/5.555556f - 0.064829f/5.555556f);
}

LDT_FUNC float odt_apply_oetf(float x, int type) {
    switch(type) {
        case 0: return x; // Linear
        case 1: return odt_oetf_davinci_intermediate(x);
        case 2: return odt_oetf_filmlight_tlog(x);
        case 3: return odt_oetf_acescct(x);
        case 4: return odt_oetf_arri_logc3(x);
        case 5: return odt_oetf_arri_logc4(x);
        case 6: return odt_oetf_red_log3g10(x);
        case 7: return odt_oetf_panasonic_vlog(x);
        case 8: return odt_oetf_sony_slog3(x);
        case 9: return odt_oetf_fujifilm_flog2(x);
        default: return x;
    }
}

// ── EOTF (display encoding) functions (verbatim) ──
LDT_FUNC float3 odt_eotf_hlg(float3 rgb, int inverse) {
    if (inverse == 1) {
        float Yd = 0.2627f*rgb.x + 0.6780f*rgb.y + 0.0593f*rgb.z;
        rgb = rgb*_powf(Yd, (1.0f - 1.2f)/1.2f);
        rgb.x = rgb.x <= 1.0f/12.0f ? _sqrtf(3.0f*rgb.x) : 0.17883277f*_logf(12.0f*rgb.x - 0.28466892f) + 0.55991073f;
        rgb.y = rgb.y <= 1.0f/12.0f ? _sqrtf(3.0f*rgb.y) : 0.17883277f*_logf(12.0f*rgb.y - 0.28466892f) + 0.55991073f;
        rgb.z = rgb.z <= 1.0f/12.0f ? _sqrtf(3.0f*rgb.z) : 0.17883277f*_logf(12.0f*rgb.z - 0.28466892f) + 0.55991073f;
    } else {
        rgb.x = rgb.x <= 0.5f ? rgb.x*rgb.x/3.0f : (_expf((rgb.x - 0.55991073f)/0.17883277f) + 0.28466892f)/12.0f;
        rgb.y = rgb.y <= 0.5f ? rgb.y*rgb.y/3.0f : (_expf((rgb.y - 0.55991073f)/0.17883277f) + 0.28466892f)/12.0f;
        rgb.z = rgb.z <= 0.5f ? rgb.z*rgb.z/3.0f : (_expf((rgb.z - 0.55991073f)/0.17883277f) + 0.28466892f)/12.0f;
        float Ys = 0.2627f*rgb.x + 0.6780f*rgb.y + 0.0593f*rgb.z;
        rgb = rgb*_powf(Ys, 1.2f - 1.0f);
    }
    return rgb;
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

LDT_FUNC float odt_apply_eotf(float x, int type, int inverse) {
    switch(type) {
        case 0: return x; // Linear
        case 1: return inverse ? _powf(_fmaxf(0.0f, x), 1.0f/2.2f) : _powf(_fmaxf(0.0f, x), 2.2f);
        case 2: return inverse ? _powf(_fmaxf(0.0f, x), 1.0f/2.4f) : _powf(_fmaxf(0.0f, x), 2.4f);
        case 3: return inverse ? _powf(_fmaxf(0.0f, x), 1.0f/2.6f) : _powf(_fmaxf(0.0f, x), 2.6f);
        case 4: return x; // PQ - handled separately
        case 5: return x; // HLG - handled separately
        default: return x;
    }
}

LDT_FUNC float3 odt_encode_for_display(float3 rgb, int eotfType) {
    switch(eotfType) {
        case 4: return odt_eotf_pq(rgb, 1);
        case 5: return odt_eotf_hlg(rgb, 1);
        default:
            rgb.x = odt_apply_eotf(rgb.x, eotfType, 1);
            rgb.y = odt_apply_eotf(rgb.y, eotfType, 1);
            rgb.z = odt_apply_eotf(rgb.z, eotfType, 1);
            return rgb;
    }
}

LDT_FUNC float3 odt_linearize(float3 rgb, int tf) {
    if (tf == 0) {
        return rgb;
    } else {
        rgb.x = odt_apply_oetf(rgb.x, tf);
        rgb.y = odt_apply_oetf(rgb.y, tf);
        rgb.z = odt_apply_oetf(rgb.z, tf);
        return rgb;
    }
}

// ── Hardcoded gamut matrices (verbatim switch tables) ──
LDT_FUNC void odt_getInputMatrix(int gamut, thread float3 matrix[3]) {
    switch(gamut) {
        case 0: // XYZ (Identity)
            matrix[0] = make_float3(1.0f, 0.0f, 0.0f);
            matrix[1] = make_float3(0.0f, 1.0f, 0.0f);
            matrix[2] = make_float3(0.0f, 0.0f, 1.0f);
            break;
        case 1: // AP0 to XYZ
            matrix[0] = make_float3(0.93863094875f, 0.338093594922f, 0.000723121511f);
            matrix[1] = make_float3(-0.00574192055f, 0.727213902811f, 0.000818441849f);
            matrix[2] = make_float3(0.017566898852f, -0.065307497733f, 1.0875161874f);
            break;
        case 2: // AP1 to XYZ
            matrix[0] = make_float3(0.652418717672f, 0.268064059194f, -0.00546992851f);
            matrix[1] = make_float3(0.127179925538f, 0.672464478993f, 0.005182799977f);
            matrix[2] = make_float3(0.170857283842f, 0.059471461813f, 1.08934487929f);
            break;
        case 3: // P3-D65 to XYZ
            matrix[0] = make_float3(0.486571133137f, 0.228974640369f, 0.0f);
            matrix[1] = make_float3(0.265667706728f, 0.691738605499f, 0.045113388449f);
            matrix[2] = make_float3(0.198217317462f, 0.079286918044f, 1.043944478035f);
            break;
        case 4: // Rec.2020 to XYZ
            matrix[0] = make_float3(0.636958122253f, 0.262700229883f, 0.0f);
            matrix[1] = make_float3(0.144616916776f, 0.677998125553f, 0.028072696179f);
            matrix[2] = make_float3(0.168880969286f, 0.059301715344f, 1.060985088348f);
            break;
        case 5: // Rec.709 to XYZ
            matrix[0] = make_float3(0.412390917540f, 0.212639078498f, 0.019330825657f);
            matrix[1] = make_float3(0.357584357262f, 0.715168714523f, 0.119194783270f);
            matrix[2] = make_float3(0.180480793118f, 0.072192311287f, 0.950532138348f);
            break;
        default:
            matrix[0] = make_float3(1.0f, 0.0f, 0.0f);
            matrix[1] = make_float3(0.0f, 1.0f, 0.0f);
            matrix[2] = make_float3(0.0f, 0.0f, 1.0f);
            break;
    }
}

LDT_FUNC void odt_getOutputMatrix(int displayGamut, thread float3 matrix[3]) {
    switch(displayGamut) {
        case 0: // P3 to Rec.709
            matrix[0] = make_float3(1.224940181f, -0.04205697775f, -0.01963755488f);
            matrix[1] = make_float3(-0.2249402404f, 1.042057037f, -0.07863604277f);
            matrix[2] = make_float3(0.0f, -1.4901e-08f, 1.098273635f);
            break;
        case 1: // P3 Identity
            matrix[0] = make_float3(1.0f, 0.0f, 0.0f);
            matrix[1] = make_float3(0.0f, 1.0f, 0.0f);
            matrix[2] = make_float3(0.0f, 0.0f, 1.0f);
            break;
        case 2: // P3 to Rec.2020
            matrix[0] = make_float3(0.7538330344f, 0.04574384897f, -0.001210340355f);
            matrix[1] = make_float3(0.1985973691f, 0.9417772198f, 0.0176017173f);
            matrix[2] = make_float3(0.04756959659f, 0.01247893122f, 0.9836086231f);
            break;
        default:
            matrix[0] = make_float3(1.0f, 0.0f, 0.0f);
            matrix[1] = make_float3(0.0f, 1.0f, 0.0f);
            matrix[2] = make_float3(0.0f, 0.0f, 1.0f);
            break;
    }
}

LDT_FUNC void odt_getCreativeWhitepointMatrix(int displayGamut, int cwp, thread float3 matrix[3]) {
    if (displayGamut == 0) { // Rec.709
        switch(cwp) {
            case 1: // D60
                matrix[0] = make_float3(1.189986856f, -0.04168263635f, -0.01937995127f);
                matrix[1] = make_float3(-0.192168414f, 0.9927757018f, -0.07933006919f);
                matrix[2] = make_float3(0.002185496045f, -5.5660878e-05f, 0.9734397041f);
                break;
            case 2: // D55
                matrix[0] = make_float3(1.149327514f, -0.0412590771f, -0.01900949528f);
                matrix[1] = make_float3(-0.1536910745f, 0.9351717477f, -0.07928282823f);
                matrix[2] = make_float3(0.004366526746f, -0.000116126221f, 0.8437884317f);
                break;
            case 3: // D50
                matrix[0] = make_float3(1.103807322f, -0.04079386701f, -0.01854055914f);
                matrix[1] = make_float3(-0.1103425121f, 0.8704694227f, -0.07857582481f);
                matrix[2] = make_float3(0.006531676079f, -0.000180522628f, 0.7105498861f);
                break;
            default: // D65 (Identity)
                matrix[0] = make_float3(1.0f, 0.0f, 0.0f);
                matrix[1] = make_float3(0.0f, 1.0f, 0.0f);
                matrix[2] = make_float3(0.0f, 0.0f, 1.0f);
                break;
        }
    } else { // P3 and Rec.2020
        switch(cwp) {
            case 1: // D60
                matrix[0] = make_float3(0.979832881f, -0.000805359793f, -0.000338382322f);
                matrix[1] = make_float3(0.01836378979f, 0.9618000331f, -0.003671835795f);
                matrix[2] = make_float3(0.001803284786f, 1.8876121e-05f, 0.894139105f);
                break;
            case 2: // D55
                matrix[0] = make_float3(0.9559790976f, -0.001771929896f, -0.000674760809f);
                matrix[1] = make_float3(0.0403850003f, 0.9163058305f, -0.0072466358f);
                matrix[2] = make_float3(0.003639287409f, 3.3300759e-05f, 0.7831189153f);
                break;
            case 3: // D50
                matrix[0] = make_float3(0.9287127388f, -0.002887159176f, -0.001009551548f);
                matrix[1] = make_float3(0.06578032793f, 0.8640709228f, -0.01073503317f);
                matrix[2] = make_float3(0.005506708345f, 4.3593718e-05f, 0.6672692039f);
                break;
            default: // D65 (Identity)
                matrix[0] = make_float3(1.0f, 0.0f, 0.0f);
                matrix[1] = make_float3(0.0f, 1.0f, 0.0f);
                matrix[2] = make_float3(0.0f, 0.0f, 1.0f);
                break;
        }
    }
}

// ── Per-pixel entry point (twin of opendrt_processPixel) ──
LDT_FUNC float3 opendrt_processPixel(float3 in_, int x, int y, int w, int h,
                                     constant OpenDRTParams* params)
{
    float3 rgb = in_;

    if (params->diagnosticsMode == 1 && y < 100) {
        float ramp = (float)x / (float)(w - 1);
        rgb = make_float3(ramp, ramp, ramp);
    }

    if (params->rgbChipsMode == 1) {
        float ramp = (float)x / (float)(w - 1);
        int band = y * 7 / h;
        switch (band) {
            case 0: rgb = make_float3(ramp, 0.0f, 0.0f); break;
            case 1: rgb = make_float3(ramp, ramp, 0.0f); break;
            case 2: rgb = make_float3(0.0f, ramp, 0.0f); break;
            case 3: rgb = make_float3(0.0f, ramp, ramp); break;
            case 4: rgb = make_float3(0.0f, 0.0f, ramp); break;
            case 5: rgb = make_float3(ramp, 0.0f, ramp); break;
            case 6: rgb = make_float3(ramp, ramp, ramp); break;
            default: rgb = make_float3(ramp, ramp, ramp); break;
        }
    }

    rgb = odt_linearize(rgb, params->inOetf);

    float3 inputMatrix[3];
    float3 outputMatrix[3];
    float3 cwpMatrix[3];
    odt_getInputMatrix(params->inGamut, inputMatrix);
    odt_getOutputMatrix(params->displayGamut, outputMatrix);
    odt_getCreativeWhitepointMatrix(params->displayGamut, params->cwp, cwpMatrix);

    rgb = odt_vdot(inputMatrix, rgb);

    if (params->hueCompressionEnable == 1) {
        float3 anchorBaseVectors[6];
        anchorBaseVectors[0] = make_float3(1.0f, 0.0f, 0.0f);
        anchorBaseVectors[1] = make_float3(0.0f, 1.0f, 0.0f);
        anchorBaseVectors[2] = make_float3(0.0f, 0.0f, 1.0f);
        anchorBaseVectors[3] = make_float3(0.0f, 1.0f, 1.0f);
        anchorBaseVectors[4] = make_float3(1.0f, 0.0f, 1.0f);
        anchorBaseVectors[5] = make_float3(1.0f, 1.0f, 0.0f);

        float3 grayAxis = odt_normalize3(make_float3(1.0f, 1.0f, 1.0f));
        float3 rotatedAnchors[6];
        float strengths[6];
        float cosFalloff[6];

        for (int i = 0; i < 6; ++i) {
            int sourceIndex = i;
            if (params->hueUseSingleAnchorSettings == 1) sourceIndex = 0;
            else if (params->hueGangRGBAnchors == 1 && i < 3) sourceIndex = 0;
            else if (params->hueGangCMYAnchors == 1 && i >= 3) sourceIndex = 3;

            float3 base    = anchorBaseVectors[sourceIndex];
            float rotation = params->hueAnchorRotations[sourceIndex];
            float strength = params->hueAnchorStrengths[sourceIndex];
            float falloffDeg = params->hueAnchorFalloffAngles[sourceIndex];

            if (params->hueGlobalRotationEnabled == 1)
                rotation += params->hueGlobalRotation;

            strength *= params->hueGlobalStrength;

            float theta = rotation * ODT_PI / 180.0f;
            float c = _cosf(theta);
            float s = _sinf(theta);
            float dotProd = odt_dot3(grayAxis, base);
            rotatedAnchors[i] = base * c + odt_cross3(grayAxis, base) * s + grayAxis * dotProd * (1.0f - c);
            strengths[i] = strength;
            cosFalloff[i] = _cosf(falloffDeg * ODT_PI / 180.0f);
        }

        float3 color = rgb;
        float lum = odt_dot3(color, make_float3(0.2126f, 0.7152f, 0.0722f));
        float3 chroma = make_float3(color.x - lum, color.y - lum, color.z - lum);

        if (odt_length3(chroma) > 1e-3f) {
            float3 chromaDir = odt_normalize3(chroma);
            float chromaMag = odt_length3(chroma);
            float3 bestDir = chromaDir;
            float bestStrength = 0.0f;

            for (int i = 0; i < 6; ++i) {
                float alignment = odt_dot3(chromaDir, odt_normalize3(rotatedAnchors[i]));
                if (alignment > cosFalloff[i]) {
                    float coneFalloff = odt_smoothstep(cosFalloff[i], 1.0f, alignment);
                    float influence = coneFalloff * strengths[i];
                    if (influence > bestStrength) {
                        bestStrength = influence;
                        bestDir = odt_mix3(chromaDir, odt_normalize3(rotatedAnchors[i]), influence);
                    }
                }
            }

            bestDir = odt_normalize3(bestDir);
            float3 finalColor = make_float3(lum + bestDir.x * chromaMag,
                                            lum + bestDir.y * chromaMag,
                                            lum + bestDir.z * chromaMag);

            rgb.x = odt_clampf(finalColor.x, 0.0f, 1.0f);
            rgb.y = odt_clampf(finalColor.y, 0.0f, 1.0f);
            rgb.z = odt_clampf(finalColor.z, 0.0f, 1.0f);
        }
    }

    float3 hardcodedxyzToP3Matrix[3];
    hardcodedxyzToP3Matrix[0] = make_float3( 2.49349691194f, -0.829488694668f,  0.0358458302915f);
    hardcodedxyzToP3Matrix[1] = make_float3(-0.931383617919f, 1.76266097069f,  -0.0761723891287f);
    hardcodedxyzToP3Matrix[2] = make_float3(-0.402710784451f, 0.0236246771724f, 0.956884503364f);
    rgb = odt_vdot(hardcodedxyzToP3Matrix, rgb);

    float crv_val = 0.0f;
    float2 pos = make_float2((float)x, (float)y);
    float2 res = make_float2((float)w, (float)h);

    if (params->tonescaleMap == 1) {
        crv_val = odt_oetf_filmlight_tlog(pos.x/res.x);
    }

    float3 rs_w = make_float3(params->rsRw, 1.0f - params->rsRw - params->rsBw, params->rsBw);
    float sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;
    rgb = make_float3(sat_L*params->rsSa + rgb.x*(1.0f - params->rsSa),
                      sat_L*params->rsSa + rgb.y*(1.0f - params->rsSa),
                      sat_L*params->rsSa + rgb.z*(1.0f - params->rsSa));

    rgb = rgb + make_float3(params->tnOff, params->tnOff, params->tnOff);
    if (params->tonescaleMap == 1) crv_val += params->tnOff;

    if (params->tnLconPresetEnable || params->tnLconUIEnable) {
        float mcon_m = _powf(2.0f, -params->tnLcon);
        float mcon_w = params->tnLconW/4.0f;
        mcon_w *= mcon_w;

        const float mcon_cnst_sc = odt_compress_toe_cubic(params->ts_x0, mcon_m, mcon_w, 1)/params->ts_x0;
        rgb = rgb * mcon_cnst_sc;

        float mcon_nm = odt_hypotf3(odt_clampminf3(rgb, 0.0f))/ODT_SQRT3;
        float mcon_sc = (mcon_nm*mcon_nm + mcon_m*mcon_w)/(mcon_nm*mcon_nm + mcon_w);

        if (params->tnLconPc > 0.0f) {
            float3 mcon_rgb = rgb;
            mcon_rgb.x = odt_compress_toe_cubic(rgb.x, mcon_m, mcon_w, 0);
            mcon_rgb.y = odt_compress_toe_cubic(rgb.y, mcon_m, mcon_w, 0);
            mcon_rgb.z = odt_compress_toe_cubic(rgb.z, mcon_m, mcon_w, 0);

            float mcon_mx = odt_fmaxf3(rgb);
            float mcon_mn = odt_fminf3(rgb);
            float mcon_ch = odt_clampf(1.0f - odt_sdivf(mcon_mn, mcon_mx), 0.0, 1.0);
            mcon_ch = _powf(mcon_ch, 4.0f*params->tnLconPc);
            rgb = make_float3(mcon_sc*rgb.x*mcon_ch + mcon_rgb.x*(1.0f - mcon_ch),
                              mcon_sc*rgb.y*mcon_ch + mcon_rgb.y*(1.0f - mcon_ch),
                              mcon_sc*rgb.z*mcon_ch + mcon_rgb.z*(1.0f - mcon_ch));
        }
        else {
            rgb = rgb * mcon_sc;
        }

        if (params->tonescaleMap == 1) {
            crv_val *= mcon_cnst_sc;
            crv_val = crv_val*(crv_val*crv_val + mcon_m*mcon_w)/(crv_val*crv_val + mcon_w);
        }
    }

    if (params->filmicMode == 1 && params->betaFeaturesEnable == 1) {
        float maxInput = _powf(2.0f, params->filmicSourceStops);
        float3 normalizedRGB = make_float3(rgb.x / maxInput, rgb.y / maxInput, rgb.z / maxInput);

        float rolloff_s = 0.05f + (params->filmicDynamicRange / 10.0f);
        float rolloff_p = 0.8f + (params->filmicDynamicRange / 25.0f);

        float3 compressedRGB = make_float3(
            odt_compress_hyperbolic_power(normalizedRGB.x, rolloff_s, rolloff_p),
            odt_compress_hyperbolic_power(normalizedRGB.y, rolloff_s, rolloff_p),
            odt_compress_hyperbolic_power(normalizedRGB.z, rolloff_s, rolloff_p));

        float maxOutput = _powf(2.0f, params->filmicTargetStops);
        float3 rescaledRGB = compressedRGB * maxOutput;

        rgb = make_float3(rgb.x * (1.0f - params->filmicStrength) + rescaledRGB.x * params->filmicStrength,
                          rgb.y * (1.0f - params->filmicStrength) + rescaledRGB.y * params->filmicStrength,
                          rgb.z * (1.0f - params->filmicStrength) + rescaledRGB.z * params->filmicStrength);

        if (params->tonescaleMap == 1) {
            float crv_normalized = crv_val / maxInput;
            float crv_compressed = odt_compress_hyperbolic_power(crv_normalized, rolloff_s, rolloff_p);
            float crv_rescaled = crv_compressed * maxOutput;
            crv_val = crv_val * (1.0f - params->filmicStrength) + crv_rescaled * params->filmicStrength;
        }
    }

    float tsn = odt_hypotf3(odt_clampminf3(rgb, 0.0f)) / ODT_SQRT3;
    float ts_pt = _sqrtf(_fmaxf(0.0f, rgb.x * rgb.x * params->ptR + rgb.y * rgb.y * params->ptG + rgb.z * rgb.z * params->ptB));

    rgb = odt_sdivf3f(odt_clampminf3(rgb, -2.0f), tsn);

    if (params->tnHconPresetEnable || params->tnHconUIEnable) {
        float hcon_p = _powf(2.0f, params->tnHcon);
        tsn = odt_contrast_high(tsn, hcon_p, params->tnHconPv, params->tnHconSt, 0);
        ts_pt = odt_contrast_high(ts_pt, hcon_p, params->tnHconPv, params->tnHconSt, 0);
        if (params->tonescaleMap == 1) crv_val = odt_contrast_high(crv_val, hcon_p, params->tnHconPv, params->tnHconSt, 0);
    }

    if (params->advHueContrast == 1) {
        float3 cym_contrast = make_float3(1.0f - rgb.x, 1.0f - rgb.y, 1.0f - rgb.z);
        float3 rgb_contrast = rgb;
        float pivot = 1.0f;
        rgb_contrast.x = odt_gamma_contrast(rgb_contrast.x, params->advHcR, pivot);
        cym_contrast.x = odt_gamma_contrast(cym_contrast.x, params->advHcC, pivot);
        rgb_contrast.y = odt_gamma_contrast(rgb_contrast.y, params->advHcG, pivot);
        cym_contrast.y = odt_gamma_contrast(cym_contrast.y, params->advHcM, pivot);
        rgb_contrast.z = odt_gamma_contrast(rgb_contrast.z, params->advHcB, pivot);
        cym_contrast.z = odt_gamma_contrast(cym_contrast.z, params->advHcY, pivot);

        rgb_contrast.x = odt_mix(rgb_contrast.x, 1.0f - cym_contrast.x, 0.5f);
        rgb_contrast.y = odt_mix(rgb_contrast.y, 1.0f - cym_contrast.y, 0.5f);
        rgb_contrast.z = odt_mix(rgb_contrast.z, 1.0f - cym_contrast.z, 0.5f);

        rgb = odt_mix3(rgb, rgb_contrast, params->advHcPower);
    }

    tsn = odt_compress_hyperbolic_power(tsn, params->ts_s, params->tnCon);
    ts_pt = odt_compress_hyperbolic_power(ts_pt, params->ts_s1, params->tnCon);

    if (params->tonescaleMap == 1) crv_val = odt_compress_hyperbolic_power(crv_val, params->ts_s, params->tnCon);

    float opp_cy = rgb.x - rgb.z;
    float opp_gm = rgb.y - (rgb.x + rgb.z)/2.0f;
    float ach_d = _sqrtf(_fmaxf(0.0f, opp_cy*opp_cy + opp_gm*opp_gm))/ODT_SQRT3;

    ach_d = (1.25f)*odt_compress_toe_quadratic(ach_d, 0.25f, 0);

    float hue = _fmodf(_atan2f(opp_cy, opp_gm) + ODT_PI + 1.10714931f, 2.0f*ODT_PI);

    float3 ha_rgb = make_float3(
        odt_gauss_window(odt_hue_offset(hue, 0.1f), 0.9f),
        odt_gauss_window(odt_hue_offset(hue, 4.3f), 0.9f),
        odt_gauss_window(odt_hue_offset(hue, 2.3f), 0.9f));

    float3 ha_cmy = make_float3(
        odt_gauss_window(odt_hue_offset(hue, 3.3f), 0.6f),
        odt_gauss_window(odt_hue_offset(hue, 1.3f), 0.6f),
        odt_gauss_window(odt_hue_offset(hue, -1.2f), 0.6f));

    float ts_pt_cmp = 1.0f - _powf(ts_pt, 1.0f/params->ptRngLow);

    float pt_rng_high_f = _fminf(1.0f, ach_d/1.2f);
    pt_rng_high_f *= pt_rng_high_f;
    pt_rng_high_f = params->ptRngHigh < 1.0f ? 1.0f - pt_rng_high_f : pt_rng_high_f;
    ts_pt_cmp = _powf(ts_pt_cmp, params->ptRngHigh)*(1.0f - pt_rng_high_f) + ts_pt_cmp*pt_rng_high_f;

    float brl_f = 1.0f;
    if (params->brlPresetEnable || params->brlUIEnable) {
        brl_f = -params->brlR*ha_rgb.x - params->brlG*ha_rgb.y - params->brlB*ha_rgb.z - params->brlC*ha_cmy.x - params->brlM*ha_cmy.y - params->brlY*ha_cmy.z;
        brl_f = (1.0f - ach_d)*brl_f + 1.0f - brl_f;
        brl_f = odt_softplus(brl_f, 0.25f, -100.0f, 0.0f);

        float brl_ts = brl_f > 1.0f ? 1.0f - ts_pt : ts_pt;
        float brl_lim = odt_spowf(brl_ts, 1.0f - params->brlRng);
        brl_f = brl_f*brl_lim + 1.0f - brl_lim;
        brl_f = _fmaxf(0.0f, _fminf(2.0f, brl_f));
    }

    float ptm_sc = 1.0f;
    if (params->ptmPresetEnable || params->ptmUIEnable) {
        float ptm_ach_d = odt_complement_power(ach_d, params->ptmLowSt);
        ptm_sc = odt_sigmoid_cubic(ptm_ach_d, params->ptmLow*(1.0f - ts_pt));

        ptm_ach_d = odt_complement_power(ach_d, params->ptmHighSt)*(1.0f - ts_pt) + ach_d*ach_d*ts_pt;
        ptm_sc *= odt_sigmoid_cubic(ptm_ach_d, params->ptmHigh*ts_pt);
        ptm_sc = _fmaxf(0.0f, ptm_sc);
    }

    ha_rgb = ha_rgb * ach_d;
    ha_cmy = ha_cmy * ((1.5f)*odt_compress_toe_quadratic(ach_d, 0.5f, 0));

    if (params->hcPresetEnable || params->hcUIEnable) {
        float hc_ts = 1.0f - ts_pt;
        float hc_c = (1.0f - ach_d)*hc_ts + ach_d*(1.0f - hc_ts);
        hc_c *= ha_rgb.x;
        hc_ts *= hc_ts;
        float hc_f = params->hcR*(hc_c - 2.0f*hc_c*hc_ts) + 1.0f;
        rgb = make_float3(rgb.x, rgb.y*hc_f, rgb.z*hc_f);
    }

    if (params->hsRgbPresetEnable || params->hsRgbUIEnable) {
        float3 hs_rgb = ha_rgb*_powf(ts_pt, 1.0f/params->hsRgbRng);
        float3 hsf = make_float3(hs_rgb.x*params->hsR, hs_rgb.y*-params->hsG, hs_rgb.z*-params->hsB);
        hsf = make_float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
        rgb = rgb + hsf;
    }

    if (params->hsCmyPresetEnable || params->hsCmyUIEnable) {
        float3 hs_cmy = ha_cmy*(1.0f - ts_pt);
        float3 hsf = make_float3(hs_cmy.x*-params->hsC, hs_cmy.y*params->hsM, hs_cmy.z*params->hsY);
        hsf = make_float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
        rgb = rgb + hsf;
    }

    rgb = rgb * brl_f;

    ts_pt_cmp *= ptm_sc;
    rgb = make_float3(rgb.x*ts_pt_cmp + 1.0f - ts_pt_cmp,
                      rgb.y*ts_pt_cmp + 1.0f - ts_pt_cmp,
                      rgb.z*ts_pt_cmp + 1.0f - ts_pt_cmp);

    sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;
    rgb = make_float3((sat_L*params->rsSa - rgb.x)/(params->rsSa - 1.0f),
                      (sat_L*params->rsSa - rgb.y)/(params->rsSa - 1.0f),
                      (sat_L*params->rsSa - rgb.z)/(params->rsSa - 1.0f));

    float3 cwp_rgb = rgb;
    cwp_rgb = odt_vdot(cwpMatrix, rgb);

    if (params->displayGamut == 0) { // Rec.709
        float3 p3ToRec709D65[3];
        p3ToRec709D65[0] = make_float3(1.224940181f, -0.04205697775f, -0.01963755488f);
        p3ToRec709D65[1] = make_float3(-0.2249402404f, 1.042057037f, -0.07863604277f);
        p3ToRec709D65[2] = make_float3(0.0f, -1.4901e-08f, 1.098273635f);
        rgb = odt_vdot(p3ToRec709D65, rgb);

        if (params->cwp == 0) cwp_rgb = rgb;
    }

    float cwp_f = _powf(tsn, 1.0f - params->cwpRng);
    rgb = make_float3(cwp_rgb.x*cwp_f + rgb.x*(1.0f - cwp_f),
                      cwp_rgb.y*cwp_f + rgb.y*(1.0f - cwp_f),
                      cwp_rgb.z*cwp_f + rgb.z*(1.0f - cwp_f));

    float3 crv_rgb = make_float3(crv_val, crv_val, crv_val);
    float3 crv_rgb_cwp = crv_rgb;
    if (params->tonescaleMap == 1) {
        if (params->displayGamut == 0) { // Rec.709
            crv_rgb_cwp = odt_vdot(cwpMatrix, crv_rgb);
            float3 p3ToRec709[3];
            p3ToRec709[0] = make_float3(1.224940181f, -0.04205697775f, -0.01963755488f);
            p3ToRec709[1] = make_float3(-0.2249402404f, 1.042057037f, -0.07863604277f);
            p3ToRec709[2] = make_float3(0.0f, -1.4901e-08f, 1.098273635f);
            crv_rgb = odt_vdot(p3ToRec709, crv_rgb);
            if (params->cwp == 0) crv_rgb_cwp = crv_rgb;
        }
        else if (params->displayGamut >= 1) {
            crv_rgb_cwp = odt_vdot(cwpMatrix, crv_rgb);
            if (params->cwp == 0) crv_rgb_cwp = crv_rgb;
        }

        float crv_rgb_cwp_f = _powf(crv_val, 1.0f - params->cwpRng);
        crv_rgb = make_float3(crv_rgb_cwp.x*crv_rgb_cwp_f + crv_rgb.x*(1.0f - crv_rgb_cwp_f),
                              crv_rgb_cwp.y*crv_rgb_cwp_f + crv_rgb.y*(1.0f - crv_rgb_cwp_f),
                              crv_rgb_cwp.z*crv_rgb_cwp_f + crv_rgb.z*(1.0f - crv_rgb_cwp_f));
    }

    if (params->ptlPresetEnable || params->ptlUIEnable) {
        float sum0 = odt_softplus(rgb.x, 0.2f, -100.0f, -0.3f) + rgb.y + odt_softplus(rgb.z, 0.2f, -100.0f, -0.3f);
        rgb.x = odt_softplus(rgb.x, 0.04f, -0.3f, 0.0f);
        rgb.y = odt_softplus(rgb.y, 0.06f, -0.3f, 0.0f);
        rgb.z = odt_softplus(rgb.z, 0.01f, -0.05f, 0.0f);

        float ptl_norm = _fminf(1.0f, odt_sdivf(sum0, rgb.x + rgb.y + rgb.z));
        rgb = rgb * ptl_norm;
    }

    tsn *= params->ts_m2;
    tsn = odt_compress_toe_quadratic(tsn, params->tnToe, 0);
    tsn *= params->ts_dsc;

    if (params->tonescaleMap == 1) {
        crv_rgb = crv_rgb * params->ts_m2;
        crv_rgb.x = odt_compress_toe_quadratic(crv_rgb.x, params->tnToe, 0);
        crv_rgb.y = odt_compress_toe_quadratic(crv_rgb.y, params->tnToe, 0);
        crv_rgb.z = odt_compress_toe_quadratic(crv_rgb.z, params->tnToe, 0);
        crv_rgb = crv_rgb * params->ts_dsc;
        if (params->eotf == 4) crv_rgb = crv_rgb * 10.0f;
    }

    rgb = rgb * tsn;

    if (params->clamp != 0) {
        rgb = odt_clampf3(rgb, 0.0f, 1.0f);
    }

    if (params->displayGamut == 2) { // Rec.2020
        rgb = odt_clampminf3(rgb, 0.0f);
        float3 p3ToRec2020[3];
        p3ToRec2020[0] = make_float3(0.7538330344f, 0.04574384897f, -0.001210340355f);
        p3ToRec2020[1] = make_float3(0.1985973691f, 0.9417772198f, 0.0176017173f);
        p3ToRec2020[2] = make_float3(0.04756959659f, 0.01247893122f, 0.9836086231f);
        rgb = odt_vdot(p3ToRec2020, rgb);
    }

    rgb = odt_encode_for_display(rgb, params->eotf);

    if (params->tonescaleMap == 1) {
        if ((params->eotf > 0) && (params->eotf < 4)) {
            float eotf_p = 2.0f + params->eotf * 0.2f;
            crv_rgb = odt_spowf3(crv_rgb, 1.0f/eotf_p);
        }
        else if (params->eotf == 4) crv_rgb = odt_eotf_pq(crv_rgb, 1);
        else if (params->eotf == 5) crv_rgb = odt_eotf_hlg(crv_rgb, 1);

        float3 crv_rgb_dst = make_float3(pos.y-crv_rgb.x*res.y, pos.y-crv_rgb.y*res.y, pos.y-crv_rgb.z*res.y);
        float crv_w0 = 0.05f;
        crv_rgb_dst.x = _expf(-crv_rgb_dst.x*crv_rgb_dst.x*crv_w0);
        crv_rgb_dst.y = _expf(-crv_rgb_dst.y*crv_rgb_dst.y*crv_w0);
        crv_rgb_dst.z = _expf(-crv_rgb_dst.z*crv_rgb_dst.z*crv_w0);
        float crv_lm = params->eotf < 4 ? 1.0f : 0.5f;
        crv_rgb_dst = odt_clampf3(crv_rgb_dst, 0.0f, 1.0f);
        rgb = make_float3(rgb.x * (1.0f - crv_rgb_dst.x) + crv_lm*crv_rgb_dst.x*crv_rgb_dst.x,
                          rgb.y * (1.0f - crv_rgb_dst.y) + crv_lm*crv_rgb_dst.y*crv_rgb_dst.y,
                          rgb.z * (1.0f - crv_rgb_dst.z) + crv_lm*crv_rgb_dst.z*crv_rgb_dst.z);
    }

    return rgb;
}

kernel void OpenDRTKernel(
    constant OpenDRTParams* p_Params  [[ buffer(0) ]],
    device const float4*    p_Input   [[ buffer(1) ]],
    device       float4*    p_Output  [[ buffer(2) ]],
    constant     int4*      p_Dims    [[ buffer(3) ]],
    uint2 gid [[ thread_position_in_grid ]])
{
    int srcPitch = p_Dims->x;   // float4 stride (rowBytes/16)
    int dstPitch = p_Dims->y;
    int pw       = p_Dims->z;
    int ph       = p_Dims->w;
    int x = int(gid.x);
    int y = int(gid.y);
    if (x >= pw || y >= ph) return;

    float4 src    = p_Input[y * srcPitch + x];
    float3 result = opendrt_processPixel(src.xyz, x, y, pw, ph, p_Params);
    p_Output[y * dstPitch + x] = float4(result, src.w);
}
)METAL";

// ── Host entry point ──────────────────────────────────────────────────────────
void RunOpenDRTMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                           int p_SrcRowBytes, int p_DstRowBytes,
                           const OpenDRTParams* p_Params,
                           const float* p_Input, float* p_Output)
{
    id<MTLCommandQueue> queue  = (__bridge id<MTLCommandQueue>)p_CmdQ;
    id<MTLDevice>       device = queue.device;

    // ── Pipeline cache lookup ─────────────────────────────────────────────────
    id<MTLComputePipelineState> pipelineState;
    {
        std::unique_lock<std::mutex> lock(s_PipelineMutex);
        auto it = s_PipelineCache.find(p_CmdQ);
        if (it == s_PipelineCache.end()) {
            NSError* err = nil;
            MTLCompileOptions* opts = [MTLCompileOptions new];
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
            opts.mathMode = MTLMathModeFast;
#else
            opts.fastMathEnabled = YES;
#endif
            id<MTLLibrary> lib = [device newLibraryWithSource:@(kOpenDRTMetalSrc)
                                                      options:opts
                                                        error:&err];
            [opts release];
            if (!lib) {
                fprintf(stderr, "[OpenDRT] Metal compile error: %s\n",
                        err.localizedDescription.UTF8String);
                return;
            }
            id<MTLFunction> fn = [lib newFunctionWithName:@"OpenDRTKernel"];
            [lib release];
            if (!fn) {
                fprintf(stderr, "[OpenDRT] Metal function not found\n");
                return;
            }
            pipelineState = [device newComputePipelineStateWithFunction:fn error:&err];
            [fn release];
            if (!pipelineState) {
                fprintf(stderr, "[OpenDRT] Metal pipeline error: %s\n",
                        err.localizedDescription.UTF8String);
                return;
            }
            s_PipelineCache[p_CmdQ] = pipelineState;
        } else {
            pipelineState = it->second;
        }
    }

    // ── Buffers ───────────────────────────────────────────────────────────────
    // Resolve passes p_Input / p_Output as id<MTLBuffer> handles — reinterpret,
    // never copy or release them.
    id<MTLBuffer> inBuf  = reinterpret_cast<id<MTLBuffer>>(const_cast<float*>(p_Input));
    id<MTLBuffer> outBuf = reinterpret_cast<id<MTLBuffer>>(p_Output);

    id<MTLBuffer> paramBuf = [device newBufferWithBytes:p_Params
                                                 length:sizeof(OpenDRTParams)
                                                options:MTLResourceStorageModeShared];

    // {srcPitch, dstPitch, width, height} — pitches in float4 units (rowBytes/16).
    int dims[4] = { p_SrcRowBytes / 16, p_DstRowBytes / 16, p_Width, p_Height };
    id<MTLBuffer> dimsBuf = [device newBufferWithBytes:dims
                                                length:sizeof(dims)
                                               options:MTLResourceStorageModeShared];

    // ── Dispatch ──────────────────────────────────────────────────────────────
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
        fprintf(stderr, "[OpenDRT] Metal execution error: %s\n",
                cmdBuf.error.localizedDescription.UTF8String);
    }

    [paramBuf release];
    [dimsBuf release];
    // inBuf and outBuf are owned by Resolve — do NOT release them.
}
