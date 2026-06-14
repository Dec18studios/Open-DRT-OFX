#pragma once
// OpenDRTAlgorithm.h — "6A. OpenDRT" per-pixel display transform.
//
// Ported VERBATIM from the canonical golden reference:
//   /Volumes/Server Sync Files/OFX Plugin Git/Open DRT/MetalKernel.mm
// (the device transform string-literal kernel, OpenDRTKernel) plus the host-side
// display-encoding remap (MetalKernel.mm:1310-1352) and the tonescale precompute
// from OpenDRTPresets::calculateTonescaleConstants.
//
// The golden was re-anchored from CudaKernel.cu to MetalKernel.mm after an audit
// found the CUDA path is a stale/incomplete transcription of Metal — it dropped
// the Hue Anchor Compression block (MetalKernel.mm:668-743), the Advanced Hue
// Contrast block (MetalKernel.mm:878-894), and the host display-encoding remap.
// All three are now present below; do NOT re-anchor back to CudaKernel.cu.
//
// Every device function from MetalKernel.mm is reproduced here as LDT_FUNC over
// the suite's shared float3 / math aliases (LookDevPlatform.h), so ONE source
// compiles on CPU, CUDA (__device__), and Metal. Math is preserved bit-for-bit:
// same constants (e.g. ArriLogC3 0.010591), same operation order, same guard
// semantics (spowf/sdivf lossless x<=0 / b==0 returns), ONE shared odt_vdot for
// every gamut matrix multiply so FMA-vs-mul+add rounding is consistent.
//
// Per-pixel entry: opendrt_processPixel(float3 in, x,y,w,h, const OpenDRTParams*).
// Per-frame constants (ts_*) are precomputed host-side in opendrt_precompute()
// (guarded out of the device compilers); the kernel only READS them.
//
// The math aliases (_powf/_expf/_exp2f/_log2f/_sqrtf/_fmaxf/_fminf/_fabs/_fmodf/
// _atan2f) map to fast device builtins under CUDA/Metal and to <cmath> on host.
// The CUDA reference calls powf/expf/etc. directly; on the CUDA backend the
// aliases ARE those same builtins, preserving the golden last-ULP behaviour.

#include "../../../shared/LookDevPlatform.h"
#include "OpenDRTParams.h"

// ── Constants (CudaKernel.cu __constant__) ────────────────────────────────────
#ifndef ODT_SQRT3
#define ODT_SQRT3 1.73205080756887729353f
#endif
#ifndef ODT_PI
#define ODT_PI 3.14159265358979323846f
#endif

// LookDevPlatform.h does not provide a natural-log alias; OpenDRT's softplus and
// HLG OETF need logf. Map it to the device builtin under CUDA/Metal and <cmath>
// on host, matching the alias style of the platform header.
#ifndef _logf
  #if defined(__CUDACC__) || defined(__METAL_VERSION__)
    #define _logf(x) logf((float)(x))
  #else
    #define _logf(x) std::log((float)(x))
  #endif
#endif

// ── Shared matrix-multiply helper (CudaKernel.cu vdot) ─────────────────────────
// ONE helper for every gamut matrix, so the rounding (plain mul+add, no FMA
// fusion forced) matches between the ref harness and this port.
LDT_FUNC float3 odt_vdot(const float3 m[3], float3 v) {
    return make_float3(m[0].x*v.x + m[1].x*v.y + m[2].x*v.z,
                       m[0].y*v.x + m[1].y*v.y + m[2].y*v.z,
                       m[0].z*v.x + m[1].z*v.y + m[2].z*v.z);
}

// ── Math helpers (verbatim from CudaKernel.cu) ─────────────────────────────────
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

// gamma_contrast — Advanced Hue Contrast helper (MetalKernel.mm:248-250).
//   mid_gray * pow(max(0, x/mid_gray), gamma)
LDT_FUNC float odt_gamma_contrast(float x, float gamma, float mid_gray) {
    return mid_gray * _powf(_fmaxf(0.0f, x / mid_gray), gamma);
}

// ── MSL-intrinsic equivalents (used only by the Hue Anchor Compression block) ──
// Reproduce metal::mix / smoothstep / normalize / cross / length / dot exactly.
//   MSL mix(x,y,a) = x + (y - x)*a   (NOT x*(1-a)+y*a — the rounding differs)
LDT_FUNC float odt_mix(float x, float y, float a) {
    return x + (y - x) * a;
}
LDT_FUNC float3 odt_mix3(float3 x, float3 y, float a) {
    return make_float3(odt_mix(x.x, y.x, a), odt_mix(x.y, y.y, a), odt_mix(x.z, y.z, a));
}
// MSL smoothstep(e0,e1,x): t=clamp((x-e0)/(e1-e0),0,1); t*t*(3-2t)
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

// ── OETF (linearization) functions (verbatim) ─────────────────────────────────
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

// ── EOTF (display encoding) functions (verbatim) ──────────────────────────────
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

// ── Hardcoded gamut matrices (verbatim switch tables) ─────────────────────────
LDT_FUNC void odt_getInputMatrix(int gamut, float3 matrix[3]) {
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

LDT_FUNC void odt_getOutputMatrix(int displayGamut, float3 matrix[3]) {
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

LDT_FUNC void odt_getCreativeWhitepointMatrix(int displayGamut, int cwp, float3 matrix[3]) {
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

// ── Per-pixel entry point ─────────────────────────────────────────────────────
// in_   : RGBA's RGB in the selected input gamut/OETF.
// x,y   : pixel coords (diagnostics/overlay). w,h: frame size.
// Mirrors OpenDRTKernel's body lines 399-832 of CudaKernel.cu verbatim.
LDT_FUNC float3 opendrt_processPixel(float3 in_, int x, int y, int w, int h,
                                     const OpenDRTParams* params)
{
    /***************************************************
     setup and extraction
    --------------------------------------------------*/
    float3 rgb = in_;

    // Diagnostics mode: top 100 rows become a grey ramp
    if (params->diagnosticsMode == 1 && y < 100) {
        float ramp = (float)x / (float)(w - 1);
        rgb = make_float3(ramp, ramp, ramp);
    }

    // RGB chips test pattern
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

    // Load dynamic matrices
    float3 inputMatrix[3];
    float3 outputMatrix[3];
    float3 cwpMatrix[3];
    odt_getInputMatrix(params->inGamut, inputMatrix);
    odt_getOutputMatrix(params->displayGamut, outputMatrix);
    odt_getCreativeWhitepointMatrix(params->displayGamut, params->cwp, cwpMatrix);

    // Apply input matrix
    rgb = odt_vdot(inputMatrix, rgb);

    /***************************************************
      Hue Anchor Compression System
      (MetalKernel.mm:668-743 — Rodrigues per-hue anchor rotation)
    --------------------------------------------------*/
    if (params->hueCompressionEnable == 1) {
        // Base anchor vectors (RGB, CMY)
        float3 anchorBaseVectors[6];
        anchorBaseVectors[0] = make_float3(1.0f, 0.0f, 0.0f); // Red
        anchorBaseVectors[1] = make_float3(0.0f, 1.0f, 0.0f); // Green
        anchorBaseVectors[2] = make_float3(0.0f, 0.0f, 1.0f); // Blue
        anchorBaseVectors[3] = make_float3(0.0f, 1.0f, 1.0f); // Cyan
        anchorBaseVectors[4] = make_float3(1.0f, 0.0f, 1.0f); // Magenta
        anchorBaseVectors[5] = make_float3(1.0f, 1.0f, 0.0f); // Yellow

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

            // Rodrigues' rotation around gray axis
            float theta = rotation * ODT_PI / 180.0f;
            float c = _cosf(theta);
            float s = _sinf(theta);
            float dotProd = odt_dot3(grayAxis, base);
            rotatedAnchors[i] = base * c + odt_cross3(grayAxis, base) * s + grayAxis * dotProd * (1.0f - c);
            strengths[i] = strength;
            cosFalloff[i] = _cosf(falloffDeg * ODT_PI / 180.0f);
        }

        // Apply hue compression to RGB
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

    // XYZ to P3 (hardcoded)
    float3 hardcodedxyzToP3Matrix[3];
    hardcodedxyzToP3Matrix[0] = make_float3( 2.49349691194f, -0.829488694668f,  0.0358458302915f);
    hardcodedxyzToP3Matrix[1] = make_float3(-0.931383617919f, 1.76266097069f,  -0.0761723891287f);
    hardcodedxyzToP3Matrix[2] = make_float3(-0.402710784451f, 0.0236246771724f, 0.956884503364f);
    rgb = odt_vdot(hardcodedxyzToP3Matrix, rgb);

    /***************************************************
     Tonescale Overlay Initialization
    --------------------------------------------------*/
    float crv_val = 0.0f;
    float2 pos = make_float2((float)x, (float)y);
    float2 res = make_float2((float)w, (float)h);

    if (params->tonescaleMap == 1) {
        crv_val = odt_oetf_filmlight_tlog(pos.x/res.x);
    }

    // Rendering Space desaturate
    float3 rs_w = make_float3(params->rsRw, 1.0f - params->rsRw - params->rsBw, params->rsBw);
    float sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;
    rgb = make_float3(sat_L*params->rsSa + rgb.x*(1.0f - params->rsSa),
                      sat_L*params->rsSa + rgb.y*(1.0f - params->rsSa),
                      sat_L*params->rsSa + rgb.z*(1.0f - params->rsSa));

    // Offset
    rgb = rgb + make_float3(params->tnOff, params->tnOff, params->tnOff);
    if (params->tonescaleMap == 1) crv_val += params->tnOff;

    /***************************************************
      Contrast Low Module
    --------------------------------------------------*/
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

    /***************************************************
      Filmic Dynamic Range Compression (BETA FEATURE)
    --------------------------------------------------*/
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

    /***************************************************
     Tonescale and RGB Ratios
    --------------------------------------------------*/
    float tsn = odt_hypotf3(odt_clampminf3(rgb, 0.0f)) / ODT_SQRT3;
    float ts_pt = _sqrtf(_fmaxf(0.0f, rgb.x * rgb.x * params->ptR + rgb.y * rgb.y * params->ptG + rgb.z * rgb.z * params->ptB));

    rgb = odt_sdivf3f(odt_clampminf3(rgb, -2.0f), tsn);

    /***************************************************
      Apply High Contrast
    --------------------------------------------------*/
    if (params->tnHconPresetEnable || params->tnHconUIEnable) {
        float hcon_p = _powf(2.0f, params->tnHcon);
        tsn = odt_contrast_high(tsn, hcon_p, params->tnHconPv, params->tnHconSt, 0);
        ts_pt = odt_contrast_high(ts_pt, hcon_p, params->tnHconPv, params->tnHconSt, 0);
        if (params->tonescaleMap == 1) crv_val = odt_contrast_high(crv_val, hcon_p, params->tnHconPv, params->tnHconSt, 0);
    }

    /***************************************************
      Apply Advanced Contrast
      (MetalKernel.mm:878-894 — per-channel RGB/CMY gamma_contrast on the ratios)
    --------------------------------------------------*/
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

    /***************************************************
      Apply Tonescale
    --------------------------------------------------*/
    tsn = odt_compress_hyperbolic_power(tsn, params->ts_s, params->tnCon);
    ts_pt = odt_compress_hyperbolic_power(ts_pt, params->ts_s1, params->tnCon);

    if (params->tonescaleMap == 1) crv_val = odt_compress_hyperbolic_power(crv_val, params->ts_s, params->tnCon);

    /***************************************************
      Prerequisite color spaces
    --------------------------------------------------*/
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

    /***************************************************
      Brilliance
    --------------------------------------------------*/
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

    /***************************************************
      Mid-Range Purity
    --------------------------------------------------*/
    float ptm_sc = 1.0f;
    if (params->ptmPresetEnable || params->ptmUIEnable) {
        float ptm_ach_d = odt_complement_power(ach_d, params->ptmLowSt);
        ptm_sc = odt_sigmoid_cubic(ptm_ach_d, params->ptmLow*(1.0f - ts_pt));

        ptm_ach_d = odt_complement_power(ach_d, params->ptmHighSt)*(1.0f - ts_pt) + ach_d*ach_d*ts_pt;
        ptm_sc *= odt_sigmoid_cubic(ptm_ach_d, params->ptmHigh*ts_pt);
        ptm_sc = _fmaxf(0.0f, ptm_sc);
    }

    /***************************************************
      Hue Angle Premultiplication
    --------------------------------------------------*/
    ha_rgb = ha_rgb * ach_d;
    ha_cmy = ha_cmy * ((1.5f)*odt_compress_toe_quadratic(ach_d, 0.5f, 0));

    /***************************************************
      Hue Contrast R
    --------------------------------------------------*/
    if (params->hcPresetEnable || params->hcUIEnable) {
        float hc_ts = 1.0f - ts_pt;
        float hc_c = (1.0f - ach_d)*hc_ts + ach_d*(1.0f - hc_ts);
        hc_c *= ha_rgb.x;
        hc_ts *= hc_ts;
        float hc_f = params->hcR*(hc_c - 2.0f*hc_c*hc_ts) + 1.0f;
        rgb = make_float3(rgb.x, rgb.y*hc_f, rgb.z*hc_f);
    }

    /***************************************************
      Hue Shift
    --------------------------------------------------*/
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

    /***************************************************
      Module Application
    --------------------------------------------------*/
    rgb = rgb * brl_f;

    ts_pt_cmp *= ptm_sc;
    rgb = make_float3(rgb.x*ts_pt_cmp + 1.0f - ts_pt_cmp,
                      rgb.y*ts_pt_cmp + 1.0f - ts_pt_cmp,
                      rgb.z*ts_pt_cmp + 1.0f - ts_pt_cmp);

    // Inverse Rendering Space
    sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;
    rgb = make_float3((sat_L*params->rsSa - rgb.x)/(params->rsSa - 1.0f),
                      (sat_L*params->rsSa - rgb.y)/(params->rsSa - 1.0f),
                      (sat_L*params->rsSa - rgb.z)/(params->rsSa - 1.0f));

    /***************************************************
      Creative White Point and Output Transform
    --------------------------------------------------*/
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

    // Overlay curve gamut / cwp
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

    /***************************************************
      Purity Compress Low
    --------------------------------------------------*/
    if (params->ptlPresetEnable || params->ptlUIEnable) {
        float sum0 = odt_softplus(rgb.x, 0.2f, -100.0f, -0.3f) + rgb.y + odt_softplus(rgb.z, 0.2f, -100.0f, -0.3f);
        rgb.x = odt_softplus(rgb.x, 0.04f, -0.3f, 0.0f);
        rgb.y = odt_softplus(rgb.y, 0.06f, -0.3f, 0.0f);
        rgb.z = odt_softplus(rgb.z, 0.01f, -0.05f, 0.0f);

        float ptl_norm = _fminf(1.0f, odt_sdivf(sum0, rgb.x + rgb.y + rgb.z));
        rgb = rgb * ptl_norm;
    }

    /***************************************************
      Final Tonescale and Display Transform
    --------------------------------------------------*/
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

    // Overlay curve EOTF + render
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

// ── Host-side per-frame precompute ─────────────────────────────────────────────
// Fills the ts_* fields the kernel reads, replicating
// OpenDRTPresets::calculateTonescaleConstants EXACTLY (OpenDRTPresets.h
// lines 587-623). The display-scale and final-factor formulas need ptHdr/tnGb
// too. Host-only: calls LDT_FUNC math that is __device__ under CUDA.
//
// NOTE: this precompute now performs the display-encoding remap FIRST (replicating
// MetalKernel.mm:1310-1352): it maps the UI (displayGamut, eotf) pair to the
// canonical (final_display_gamut, final_eotf) pair, writes them back into
// params->displayGamut / params->eotf, and only then computes the tonescale
// constants (ts_dsc keys off the final eotf). The kernel reads the post-remap
// values. The remap is idempotent, so passing an already-canonical pair is safe.
#if !defined(__CUDACC__) && !defined(__METAL_VERSION__)
LDT_HOST inline void opendrt_precompute(OpenDRTParams* p)
{
    // ── Display-encoding remap (MetalKernel.mm:1310-1352) ──────────────────────
    // The host maps the UI (displayGamut, eotf) pair through a display_encoding
    // preset to a CANONICAL (final_display_gamut, final_eotf) pair BEFORE the
    // tonescale constants are computed (ts_dsc keys off the final eotf) and
    // BEFORE the kernel reads params->displayGamut / params->eotf. CudaKernel.cu
    // skips this remap (raw enum passthrough) — that is the stale behaviour we
    // are correcting. Idempotent: feeding back a canonical pair maps to itself.
    {
        int dg = p->displayGamut;
        int e  = p->eotf;
        int preset;
        if      (dg == 0 && e == 2) preset = 0; // Rec.1886   (Rec.709 + 2.4)
        else if (dg == 0 && e == 1) preset = 1; // sRGB       (Rec.709 + 2.2)
        else if (dg == 1 && e == 1) preset = 2; // Display P3 (P3-D65 + 2.2)
        else if (dg == 2 && e == 4) preset = 3; // Rec.2100 PQ
        else if (dg == 2 && e == 5) preset = 4; // Rec.2100 HLG
        else if (dg == 1 && e == 4) preset = 5; // Dolby PQ   (P3-D65 + PQ)
        else                        preset = 0; // fallback -> Rec.1886

        int final_dg = 0, final_e = 2;
        if      (preset == 0) { final_dg = 0; final_e = 2; }
        else if (preset == 1) { final_dg = 0; final_e = 1; }
        else if (preset == 2) { final_dg = 1; final_e = 1; }
        else if (preset == 3) { final_dg = 2; final_e = 4; }
        else if (preset == 4) { final_dg = 2; final_e = 5; }
        else if (preset == 5) { final_dg = 1; final_e = 4; }

        p->displayGamut = final_dg;
        p->eotf         = final_e;

        // Mirror the host's oetfType/eotfType bookkeeping (carried, not read by
        // the device path which switches on inOetf/eotf directly, but kept in
        // sync for byte-layout / future-use fidelity with MetalKernel.mm).
        p->oetfType = p->inOetf; // OETF_PRESETS[i].oetf_type == i for i in [0,9]
        p->eotfType = final_e;   // EOTF_PRESETS[i].eotf_type == i for i in [0,5]
    }

    const float tn_Lp  = p->tnLp;
    const float tn_gb  = p->tnGb;
    const float pt_hdr = p->ptHdr;
    const float tn_Lg  = p->tnLg;
    const float tn_con = p->tnCon;
    const float tn_sh  = p->tnSh;
    const float tn_toe = p->tnToe;
    const float tn_off = p->tnOff;
    const int   eotf   = p->eotf;

    p->ts_x1 = _powf(2.0f, 6.0f * tn_sh + 4.0f);
    p->ts_y1 = tn_Lp / 100.0f;
    p->ts_x0 = 0.18f + tn_off;
    p->ts_y0 = (tn_Lg / 100.0f) * (1.0f + tn_gb * _log2f(p->ts_y1));

    p->ts_s0  = odt_compress_toe_quadratic(p->ts_y0, tn_toe, 1);
    p->ts_s10 = p->ts_x0 * (_powf(p->ts_s0, -1.0f / tn_con) - 1.0f);

    p->ts_m1 = p->ts_y1 / _powf(p->ts_x1 / (p->ts_x1 + p->ts_s10), tn_con);
    p->ts_m2 = odt_compress_toe_quadratic(p->ts_m1, tn_toe, 1);

    p->ts_s = p->ts_x0 * (_powf(p->ts_s0 / p->ts_m2, -1.0f / tn_con) - 1.0f);

    p->ts_dsc = (eotf == 4) ? 0.01f : (eotf == 5) ? 0.1f : 100.0f / tn_Lp;

    p->pt_cmp_Lf = pt_hdr * _fminf(1.0f, (tn_Lp - 100.0f) / 900.0f);

    p->s_Lp100 = p->ts_x0 * (_powf(tn_Lg / 100.0f, -1.0f / tn_con) - 1.0f);

    p->ts_s1 = p->ts_s * p->pt_cmp_Lf + p->s_Lp100 * (1.0f - p->pt_cmp_Lf);
}
#endif
