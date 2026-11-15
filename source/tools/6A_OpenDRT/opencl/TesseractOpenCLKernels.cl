// TesseractOpenCLKernels.cl
// OpenCL C 1.2 kernel for the Tesseract view transform (6A).
//
// What it is: a hand port of the MSL kernel string in
// tools/6A_OpenDRT/metal/TesseractMetal.mm (kernel TesseractKernel), which is itself the
// twin of core/TesseractAlgorithm.h. Metal is ground truth: every helper below is a
// line-for-line twin of its MSL original, same math, same order of operations,
// no approximations. When the MSL changes, change this file in the same commit.
//
// Kernel contract (fixed; the host binds exactly these four args):
//   arg 0  __constant TesseractParams* p_Params
//   arg 1  __global const float4*    p_Input
//   arg 2  __global float4*          p_Output
//   arg 3  __constant int4*          p_Dims   = {srcPitch, dstPitch, w, h}
//          (pitches in float4 units, i.e. rowBytes / 16)
// Global size covers (w, h); out-of-range work items return. Alpha passes through.
// This is the same p_Dims layout the Metal kernel uses.
//
// Params layout: TesseractParams below mirrors the struct in
// tools/6A_OpenDRT/core/TesseractParams.h field for field, ints and floats only
// (172 bytes, no vector members, so OpenCL and C++ layouts agree). The host
// static_asserts sizeof(TesseractParams) == 172, and the typedef below fails to compile
// on the device side if the twin drifts. The caller has already run the host
// precompute; the params stay in __constant and are read through the pointer.
//
// Port notes: constant -> __constant; make_float3 -> vector literal; TESS_EXP10F
// -> exp10 (OpenCL C 1.2 builtin; the MSL twin uses exp10 too).

#define LDT_FUNC static inline

// Math-builtin aliases so the bodies below are textually identical to the
// C++/CUDA source and to the MSL twin.
#define _powf(x,y)       pow((float)(x),(float)(y))
#define _exp2f(x)        exp2((float)(x))
#define _expf(x)         exp((float)(x))
#define _logf(x)         log((float)(x))
#define _log2f(x)        log2((float)(x))
#define _fmaxf(x,y)      fmax((float)(x),(float)(y))
#define _fminf(x,y)      fmin((float)(x),(float)(y))
#define _sqrtf(x)        sqrt((float)(x))
#define make_float3(a,b,c) ((float3)((a),(b),(c)))
// exp10 is an OpenCL C 1.2 builtin (same form as the MSL twin's exp10).
#define TESS_EXP10F(x)   exp10((float)(x))

// Byte-identical twin of TesseractParams (core/TesseractParams.h), 172 bytes.
typedef struct TesseractParams {
    int   in_gamut;
    int   in_oetf;
    int   display_gamut;
    int   eotf;
    int   clamp;
    float Lp;
    float Lg;
    float off;
    float con;
    float pb;
    float hl_r;
    float hl_g;
    float hl_b;
    float hh_r;
    float hh_g;
    float hh_b;
    float in_to_xyz[9];
    float xyz_to_display[9];
    float ds;
    float ts_x0;
    float ts_s0;
    float ts_o0;
    float ts_m0;
    float ts_m;
    float ts_s;
    float eotf_p;
    int   _pad0;
} TesseractParams;

typedef char TesseractParams_size_check[(sizeof(TesseractParams) == 172) ? 1 : -1];

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

// ---- tess_ helpers (core/TesseractAlgorithm.h) -----------------------------------
LDT_FUNC float3 tess_vdot9(__constant float* m, float3 v) {
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
LDT_FUNC float tess_compress_quadratic_lift(float x, float t0, float x0) {
    const float s0 = _sqrtf(_fmaxf(1e-6f, x0))/-t0;
    const float o0 = 1.0f/(4.0f*s0*s0) + t0;
    const float o1 = t0 - _sqrtf(o0 - t0)/s0;
    return x > t0 ? x : o1 + _sqrtf(o0 - x)/s0;
}
LDT_FUNC float tess_tonescale(float x, __constant TesseractParams* p) {
    x = x < p->ts_x0 ? p->ts_s0*odt_spowf(x, p->con) : p->ts_m0*(x - p->ts_x0) + p->ts_o0;
    return p->ts_m*x/(x + p->ts_s);
}

// ---- Per-pixel entry point (tesseract_processPixel) --------------------------------
LDT_FUNC float3 tesseract_processPixel(float3 rgb_in, int x, int y, int w, int h,
                                       __constant TesseractParams* p) {
    float3 rgb = rgb_in;

    // Linearize if a non-linear input transfer function is selected
    rgb = tess_linearize(rgb, p->in_oetf);

    // Convert into display gamut
    rgb = tess_vdot9(p->in_to_xyz, rgb);
    rgb = tess_vdot9(p->xyz_to_display, rgb);

    // Offset
    rgb.x += p->off;
    rgb.y += p->off;
    rgb.z += p->off;

    float mx = odt_fmaxf3(rgb);
    float mn = odt_fminf3(rgb);

    // RGB Ratios
    rgb = odt_sdivf3f(rgb, mx);

    // Limit ratios in the shadows
    rgb = odt_clampf3(rgb, -2.0f, 2.0f);

    // Purity compression
    float3 pc_rats = rgb;
    pc_rats.x = tess_compress_quadratic_lift(pc_rats.x, 0.2f, 0.3f);
    pc_rats.y = tess_compress_quadratic_lift(pc_rats.y, 0.2f, 0.3f);
    pc_rats.z = tess_compress_quadratic_lift(pc_rats.z, 0.2f, 0.03f);

    float pc_rats_mn = odt_fminf3(pc_rats);

    // remove achromatic
    pc_rats.x = pc_rats.x - pc_rats_mn;
    pc_rats.y = pc_rats.y - pc_rats_mn;
    pc_rats.z = pc_rats.z - pc_rats_mn;
    // restore full range for tonescale curve
    pc_rats_mn *= mx;

    // purity compression strength, mix factor back to maxrgb
    const float pc_s_x = 0.03f;
    const float pc_s_y = 0.16f;
    const float pc_s_z = 0.06f;
    float pc_f = pc_s_x*pc_rats.x + pc_s_y*pc_rats.y + pc_s_z*pc_rats.z;

    // minrgb mix to maxrgb by pc_f
    mn = _fmaxf(0.0f, pc_rats_mn*(1.0f - pc_f) + mx*pc_f);

    // Tonescale the modified minrgb and the maxrgb
    mn = tess_tonescale(mn, p);
    mx = tess_tonescale(mx, p);

    // Purity boost
    float pb_m0 = 1.0f + p->pb;
    float pb_m1 = 2.0f - pb_m0;
    float pb_f = mx*(pb_m1 - pb_m0) + pb_m0;
    float rats_mn = _fmaxf(0.0f, odt_fminf3(rgb));
    rgb = make_float3((rgb.x*pb_f + 1.0f - pb_f)*rats_mn + rgb.x*(1.0f - rats_mn),
                      (rgb.y*pb_f + 1.0f - pb_f)*rats_mn + rgb.y*(1.0f - rats_mn),
                      (rgb.z*pb_f + 1.0f - pb_f)*rats_mn + rgb.z*(1.0f - rats_mn));

    // Purity compress using mn by lerping to 1.0 in rgb ratios
    rgb = make_float3(rgb.x*(1.0f - mn) + mn,
                      rgb.y*(1.0f - mn) + mn,
                      rgb.z*(1.0f - mn) + mn);

    float mn_pc = odt_fminf3(rgb);
    float3 ha_rgb = make_float3(rgb.x - mn_pc, rgb.y - mn_pc, rgb.z - mn_pc);
    // Narrow hue angles
    ha_rgb = make_float3(_fminf(1.0f, _fmaxf(0.0f, ha_rgb.x - (ha_rgb.y + ha_rgb.z))),
                         _fminf(1.0f, _fmaxf(0.0f, ha_rgb.y - (ha_rgb.x + ha_rgb.z))),
                         _fminf(1.0f, _fmaxf(0.0f, ha_rgb.z - (ha_rgb.x + ha_rgb.y))));

    // Hue contrast (opposing)
    float3 ha_rgb_hl = odt_spowf3(ha_rgb, 2.0f);
    float3 hc_m0 = make_float3(1.0f - p->hl_r, 1.0f - p->hl_g, 1.0f - p->hl_b);
    float3 hc_m1 = make_float3(1.0f + p->hl_r, 1.0f + p->hl_g, 1.0f + p->hl_b);

    float3 f = make_float3(mn*(hc_m1.x - hc_m0.x) + hc_m0.x,
                           mn*(hc_m1.y - hc_m0.y) + hc_m0.y,
                           mn*(hc_m1.z - hc_m0.z) + hc_m0.z);

    rgb = make_float3(
        rgb.x*f.y*ha_rgb_hl.y + rgb.x*f.z*ha_rgb_hl.z + rgb.x*(1.0f - (ha_rgb_hl.y + ha_rgb_hl.z)),
        rgb.y*f.x*ha_rgb_hl.x + rgb.y*f.z*ha_rgb_hl.z + rgb.y*(1.0f - (ha_rgb_hl.x + ha_rgb_hl.z)),
        rgb.z*f.x*ha_rgb_hl.x + rgb.z*f.y*ha_rgb_hl.y + rgb.z*(1.0f - (ha_rgb_hl.x + ha_rgb_hl.y)));

    // Hue shift on RGB ratios, mixed by mn
    float3 hs = make_float3(rgb.x + ha_rgb.z*p->hh_b - ha_rgb.y*p->hh_g,
                            rgb.y + ha_rgb.x*p->hh_r - ha_rgb.z*p->hh_b,
                            rgb.z + ha_rgb.y*p->hh_g - ha_rgb.x*p->hh_r);
    rgb = make_float3(rgb.x*(1.0f - mn) + hs.x*mn,
                      rgb.y*(1.0f - mn) + hs.y*mn,
                      rgb.z*(1.0f - mn) + hs.z*mn);

    // Final low-end gamut compress
    rgb.x = tess_compress_quadratic_lift(rgb.x, 0.06f, 0.02f);
    rgb.y = tess_compress_quadratic_lift(rgb.y, 0.06f, 0.06f);
    rgb.z = tess_compress_quadratic_lift(rgb.z, 0.06f, 0.02f);

    // Depart from RGB ratios
    rgb = mx*rgb;

    // Display scale
    rgb = rgb*p->ds;

    // Clamp
    if (p->clamp == 1)
        rgb = odt_clampf3(rgb, 0.0f, p->ds);

    // Display encode
    rgb = tess_display_encode(rgb, p->eotf, p->eotf_p);

    return rgb;
}

// ---- Kernel ------------------------------------------------------------------------
__kernel void TesseractKernel(__constant TesseractParams* p_Params,
                              __global const float4* p_Input,
                              __global float4* p_Output,
                              __constant int4* p_Dims)
{
    int srcPitch = p_Dims->x;   // float4 stride (rowBytes/16)
    int dstPitch = p_Dims->y;
    int pw       = p_Dims->z;
    int ph       = p_Dims->w;
    int x = (int)get_global_id(0);
    int y = (int)get_global_id(1);
    if (x >= pw || y >= ph) return;

    float4 src    = p_Input[y * srcPitch + x];
    float3 result = tesseract_processPixel(src.xyz, x, y, pw, ph, p_Params);
    p_Output[y * dstPitch + x] = (float4)(result, src.w);
}
