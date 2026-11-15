#pragma once
// TesseractAlgorithm.h
// Pure-math core of the Tesseract view transform, ported from
// upstream Tesseract v0.0.3 (GPL-3.0),
// open-display-transform commit 29df1e643e4c078cbdaac8e5402141c7750b6a58
// (display-transforms/tesseract/Tesseract.dctl).
//
// Status: GPU twins exist (metal/TesseractMetal.mm, opencl/TesseractOpenCL*,
// cuda/TesseractCuda.cu) and are parity gated against this core by
// harness/alt_drt_models/build_alt_gpu_parity.sh, but the model is still
// UNWIRED in the plugin: no OFX params, no render dispatch. Callers today are
// the harness programs under harness/alt_drt_models/ only.
//
// Deviations from upstream (each is marked DEVIATION at its site):
//   tess_eotf_hlg inverse: sqrt(3*x) takes fmax(0, x), so negative inputs
//     (reachable with clamp off) give 0 instead of NaN. x >= 0 is bit exact,
//     so every clamped or in-range render is unchanged.
//
// Port rules:
//   - Operation order of the DCTL transform() is kept line for line.
//   - Matrices are row-major 9-float arrays applied with tess_vdot9, which has
//     the same sum order as the upstream vdot (m.x.x*v.x + m.x.y*v.y + ...).
//   - Per-frame constants (matrices, tonescale scalars, display scale) are
//     resolved on the host by tesseract_precompute; the kernel only reads them.
//   - Every pow on a value that can be zero or negative goes through
//     odt_spowf (returns the input unchanged when it is <= 0), as upstream.
//   - Upstream _exp10f is TESS_EXP10F: exp10f on CUDA, exp10 on Metal, and
//     _powf(10, x) on the host (the 6A OETFs use the same host form).
//
// Reused verbatim from OpenDRTAlgorithm.h (byte-identical to upstream):
//   odt_spowf, odt_spowf3, odt_sdivf3f, odt_clampf3, odt_fmaxf3, odt_fminf3,
//   odt_eotf_pq, odt_oetf_davinci_intermediate, odt_oetf_filmlight_tlog,
//   odt_oetf_arri_logc4, plus the 6A-only odt_oetf_acescct,
//   odt_oetf_red_log3g10, odt_oetf_fujifilm_flog2 and odt_getInputMatrix
//   (host precompute only).
// Rewritten here (tess_ prefix):
//   tess_vdot9 (row-major float[9]), the exp10 forms of LogC3 / V-Log /
//   S-Log3 / F-Log v1, tess_eotf_hlg (the 6A odt_eotf_hlg uses an unguarded
//   pow and a fixed gamma; upstream uses spowf and the surround model),
//   tess_compress_quadratic_lift, tess_tonescale, tess_display_encode (the
//   6A odt_apply_eotf clamps with fmax(0); upstream uses spowf).

#include "../../../shared/LookDevPlatform.h"
#include "OpenDRTAlgorithm.h"
#include "TesseractParams.h"

#ifndef TESS_EXP10F
  #if defined(__CUDACC__)
    #define TESS_EXP10F(x) exp10f((float)(x))
  #elif defined(__METAL_VERSION__)
    #define TESS_EXP10F(x) exp10((float)(x))
  #else
    #define TESS_EXP10F(x) _powf(10.0f, (float)(x))
  #endif
#endif

// ---- Matrix multiply, row-major 3x3 ------------------------------------------
LDT_FUNC float3 tess_vdot9(const float* m, float3 v) {
    return make_float3(m[0]*v.x + m[1]*v.y + m[2]*v.z,
                       m[3]*v.x + m[4]*v.y + m[5]*v.z,
                       m[6]*v.x + m[7]*v.y + m[8]*v.z);
}

// ---- Input OETFs that upstream writes with _exp10f ----------------------------
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

// 6A input transfer index (see TesseractParams.h), 10 = F-Log v1.
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

// ---- HLG (upstream eotf_hlg, surround model with Lw=1000, Ls=5) ----------------
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
        // DEVIATION: upstream sqrt(3*x) is NaN for x < 0 (reachable with
        // clamp off). fmax(0, x) makes it 0; x >= 0 is unchanged.
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

// Upstream "Apply inverse Display EOTF" block (display encode).
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

// ---- Curves -------------------------------------------------------------------
// Forward branch only; the kernel never calls the inverse (upstream inv == 0).
LDT_FUNC float tess_compress_quadratic_lift(float x, float t0, float x0) {
    const float s0 = _sqrtf(_fmaxf(1e-6f, x0))/-t0;
    const float o0 = 1.0f/(4.0f*s0*s0) + t0;
    const float o1 = t0 - _sqrtf(o0 - t0)/s0;
    return x > t0 ? x : o1 + _sqrtf(o0 - x)/s0;
}

// Upstream tonescale(x, Lp, Lg, off, p, 0) with its constants hoisted into the
// params block by tesseract_precompute.
LDT_FUNC float tess_tonescale(float x, const TesseractParams* p) {
    x = x < p->ts_x0 ? p->ts_s0*odt_spowf(x, p->con) : p->ts_m0*(x - p->ts_x0) + p->ts_o0;
    return p->ts_m*x/(x + p->ts_s);
}

// ---- Per-pixel entry point -----------------------------------------------------
LDT_FUNC float3 tesseract_processPixel(float3 in, int x, int y, int w, int h,
                                       const TesseractParams* p) {
    (void)x; (void)y; (void)w; (void)h;
    float3 rgb = in;

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

// ================================================================================
// Host-only: defaults, presets, precompute
// ================================================================================
#if !defined(__CUDACC__) && !defined(__METAL_VERSION__)

// Presets: upstream Tesseract v0.0.3 ships no preset table (the DCTL has only
// the DEFINE_UI_PARAMS defaults), so there is none here.

// Upstream XYZ -> display matrices, row-major (Tesseract.dctl matrix_xyz_to_*).
LDT_HOST inline void tess_display_matrix(int display_gamut, float out[9]) {
    static const float rec709[9] = {
        3.2409699419f, -1.53738317757f, -0.498610760293f,
        -0.969243636281f, 1.87596750151f, 0.041555057407f,
        0.055630079697f, -0.203976958889f, 1.05697151424f };
    static const float p3d65[9] = {
        2.49349691194f, -0.931383617919f, -0.402710784451f,
        -0.829488969562f, 1.76266406032f, 0.023624685842f,
        0.035845830244f, -0.076172389268f, 0.956884524008f };
    static const float rec2020[9] = {
        1.71665118797f, -0.355670783776f, -0.253366281374f,
        -0.666684351832f, 1.61648123664f, 0.015768545814f,
        0.017639857445f, -0.042770613258f, 0.942103121235f };
    const float* src = display_gamut == 1 ? p3d65 : display_gamut == 2 ? rec2020 : rec709;
    for (int i = 0; i < 9; ++i) out[i] = src[i];
}

// 6A input gamut -> XYZ, row-major. odt_getInputMatrix stores columns; the
// numbers are the same as the upstream row-major macros, transposed.
LDT_HOST inline void tess_input_matrix(int in_gamut, float out[9]) {
    float3 c[3];
    odt_getInputMatrix(in_gamut, c);
    for (int col = 0; col < 3; ++col) {
        out[0*3 + col] = c[col].x;
        out[1*3 + col] = c[col].y;
        out[2*3 + col] = c[col].z;
    }
}

LDT_HOST inline void tesseract_defaults(TesseractParams* p) {
    p->in_gamut = 15;
    p->in_oetf = 0;
    p->display_gamut = 0;
    p->eotf = 2;
    p->clamp = 1;
    p->Lp = 100.0f;
    p->Lg = 10.0f;
    p->off = 0.005f;
    p->con = 1.6f;
    p->pb = 0.6f;
    p->hl_r = 0.3f;
    p->hl_g = 0.2f;
    p->hl_b = 0.2f;
    p->hh_r = 0.5f;
    p->hh_g = 0.0f;
    p->hh_b = -0.5f;
    for (int i = 0; i < 9; ++i) { p->in_to_xyz[i] = 0.0f; p->xyz_to_display[i] = 0.0f; }
    p->ds = 1.0f;
    p->ts_x0 = p->ts_s0 = p->ts_o0 = p->ts_m0 = p->ts_m = p->ts_s = 0.0f;
    p->eotf_p = 2.4f;
    p->_pad0 = 0;
}

LDT_HOST inline void tesseract_precompute(TesseractParams* p) {
    tess_input_matrix(p->in_gamut, p->in_to_xyz);
    tess_display_matrix(p->display_gamut, p->xyz_to_display);

    const int eotf = p->eotf;
    p->ds = eotf == 4 ? p->Lp/10000.0f : eotf == 5 ? p->Lp/1000.0f : 1.0f;
    p->eotf_p = 2.0f + eotf*0.2f;

    // tonescale() constants, same expressions as upstream
    const float Lp = p->Lp;
    const float Lg = p->Lg;
    const float off = p->off;
    const float con = p->con;
    const float Lgb = 0.12f;
    const float x0 = 0.18f + off;
    const float y0 = Lg/Lp*(1.0f + Lgb*_logf(Lp/100.0f)/_logf(2.0f));
    const float s0 = y0*odt_spowf(x0, -con);
    const float o0 = s0*odt_spowf(x0, con);
    const float m0 = x0 == 0.0f ? 0.0f : con*o0/x0;
    // upstream: 256.0*_logf(Lp)/_logf(100.0) - 128.0f (double literals)
    const float w0 = (float)(256.0*(double)_logf(Lp)/(double)_logf(100.0f) - 128.0);
    const float w1 = 1.0f;
    const float m = (w1*(m0*w0 - y0))/(m0*w0 - w1);
    const float s = m - y0;
    p->ts_x0 = x0;
    p->ts_s0 = s0;
    p->ts_o0 = o0;
    p->ts_m0 = m0;
    p->ts_m = m;
    p->ts_s = s;
}

#endif // host only
