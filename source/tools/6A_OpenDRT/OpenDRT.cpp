// OpenDRT.cpp - "6A. OpenDRT" OFX plugin (factory + dispatch).
//
// Full OpenDRT display-rendering transform: input linearize + gamut -> tonescale ->
// per-hue purity/brilliance/hueshift/contrast -> creative white point -> display
// encode. The per-pixel math + host precompute live in core/OpenDRTAlgorithm.h
// (bit-exact vs the Metal golden - see that header's provenance note). This file
// only builds the param UI, populates OpenDRTParams on the host, runs
// opendrt_precompute(), and dispatches to CPU/CUDA/Metal.
//
// The params are ported VERBATIM (names, defaults, ranges, groups) from the
// canonical OpenDRT.cpp so a project authored against canonical OpenDRT round-
// trips. The Look/Tonescale preset-value override engine IS reproduced
// (applyPresetValues, gated by per-module Persistent-Values locks) and all 16
// input gamuts are matrixed (odt_getInputMatrix). The canonical's dynamic-matrix
// layer (ColorMatrices.json / MatrixManager) is the only piece intentionally NOT
// reproduced - the input-to-XYZ matrices are baked in directly instead.

#ifndef NOMINMAX
  #define NOMINMAX
#endif

#include "OpenDRT.h"
#include "LdtFileDialogs.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <memory>

// -- Dual-build plugin identity ------------------------------------------------
// Standalone (free) and suite builds use different identifiers so they co-install
// in Resolve without dedup. OPENDRT_PLUGIN_ID is defined in OpenDRT.h (it also
// backs the factory ctor); the label/group differ per build here.
#ifdef LDT_OPENDRT_STANDALONE
  #define OPENDRT_PLUGIN_LABEL "OpenDRT"
  #define OPENDRT_PLUGIN_GROUP "Dec. 18 Studios"
#else
  #define OPENDRT_PLUGIN_LABEL "6A. OpenDRT"
  #define OPENDRT_PLUGIN_GROUP "Look Dev Tools - Dec. 18 Studios"
#endif

#define kPluginDescription \
    "OpenDRT display rendering transform - scene-linear to display. Tonescale, " \
    "creative white point, per-hue purity / brilliance / hueshift / contrast, and " \
    "display encoding (Rec.709/P3/Rec.2020, gamma/PQ/HLG). Ported from OpenDRT; " \
    "part of the Look Dev Tools suite."

#define kSupportsTiles            false
#define kSupportsMultiResolution  false
#define kSupportsMultipleClipPARs false

// -- Look + Tonescale preset value tables --------------------------------------
// Transcribed VERBATIM from OpenDRTPresets.h (LOOK_PRESETS / TONESCALE_PRESETS).
// One source of truth: buildParams() reads the module enable flags + cwp from
// here to drive the kernel's (PresetEnable || UIEnable) gates, and
// applyPresetValues() writes the slider VALUES into the UI when a preset is
// chosen - exactly as canonical OpenDRT does.
namespace {
struct LookPreset {
    float tn_Lg, tn_con, tn_sh, tn_toe, tn_off;
    int   hcon_enable, lcon_enable;
    float tn_hcon, tn_hcon_pv, tn_hcon_st;
    float tn_lcon, tn_lcon_w, tn_lcon_pc;
    int   cwp;
    float cwp_rng;
    float rs_sa, rs_rw, rs_bw;
    float pt_r, pt_g, pt_b, pt_rng_low, pt_rng_high;
    int   ptl_enable, ptm_enable;
    float ptm_low, ptm_low_st, ptm_high, ptm_high_st;
    int   brl_enable;
    float brl_r, brl_g, brl_b, brl_c, brl_m, brl_y, brl_rng;
    int   hs_rgb_enable;
    float hs_r, hs_g, hs_b, hs_rgb_rng;
    int   hs_cmy_enable;
    float hs_c, hs_m, hs_y;
    int   hc_enable;
    float hc_r;
};
// idx: 0 Default, 1 Colorful, 2 Umbra, 3 Base
static const LookPreset kLookPresets[4] = {
    // Default
    { 11.1f, 1.4f, 0.5f, 0.003f, 0.005f, 0, 1, 0.0f, 1.0f, 4.0f, 1.0f, 0.5f, 1.0f,
      0, 0.5f, 0.35f, 0.25f, 0.55f, 0.5f, 2.0f, 2.0f, 0.2f, 0.8f, 1, 1,
      0.2f, 0.5f, -0.8f, 0.3f, 1, -0.5f, -0.4f, -0.2f, 0.0f, 0.0f, 0.0f, 0.66f,
      1, 0.35f, 0.25f, 0.5f, 0.6f, 1, 0.2f, 0.2f, 0.2f, 1, 0.6f },
    // Colorful
    { 11.1f, 1.3f, 0.5f, 0.005f, 0.005f, 0, 1, 0.0f, 1.0f, 4.0f, 0.75f, 1.0f, 1.0f,
      0, 0.5f, 0.35f, 0.15f, 0.55f, 0.5f, 0.8f, 0.5f, 0.25f, 0.5f, 1, 1,
      0.5f, 0.5f, -0.8f, 0.3f, 1, -0.55f, -0.5f, 0.0f, 0.0f, 0.0f, 0.1f, 0.5f,
      1, 0.4f, 0.6f, 0.5f, 0.6f, 1, 0.2f, 0.1f, 0.2f, 1, 0.8f },
    // Umbra
    { 6.0f, 1.8f, 0.5f, 0.001f, 0.015f, 0, 1, 0.0f, 1.0f, 4.0f, 1.0f, 1.0f, 1.0f,
      3, 0.8f, 0.45f, 0.1f, 0.35f, 0.1f, 0.4f, 2.5f, 0.2f, 0.8f, 1, 1,
      0.4f, 0.5f, -0.8f, 0.3f, 1, -0.7f, -0.6f, -0.2f, 0.0f, -0.25f, 0.1f, 0.9f,
      1, 0.4f, 0.8f, 0.4f, 1.0f, 1, 1.0f, 0.6f, 1.0f, 1, 0.8f },
    // Base
    { 11.1f, 1.4f, 0.5f, 0.003f, 0.0f, 0, 0, 0.0f, 1.0f, 4.0f, 0.0f, 0.5f, 1.0f,
      0, 0.5f, 0.35f, 0.25f, 0.5f, 1.0f, 2.0f, 2.5f, 0.25f, 0.25f, 1, 0,
      0.0f, 0.5f, 0.0f, 0.3f, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f,
      0, 0.0f, 0.0f, 0.0f, 0.5f, 0, 0.0f, 0.0f, 0.0f, 0, 0.0f },
};

struct TonescalePreset {
    float tn_Lg, tn_con, tn_sh, tn_toe, tn_off;
    float tn_hcon, tn_hcon_pv, tn_hcon_st;
    float tn_lcon, tn_lcon_w, tn_lcon_pc;
};
// idx 0-8 -> tonescale_preset dropdown options 1-9 (option 0 = "Use Look Preset").
static const TonescalePreset kTonescalePresets[9] = {
    { 11.1f,  1.4f,  0.5f,   0.003f, 0.005f, 0.0f,  1.0f, 4.0f, 1.0f, 0.5f, 1.0f }, // High-Contrast
    { 11.1f,  1.4f,  0.5f,   0.003f, 0.005f, 0.0f,  1.0f, 4.0f, 0.0f, 0.5f, 1.0f }, // Low-Contrast
    { 10.0f,  1.0f,  0.245f, 0.02f,  0.0f,   0.55f, 0.0f, 2.0f, 1.13f,1.0f, 1.0f }, // ACES-1.x
    { 10.0f,  1.15f, 0.5f,   0.04f,  0.0f,   1.0f,  1.0f, 1.0f, 1.0f, 0.6f, 1.0f }, // ACES-2.0
    { 6.0f,   1.5f,  0.5f,   0.003f, 0.01f,  0.25f, 0.0f, 4.0f, 1.0f, 1.0f, 1.0f }, // Marvelous Tonescape
    { 11.1f,  1.05f, 0.5f,   0.1f,   0.015f, 0.0f,  0.0f, 2.0f, 2.0f, 0.2f, 1.0f }, // Arriba Tonecall
    { 10.42f, 1.2f,  0.5f,   0.02f,  0.0f,   0.0f,  1.0f, 1.0f, 0.0f, 0.6f, 1.0f }, // DaGrinchi Tonegroan
    { 11.1f,  1.15f, 0.5f,   0.04f,  0.006f, 0.0f,  0.0f, 0.5f, 0.5f, 2.0f, 0.5f }, // Aery Tonescale
    { 6.0f,   1.8f,  0.5f,   0.001f, 0.015f, 0.0f,  1.0f, 4.0f, 1.0f, 1.0f, 1.0f }, // Umbra Tonescale
};

// High / low contrast preset enable flags {hcon, lcon} per tonescale preset,
// same index as kTonescalePresets. Upstream OpenDRT v1.0.0 sets both flags
// from the tonescale preset when one is picked (option 1-9), not from the look.
static const int kTonescalePresetFlags[9][2] = {
    { 0, 1 }, // High-Contrast
    { 0, 0 }, // Low-Contrast
    { 1, 1 }, // ACES-1.x
    { 0, 0 }, // ACES-2.0
    { 1, 1 }, // Marvelous Tonescape
    { 0, 1 }, // Arriba Tonecall
    { 0, 0 }, // DaGrinchi Tonegroan
    { 0, 1 }, // Aery Tonescale
    { 0, 1 }, // Umbra Tonescale
};

// Model choice indices (kODT_Model).
enum OdtModel {
    ODT_MODEL_CURRENT = 0,   // the original model existing grades were made on
    ODT_MODEL_V101    = 1,   // same controls, upstream purity softclip fix
    ODT_MODEL_V11     = 2,   // upstream v1.1 model, own controls (odt11_*)
    ODT_MODEL_TESSERACT  = 3,  // Tesseract display transform, own controls (tess_*)
    ODT_MODEL_CHROMAGNON = 4,  // Chromagnon display transform, own controls (cmg_*)
    ODT_MODEL_COUNT   = 5
};
} // namespace

// -- Param-definition helpers --------------------------------------------------
static OFX::DoubleParamDescriptor* defineDouble(
    OFX::ImageEffectDescriptor& desc,
    const char* name, const char* label, const char* hint,
    double def, double min, double max, double inc,
    OFX::GroupParamDescriptor* parent, OFX::PageParamDescriptor* page)
{
    OFX::DoubleParamDescriptor* p = desc.defineDoubleParam(name);
    p->setLabels(label, label, label);
    p->setScriptName(name);
    p->setHint(hint);
    p->setDefault(def);
    p->setRange(min, max);
    p->setIncrement(inc);
    p->setDisplayRange(min, max);
    p->setDoubleType(OFX::eDoubleTypePlain);
    if (parent) p->setParent(*parent);
    if (page)   page->addChild(*p);
    return p;
}

static OFX::BooleanParamDescriptor* defineBool(
    OFX::ImageEffectDescriptor& desc,
    const char* name, const char* label, const char* hint, bool def,
    OFX::GroupParamDescriptor* parent, OFX::PageParamDescriptor* page)
{
    OFX::BooleanParamDescriptor* p = desc.defineBooleanParam(name);
    p->setLabels(label, label, label);
    p->setHint(hint);
    p->setDefault(def);
    if (parent) p->setParent(*parent);
    if (page)   page->addChild(*p);
    return p;
}

// -- OpenDRTProcessor (CPU path) -----------------------------------------------
void OpenDRTProcessor::multiThreadProcessImages(OfxRectI window) {
    const float* srcBase = static_cast<const float*>(
        _srcImg ? _srcImg->getPixelAddress(window.x1, window.y1) : nullptr);
    float* dstBase = static_cast<float*>(
        _dstImg->getPixelAddress(window.x1, window.y1));
    if (!dstBase) return;

    const int srcRowStride = (_srcImg ? _srcImg->getRowBytes()
                                       : _dstImg->getRowBytes()) / (int)sizeof(float);
    const int dstRowStride = _dstImg->getRowBytes() / (int)sizeof(float);
    const int w = window.x2 - window.x1;
    const int h = window.y2 - window.y1;

    for (int j = 0; j < h; ++j) {
        if (_effect.abort()) break;
        const float* srcRow = srcBase ? srcBase + (size_t)j * srcRowStride : nullptr;
        float*       dstRow = dstBase + (size_t)j * dstRowStride;
        for (int i = 0; i < w; ++i) {
            const int s = i * 4;
            float3 in_ = srcRow ? make_float3(srcRow[s + 0], srcRow[s + 1], srcRow[s + 2])
                                : make_float3(0.f, 0.f, 0.f);
            float a    = srcRow ? srcRow[s + 3] : 1.f;
            float3 out = opendrt_processPixel(in_, window.x1 + i, window.y1 + j,
                                              w, h, &_params);
            dstRow[s + 0] = out.x;
            dstRow[s + 1] = out.y;
            dstRow[s + 2] = out.z;
            dstRow[s + 3] = a;
        }
    }
}

// v1.1 CPU path: same buffer walk as the Current model, v1.1 pixel function.
void OpenDRTv11Processor::multiThreadProcessImages(OfxRectI window) {
    const float* srcBase = static_cast<const float*>(
        _srcImg ? _srcImg->getPixelAddress(window.x1, window.y1) : nullptr);
    float* dstBase = static_cast<float*>(
        _dstImg->getPixelAddress(window.x1, window.y1));
    if (!dstBase) return;

    const int srcRowStride = (_srcImg ? _srcImg->getRowBytes()
                                       : _dstImg->getRowBytes()) / (int)sizeof(float);
    const int dstRowStride = _dstImg->getRowBytes() / (int)sizeof(float);
    const int w = window.x2 - window.x1;
    const int h = window.y2 - window.y1;

    for (int j = 0; j < h; ++j) {
        if (_effect.abort()) break;
        const float* srcRow = srcBase ? srcBase + (size_t)j * srcRowStride : nullptr;
        float*       dstRow = dstBase + (size_t)j * dstRowStride;
        for (int i = 0; i < w; ++i) {
            const int s = i * 4;
            float3 in_ = srcRow ? make_float3(srcRow[s + 0], srcRow[s + 1], srcRow[s + 2])
                                : make_float3(0.f, 0.f, 0.f);
            float a    = srcRow ? srcRow[s + 3] : 1.f;
            float3 out = opendrt_v11_processPixel(in_, window.x1 + i, window.y1 + j,
                                                  w, h, &_params);
            dstRow[s + 0] = out.x;
            dstRow[s + 1] = out.y;
            dstRow[s + 2] = out.z;
            dstRow[s + 3] = a;
        }
    }
}

// Tesseract CPU path: same buffer walk as the Current model.
void TesseractProcessor::multiThreadProcessImages(OfxRectI window) {
    const float* srcBase = static_cast<const float*>(
        _srcImg ? _srcImg->getPixelAddress(window.x1, window.y1) : nullptr);
    float* dstBase = static_cast<float*>(
        _dstImg->getPixelAddress(window.x1, window.y1));
    if (!dstBase) return;

    const int srcRowStride = (_srcImg ? _srcImg->getRowBytes()
                                       : _dstImg->getRowBytes()) / (int)sizeof(float);
    const int dstRowStride = _dstImg->getRowBytes() / (int)sizeof(float);
    const int w = window.x2 - window.x1;
    const int h = window.y2 - window.y1;

    for (int j = 0; j < h; ++j) {
        if (_effect.abort()) break;
        const float* srcRow = srcBase ? srcBase + (size_t)j * srcRowStride : nullptr;
        float*       dstRow = dstBase + (size_t)j * dstRowStride;
        for (int i = 0; i < w; ++i) {
            const int s = i * 4;
            float3 in_ = srcRow ? make_float3(srcRow[s + 0], srcRow[s + 1], srcRow[s + 2])
                                : make_float3(0.f, 0.f, 0.f);
            float a    = srcRow ? srcRow[s + 3] : 1.f;
            float3 out = tesseract_processPixel(in_, window.x1 + i, window.y1 + j,
                                                w, h, &_params);
            dstRow[s + 0] = out.x;
            dstRow[s + 1] = out.y;
            dstRow[s + 2] = out.z;
            dstRow[s + 3] = a;
        }
    }
}

// Chromagnon CPU path: same buffer walk as the Current model.
void ChromagnonProcessor::multiThreadProcessImages(OfxRectI window) {
    const float* srcBase = static_cast<const float*>(
        _srcImg ? _srcImg->getPixelAddress(window.x1, window.y1) : nullptr);
    float* dstBase = static_cast<float*>(
        _dstImg->getPixelAddress(window.x1, window.y1));
    if (!dstBase) return;

    const int srcRowStride = (_srcImg ? _srcImg->getRowBytes()
                                       : _dstImg->getRowBytes()) / (int)sizeof(float);
    const int dstRowStride = _dstImg->getRowBytes() / (int)sizeof(float);
    const int w = window.x2 - window.x1;
    const int h = window.y2 - window.y1;

    for (int j = 0; j < h; ++j) {
        if (_effect.abort()) break;
        const float* srcRow = srcBase ? srcBase + (size_t)j * srcRowStride : nullptr;
        float*       dstRow = dstBase + (size_t)j * dstRowStride;
        for (int i = 0; i < w; ++i) {
            const int s = i * 4;
            float3 in_ = srcRow ? make_float3(srcRow[s + 0], srcRow[s + 1], srcRow[s + 2])
                                : make_float3(0.f, 0.f, 0.f);
            float a    = srcRow ? srcRow[s + 3] : 1.f;
            float3 out = chromagnon_processPixel(in_, window.x1 + i, window.y1 + j,
                                                 w, h, &_params);
            dstRow[s + 0] = out.x;
            dstRow[s + 1] = out.y;
            dstRow[s + 2] = out.z;
            dstRow[s + 3] = a;
        }
    }
}

// -- OpenDRTPlugin -------------------------------------------------------------
OpenDRTPlugin::OpenDRTPlugin(OfxImageEffectHandle handle)
    : OFX::ImageEffect(handle)
{
    _srcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);
    _dstClip = fetchClip(kOfxImageEffectOutputClipName);

    _inGamut         = fetchChoiceParam(kODT_InGamut);
    _inOetf          = fetchChoiceParam(kODT_InOetf);
    _displayGamut    = fetchChoiceParam(kODT_DisplayGamut);
    _eotf            = fetchChoiceParam(kODT_Eotf);
    _lookPreset      = fetchChoiceParam(kODT_LookPreset);
    _tonescalePreset = fetchChoiceParam(kODT_TonescalePreset);
    _lockStickshift  = fetchBooleanParam(kODT_LockStickshift);
    _lockHcon        = fetchBooleanParam(kODT_LockHcon);
    _lockLcon        = fetchBooleanParam(kODT_LockLcon);
    _lockPtl         = fetchBooleanParam(kODT_LockPtl);
    _lockPtm         = fetchBooleanParam(kODT_LockPtm);
    _lockBrl         = fetchBooleanParam(kODT_LockBrl);
    _lockHsRgb       = fetchBooleanParam(kODT_LockHsRgb);
    _lockHsCmy       = fetchBooleanParam(kODT_LockHsCmy);
    _lockHc          = fetchBooleanParam(kODT_LockHc);

    _tnLp            = fetchDoubleParam(kODT_TnLp);
    _tnGb            = fetchDoubleParam(kODT_TnGb);
    _ptHdr           = fetchDoubleParam(kODT_PtHdr);

    _clamp           = fetchBooleanParam(kODT_Clamp);
    _tnLg            = fetchDoubleParam(kODT_TnLg);
    _tnCon           = fetchDoubleParam(kODT_TnCon);
    _tnSh            = fetchDoubleParam(kODT_TnSh);
    _tnToe           = fetchDoubleParam(kODT_TnToe);
    _tnOff           = fetchDoubleParam(kODT_TnOff);

    _tnHconEnable    = fetchBooleanParam(kODT_TnHconEnable);
    _tnHconOn = fetchBooleanParam(kODT_TnHconOn);
    _tnHcon          = fetchDoubleParam(kODT_TnHcon);
    _tnHconPv        = fetchDoubleParam(kODT_TnHconPv);
    _tnHconSt        = fetchDoubleParam(kODT_TnHconSt);

    _tnLconEnable    = fetchBooleanParam(kODT_TnLconEnable);
    _tnLconOn = fetchBooleanParam(kODT_TnLconOn);
    _tnLcon          = fetchDoubleParam(kODT_TnLcon);
    _tnLconW         = fetchDoubleParam(kODT_TnLconW);
    _tnLconPc        = fetchDoubleParam(kODT_TnLconPc);

    _cwp             = fetchChoiceParam(kODT_Cwp);
    _cwpRng          = fetchDoubleParam(kODT_CwpRng);
    _rsSa            = fetchDoubleParam(kODT_RsSa);
    _rsRw            = fetchDoubleParam(kODT_RsRw);
    _rsBw            = fetchDoubleParam(kODT_RsBw);

    _ptlEnable       = fetchBooleanParam(kODT_PtlEnable);
    _ptlOn = fetchBooleanParam(kODT_PtlOn);
    _ptR             = fetchDoubleParam(kODT_PtR);
    _ptG             = fetchDoubleParam(kODT_PtG);
    _ptB             = fetchDoubleParam(kODT_PtB);
    _ptRngLow        = fetchDoubleParam(kODT_PtRngLow);
    _ptRngHigh       = fetchDoubleParam(kODT_PtRngHigh);

    _ptmEnable       = fetchBooleanParam(kODT_PtmEnable);
    _ptmOn = fetchBooleanParam(kODT_PtmOn);
    _ptmLow          = fetchDoubleParam(kODT_PtmLow);
    _ptmLowSt        = fetchDoubleParam(kODT_PtmLowSt);
    _ptmHigh         = fetchDoubleParam(kODT_PtmHigh);
    _ptmHighSt       = fetchDoubleParam(kODT_PtmHighSt);

    _brlEnable       = fetchBooleanParam(kODT_BrlEnable);
    _brlOn = fetchBooleanParam(kODT_BrlOn);
    _brlR            = fetchDoubleParam(kODT_BrlR);
    _brlG            = fetchDoubleParam(kODT_BrlG);
    _brlB            = fetchDoubleParam(kODT_BrlB);
    _brlC            = fetchDoubleParam(kODT_BrlC);
    _brlM            = fetchDoubleParam(kODT_BrlM);
    _brlY            = fetchDoubleParam(kODT_BrlY);
    _brlRng          = fetchDoubleParam(kODT_BrlRng);

    _hsRgbEnable     = fetchBooleanParam(kODT_HsRgbEnable);
    _hsRgbOn = fetchBooleanParam(kODT_HsRgbOn);
    _hsR             = fetchDoubleParam(kODT_HsR);
    _hsG             = fetchDoubleParam(kODT_HsG);
    _hsB             = fetchDoubleParam(kODT_HsB);
    _hsRgbRng        = fetchDoubleParam(kODT_HsRgbRng);

    _hsCmyEnable     = fetchBooleanParam(kODT_HsCmyEnable);
    _hsCmyOn = fetchBooleanParam(kODT_HsCmyOn);
    _hsC             = fetchDoubleParam(kODT_HsC);
    _hsM             = fetchDoubleParam(kODT_HsM);
    _hsY             = fetchDoubleParam(kODT_HsY);

    _hcEnable        = fetchBooleanParam(kODT_HcEnable);
    _hcOn = fetchBooleanParam(kODT_HcOn);
    _hcR             = fetchDoubleParam(kODT_HcR);

    _advHueContrast  = fetchBooleanParam(kODT_AdvHueContrast);
    _advHcR          = fetchDoubleParam(kODT_AdvHcR);
    _advHcG          = fetchDoubleParam(kODT_AdvHcG);
    _advHcB          = fetchDoubleParam(kODT_AdvHcB);
    _advHcC          = fetchDoubleParam(kODT_AdvHcC);
    _advHcM          = fetchDoubleParam(kODT_AdvHcM);
    _advHcY          = fetchDoubleParam(kODT_AdvHcY);
    _advHcPower      = fetchDoubleParam(kODT_AdvHcPower);

    _betaFeatures    = fetchBooleanParam(kODT_BetaFeatures);
    _filmicMode      = fetchBooleanParam(kODT_FilmicMode);
    _filmicSrcStops  = fetchDoubleParam(kODT_FilmicSrcStops);
    _filmicTgtStops  = fetchDoubleParam(kODT_FilmicTgtStops);
    _filmicDynRange  = fetchDoubleParam(kODT_FilmicDynRange);
    _filmicStrength  = fetchDoubleParam(kODT_FilmicStrength);
    _filmicProjSim   = fetchChoiceParam(kODT_FilmicProjSim);
    _tonescaleMap    = fetchBooleanParam(kODT_TonescaleMap);
    _diagnosticsMode = fetchBooleanParam(kODT_DiagnosticsMode);
    _rgbChips        = fetchBooleanParam(kODT_RgbChips);

    // -- Feature-detail groups (hidden until their enable toggle is on) --
    _highContrastGroup       = fetchGroupParam("HighContrastGroup");
    _lowContrastGroup        = fetchGroupParam("LowContrastGroup");
    _purityLowGroup          = fetchGroupParam("PurityLowGroup");
    _midPurityGroup          = fetchGroupParam("MidPurityGroup");
    _brillianceGroup         = fetchGroupParam("BrillianceGroup");
    _hueshiftRgbGroup        = fetchGroupParam("HueshiftRgbGroup");
    _hueshiftCmyGroup        = fetchGroupParam("HueshiftCmyGroup");
    _hueContrastGroup        = fetchGroupParam("HueContrastGroup");
    _advHueContrastGroup     = fetchGroupParam("AdvHueContrastGroup");
    _filmicDynamicRangeGroup = fetchGroupParam("FilmicDynamicRangeGroup");
    _filmicProjectorSimGroup = fetchGroupParam("FilmicProjectorSimGroup");

    // Current-model section groups, hidden while the v1.1 model is selected.
    _presetGroup       = fetchGroupParam("PresetGroup");
    _stickshiftGroup   = fetchGroupParam("StickshiftGroup");
    _tonescaleGroup    = fetchGroupParam("TonescaleGroup");
    _contrastGroup     = fetchGroupParam("ContrastGroup");
    _colorGroup        = fetchGroupParam("ColorGroup");
    _betaFeaturesGroup = fetchGroupParam("BetaFeaturesGroup");

    // -- Model selector and the v1.1 model params --
    _model = fetchChoiceParam(kODT_Model);
    for (int i = 0; i < ODT11_NDESC; ++i) {
        const Odt11Desc& d = kOdt11Desc[i];
        switch (d.kind) {
        case ODT11_DOUBLE: _v11Double[i] = fetchDoubleParam(d.name);  break;
        case ODT11_BOOL:   _v11Bool[i]   = fetchBooleanParam(d.name); break;
        default:           _v11Choice[i] = fetchChoiceParam(d.name);  break;
        }
    }
    // Slot 0 is the shared Input group, already visible on every model.
    for (int g = 1; g < ODT11_G_COUNT; ++g)
        _v11Groups[g] = fetchGroupParam(kOdt11Groups[g].name);

    // -- Tesseract and Chromagnon params (slot 0 is the shared Input group) --
    for (int i = 0; i < TESS_NDESC; ++i) {
        const TessDesc& d = kTessDesc[i];
        switch (d.kind) {
        case TESS_DOUBLE: _tessDouble[i] = fetchDoubleParam(d.name);  break;
        case TESS_BOOL:   _tessBool[i]   = fetchBooleanParam(d.name); break;
        default:          _tessChoice[i] = fetchChoiceParam(d.name);  break;
        }
    }
    for (int g = 1; g < TESS_G_COUNT; ++g)
        _tessGroups[g] = fetchGroupParam(kTessGroups[g].name);
    for (int i = 0; i < CMG_NDESC; ++i) {
        const CmgDesc& d = kCmgDesc[i];
        switch (d.kind) {
        case CMG_DOUBLE: _cmgDouble[i] = fetchDoubleParam(d.name);  break;
        case CMG_BOOL:   _cmgBool[i]   = fetchBooleanParam(d.name); break;
        default:         _cmgChoice[i] = fetchChoiceParam(d.name);  break;
        }
    }
    for (int g = 1; g < CMG_G_COUNT; ++g)
        _cmgGroups[g] = fetchGroupParam(kCmgGroups[g].name);
    _diagnosticsGroup = fetchGroupParam("DiagnosticsGroup");

    // -- My Presets --
    _myPreset   = fetchChoiceParam(kODT_MyPreset);
    _pmSelected = fetchStringParam(kODT_PmSelected);
    _pmFamily   = fetchChoiceParam(kODT_PmFamily);
    _pmName     = fetchStringParam(kODT_PmName);
    _pmUpdate   = fetchPushButtonParam(kODT_PmUpdate);
    _pmRename   = fetchPushButtonParam(kODT_PmRename);
    _pmDelete   = fetchPushButtonParam(kODT_PmDelete);

    // Seed initial visibility from the restored param state.
    refreshVisibility();

    // Option list only: reads the store and the restored hidden ref, never
    // calls setValue. A stale my_preset index is fixed later (beginEdit or the
    // next user edit).
    rebuildPresetMenu(0.0, false);

    // Factory Settings: before the control channel binds (bind mints the uuid,
    // which the new-node test reads). Writes nothing persistent on an old node.
    _factory.reset(new ldt::factory::Controller(*this, OPENDRT_PLUGIN_ID,
        [this](double time) {
            refreshVisibility();
            // A starting point can switch the model, which reorders the menu.
            // In the ctor (new node) the index fix-up is deferred.
            rebuildPresetMenu(time, _projectLoaded);
        }));
    _factory->onConstructed();

    // -- Claude control channel --
    // Last, after every fetch above: bind() re-fetches each whitelisted param by
    // name. Disarmed this costs one stat per render and starts no thread.
    _control.reset(new ldt::control::Channel(*this, "6A",
                                             kODT_ControlSpecs, kODT_ControlSpecCount));
    _control->bind();
}

OpenDRTPlugin::~OpenDRTPlugin() {
    if (_control) _control->shutdown();
}

// The Stickshift Mode sub-groups are always shown on the Current UI; each one
// carries its own Enable checkbox so a module can be flipped on and off while
// its sliders stay in view. Only the two Filmic groups are still gated: they
// need the Beta Features master gate plus Filmic Mode.
void OpenDRTPlugin::setEnabledness() {
    bool beta = false;   _betaFeatures->getValue(beta);
    bool filmic = false; _filmicMode->getValue(filmic);

    // Filmic Mode is itself greyed out until Beta Features is on.
    _filmicMode->setEnabled(beta);

    _filmicDynamicRangeGroup->setIsSecret(!(beta && filmic));
    _filmicProjectorSimGroup->setIsSecret(!(beta && filmic));
}

int OpenDRTPlugin::modelAt(double time) const {
    int m = 0;
    _model->getValueAtTime(time, m);
    return (m >= 0 && m < ODT_MODEL_COUNT) ? m : ODT_MODEL_CURRENT;
}

// Show the controls of the selected model only. Current and v1.0.1 share
// today's UI exactly; v1.1 shows its own odt11_* params plus the two shared
// params (Input Gamut, and Tonescale Curve which drives the v1.1 curve view).
// Tesseract and Chromagnon show their own tess_* / cmg_* params plus Input
// Gamut; neither has a curve view, so the Diagnostics group is hidden.
void OpenDRTPlugin::refreshVisibility() {
    int m = 0;
    _model->getValue(m);
    if (m < 0 || m >= ODT_MODEL_COUNT) m = ODT_MODEL_CURRENT;

    const bool cur  = (m == ODT_MODEL_CURRENT || m == ODT_MODEL_V101);
    const bool v11  = (m == ODT_MODEL_V11);
    const bool tess = (m == ODT_MODEL_TESSERACT);
    const bool cmg  = (m == ODT_MODEL_CHROMAGNON);

    // Current-model params that sit in groups shared with the other models.
    _inOetf->setIsSecret(!cur);
    _displayGamut->setIsSecret(!cur);
    _eotf->setIsSecret(!cur);
    _diagnosticsMode->setIsSecret(!cur);
    _rgbChips->setIsSecret(!cur);

    // Tonescale Curve is shared by Current, v1.0.1 and v1.1 only.
    _tonescaleMap->setIsSecret(!(cur || v11));
    if (_diagnosticsGroup) _diagnosticsGroup->setIsSecret(!(cur || v11));

    // Current-model section groups.
    _presetGroup->setIsSecret(!cur);
    _stickshiftGroup->setIsSecret(!cur);
    _tonescaleGroup->setIsSecret(!cur);
    _contrastGroup->setIsSecret(!cur);
    _colorGroup->setIsSecret(!cur);
    _betaFeaturesGroup->setIsSecret(!cur);

    // Stickshift Mode sub-groups: set explicitly rather than relying on the
    // hidden parent, so every host shows and hides them with the model.
    _highContrastGroup->setIsSecret(!cur);
    _lowContrastGroup->setIsSecret(!cur);
    _purityLowGroup->setIsSecret(!cur);
    _midPurityGroup->setIsSecret(!cur);
    _brillianceGroup->setIsSecret(!cur);
    _hueshiftRgbGroup->setIsSecret(!cur);
    _hueshiftCmyGroup->setIsSecret(!cur);
    _hueContrastGroup->setIsSecret(!cur);
    _advHueContrastGroup->setIsSecret(!cur);
    // Advanced Hue Contrast no longer depends on Beta Features; clear any
    // greyed-out state an older build may have left on the instance.
    _advHueContrast->setEnabled(true);

    if (!cur) {
        _filmicDynamicRangeGroup->setIsSecret(true);
        _filmicProjectorSimGroup->setIsSecret(true);
    } else {
        setEnabledness();   // the Filmic groups follow Beta Features + Filmic Mode
    }

    // v1.1 params and groups.
    for (int i = 0; i < ODT11_NDESC; ++i) {
        if (_v11Double[i]) _v11Double[i]->setIsSecret(!v11);
        if (_v11Bool[i])   _v11Bool[i]->setIsSecret(!v11);
        if (_v11Choice[i]) _v11Choice[i]->setIsSecret(!v11);
    }
    for (int g = 1; g < ODT11_G_COUNT; ++g)
        if (_v11Groups[g]) _v11Groups[g]->setIsSecret(!v11);

    // Tesseract params and groups.
    for (int i = 0; i < TESS_NDESC; ++i) {
        if (_tessDouble[i]) _tessDouble[i]->setIsSecret(!tess);
        if (_tessBool[i])   _tessBool[i]->setIsSecret(!tess);
        if (_tessChoice[i]) _tessChoice[i]->setIsSecret(!tess);
    }
    for (int g = 1; g < TESS_G_COUNT; ++g)
        if (_tessGroups[g]) _tessGroups[g]->setIsSecret(!tess);

    // Chromagnon params and groups.
    for (int i = 0; i < CMG_NDESC; ++i) {
        if (_cmgDouble[i]) _cmgDouble[i]->setIsSecret(!cmg);
        if (_cmgBool[i])   _cmgBool[i]->setIsSecret(!cmg);
        if (_cmgChoice[i]) _cmgChoice[i]->setIsSecret(!cmg);
    }
    for (int g = 1; g < CMG_G_COUNT; ++g)
        if (_cmgGroups[g]) _cmgGroups[g]->setIsSecret(!cmg);
}

// Slider values of every v1.1 param, in kOdt11Desc order.
void OpenDRTPlugin::readV11Values(double time, double* v) const {
    for (int i = 0; i < ODT11_NDESC; ++i) {
        v[i] = 0.0;
        if (_v11Double[i]) {
            _v11Double[i]->getValueAtTime(time, v[i]);
        } else if (_v11Bool[i]) {
            bool b = false; _v11Bool[i]->getValueAtTime(time, b); v[i] = b ? 1.0 : 0.0;
        } else if (_v11Choice[i]) {
            int c = 0; _v11Choice[i]->getValueAtTime(time, c); v[i] = (double)c;
        }
    }
}

// Sliders are the source of truth; presets only ever write the sliders.
OpenDRTv11Params OpenDRTPlugin::buildV11Params(double time) const {
    double v[ODT11_NDESC];
    readV11Values(time, v);
    int inGamut = 15;
    _inGamut->getValueAtTime(time, inGamut);
    bool crv = false;
    _tonescaleMap->getValueAtTime(time, crv);
    OpenDRTv11Params p;
    odt11_build(v, inGamut, crv ? 1 : 0, &p);
    return p;
}

// Control values of every Tesseract param, in kTessDesc order.
void OpenDRTPlugin::readTessValues(double time, double* v) const {
    for (int i = 0; i < TESS_NDESC; ++i) {
        v[i] = 0.0;
        if (_tessDouble[i]) {
            _tessDouble[i]->getValueAtTime(time, v[i]);
        } else if (_tessBool[i]) {
            bool b = false; _tessBool[i]->getValueAtTime(time, b); v[i] = b ? 1.0 : 0.0;
        } else if (_tessChoice[i]) {
            int c = 0; _tessChoice[i]->getValueAtTime(time, c); v[i] = (double)c;
        }
    }
}

TesseractParams OpenDRTPlugin::buildTessParams(double time) const {
    double v[TESS_NDESC];
    readTessValues(time, v);
    int inGamut = 15;
    _inGamut->getValueAtTime(time, inGamut);
    TesseractParams p;
    tess_build(v, inGamut, &p);
    return p;
}

// Control values of every Chromagnon param, in kCmgDesc order.
void OpenDRTPlugin::readCmgValues(double time, double* v) const {
    for (int i = 0; i < CMG_NDESC; ++i) {
        v[i] = 0.0;
        if (_cmgDouble[i]) {
            _cmgDouble[i]->getValueAtTime(time, v[i]);
        } else if (_cmgBool[i]) {
            bool b = false; _cmgBool[i]->getValueAtTime(time, b); v[i] = b ? 1.0 : 0.0;
        } else if (_cmgChoice[i]) {
            int c = 0; _cmgChoice[i]->getValueAtTime(time, c); v[i] = (double)c;
        }
    }
}

ChromagnonParams OpenDRTPlugin::buildCmgParams(double time) const {
    double v[CMG_NDESC];
    readCmgValues(time, v);
    int inGamut = 15;
    _inGamut->getValueAtTime(time, inGamut);
    ChromagnonParams p;
    cmg_build(v, inGamut, &p);
    return p;
}

// Write a v1.1 preset's rows into the sliders. Only on eChangeUserEdit (see
// changedParam), so a project load never overwrites restored values. Only
// params whose value changes are written.
void OpenDRTPlugin::applyV11Rows(int which, double time) {
    double before[ODT11_NDESC], after[ODT11_NDESC];
    readV11Values(time, before);
    for (int i = 0; i < ODT11_NDESC; ++i) after[i] = before[i];
    odt11_write_rows(after, which);

    for (int i = 0; i < ODT11_NDESC; ++i) {
        if (_v11Double[i]) {
            if ((float)after[i] != (float)before[i]) _v11Double[i]->setValue(after[i]);
        } else if (_v11Bool[i]) {
            const bool a = after[i] != 0.0, b = before[i] != 0.0;
            if (a != b) _v11Bool[i]->setValue(a);
        } else if (_v11Choice[i]) {
            const int a = (int)std::lround(after[i]), b = (int)std::lround(before[i]);
            if (a != b) _v11Choice[i]->setValue(a);
        }
    }
}

// Write the chosen Look/Tonescale preset's values into the UI sliders. Mirrors
// canonical OpenDRT applyPresetValues(): basic tonescale / creative-white /
// render-space are always written; each feature module is written only if its
// "Persistent Values" lock is off (the master Lock Manual Adjustments forces all
// locks on). Called only on eChangeUserEdit (see changedParam) so a project
// reload never clobbers restored slider values.
void OpenDRTPlugin::applyPresetValues(const OFX::InstanceChangedArgs& args) {
    int lookPreset = 0;      _lookPreset->getValueAtTime(args.time, lookPreset);
    int tonescalePreset = 0; _tonescalePreset->getValueAtTime(args.time, tonescalePreset);

    bool master = false; _lockStickshift->getValue(master);
    auto locked = [&](OFX::BooleanParam* p) {
        bool b = false; p->getValue(b); return b || master;
    };
    const bool lockHcon  = locked(_lockHcon);
    const bool lockLcon  = locked(_lockLcon);
    const bool lockPtl   = locked(_lockPtl);
    const bool lockPtm   = locked(_lockPtm);
    const bool lockBrl   = locked(_lockBrl);
    const bool lockHsRgb = locked(_lockHsRgb);
    const bool lockHsCmy = locked(_lockHsCmy);
    const bool lockHc    = locked(_lockHc);

    // A preset write disarms the module it writes; the on switches stay put.
    auto disarm = [](OFX::BooleanParam* armed) {
        bool b = false; armed->getValue(b);
        if (b) armed->setValue(false);
    };

    if (lookPreset >= 0 && lookPreset <= 3) {
        const LookPreset& p = kLookPresets[lookPreset];

        // Always (never locked): basic tonescale, creative white, render space.
        _tnLg->setValue(p.tn_Lg);   _tnCon->setValue(p.tn_con); _tnSh->setValue(p.tn_sh);
        _tnToe->setValue(p.tn_toe); _tnOff->setValue(p.tn_off);
        _cwp->setValue(p.cwp);      _cwpRng->setValue(p.cwp_rng);
        _rsSa->setValue(p.rs_sa);   _rsRw->setValue(p.rs_rw);   _rsBw->setValue(p.rs_bw);

        if (!lockHcon) { _tnHcon->setValue(p.tn_hcon); _tnHconPv->setValue(p.tn_hcon_pv); _tnHconSt->setValue(p.tn_hcon_st); }
        if (!lockLcon) { _tnLcon->setValue(p.tn_lcon); _tnLconW->setValue(p.tn_lcon_w);  _tnLconPc->setValue(p.tn_lcon_pc); }
        if (!lockPtl)  { _ptR->setValue(p.pt_r); _ptG->setValue(p.pt_g); _ptB->setValue(p.pt_b);
                         _ptRngLow->setValue(p.pt_rng_low); _ptRngHigh->setValue(p.pt_rng_high); }
        if (!lockPtm)  { _ptmLow->setValue(p.ptm_low); _ptmLowSt->setValue(p.ptm_low_st);
                         _ptmHigh->setValue(p.ptm_high); _ptmHighSt->setValue(p.ptm_high_st); }
        if (!lockBrl)  { _brlR->setValue(p.brl_r); _brlG->setValue(p.brl_g); _brlB->setValue(p.brl_b);
                         _brlC->setValue(p.brl_c); _brlM->setValue(p.brl_m); _brlY->setValue(p.brl_y);
                         _brlRng->setValue(p.brl_rng); }
        if (!lockHsRgb){ _hsR->setValue(p.hs_r); _hsG->setValue(p.hs_g); _hsB->setValue(p.hs_b);
                         _hsRgbRng->setValue(p.hs_rgb_rng); }
        if (!lockHsCmy){ _hsC->setValue(p.hs_c); _hsM->setValue(p.hs_m); _hsY->setValue(p.hs_y); }
        if (!lockHc)   { _hcR->setValue(p.hc_r); }

        if (!lockHcon)  disarm(_tnHconEnable);
        if (!lockLcon)  disarm(_tnLconEnable);
        if (!lockPtl)   disarm(_ptlEnable);
        if (!lockPtm)   disarm(_ptmEnable);
        if (!lockBrl)   disarm(_brlEnable);
        if (!lockHsRgb) disarm(_hsRgbEnable);
        if (!lockHsCmy) disarm(_hsCmyEnable);
        if (!lockHc)    disarm(_hcEnable);
    }

    // Tonescale preset: option 0 = "Use Look Preset" (no-op); 1-9 -> table[idx-1].
    if (tonescalePreset >= 1 && tonescalePreset <= 9) {
        const TonescalePreset& p = kTonescalePresets[tonescalePreset - 1];
        const int* flags = kTonescalePresetFlags[tonescalePreset - 1];
        _tnLg->setValue(p.tn_Lg);   _tnCon->setValue(p.tn_con); _tnSh->setValue(p.tn_sh);
        _tnToe->setValue(p.tn_toe); _tnOff->setValue(p.tn_off);
        // A gated-off module gets a neutral amount, so arming it later is a no-op.
        if (!lockHcon) { _tnHcon->setValue(flags[0] ? p.tn_hcon : 0.0); _tnHconPv->setValue(p.tn_hcon_pv); _tnHconSt->setValue(p.tn_hcon_st);
                         disarm(_tnHconEnable); }
        if (!lockLcon) { _tnLcon->setValue(flags[1] ? p.tn_lcon : 0.0); _tnLconW->setValue(p.tn_lcon_w);  _tnLconPc->setValue(p.tn_lcon_pc);
                         disarm(_tnLconEnable); }
    }

    setEnabledness();
}

// -- My Presets (user preset manager) -----------------------------------------
// Design: docs/design/OPENDRT-PRESET-MANAGER-DESIGN.md. Restore safety rules:
//   - values are written only on eChangeUserEdit, with _isApplying clear;
//   - the ctor only rebuilds the option list (no setValue);
//   - the hidden preset_manager_selected ref ("family|model|id|name") is the
//     truth, the my_preset index is a hint fixed up quietly later;
//   - the render never reads the store.
namespace {
namespace odp = ldt::opendrt_presets;
using ldt::presets::Family;
using ldt::presets::Preset;
using ldt::presets::Status;
using ldt::presets::ClashPolicy;
using ldt::presets::OpResult;
using ldt::presets::SelectedRef;

ldt::presets::Store& odtPresetStore() {
    static ldt::presets::Store store(ldt::presets::presetDirectory(), [] {
        ldt::presets::StoreConfig c = odp::modelConfig();
        c.pluginId = OPENDRT_PLUGIN_ID;   // recorded in new files
        return c;
    }());
    return store;
}

Family familyFromIndex(int i) {
    if (i == 1) return Family::Look;
    if (i == 2) return Family::Tonescale;
    return Family::Full;
}

std::string quotedName(const std::string& s) { return "\"" + s + "\""; }

SelectedRef refFor(const Preset& p) {
    SelectedRef r;
    r.family = p.family; r.model = p.model; r.id = p.id; r.name = p.name;
    return r;
}

// Restores a flag on scope exit.
struct ApplyGuard {
    bool& b; bool prev;
    explicit ApplyGuard(bool& f) : b(f), prev(f) { b = true; }
    ~ApplyGuard() { b = prev; }
};
} // namespace

void OpenDRTPlugin::beginEdit() {
    _projectLoaded = true;
    runPendingPresetFixup();
}

void OpenDRTPlugin::presetMessage(const std::string& msg, bool warning) {
    try {
        sendMessage(warning ? OFX::Message::eMessageWarning : OFX::Message::eMessageMessage,
                    "", msg);
    } catch (...) {}
}

bool OpenDRTPlugin::presetAsk(const std::string& msg, bool failedMeans) {
    OFX::Message::MessageReplyEnum r = OFX::Message::eMessageReplyFailed;
    try { r = sendMessage(OFX::Message::eMessageQuestion, "", msg); } catch (...) {}
    if (r == OFX::Message::eMessageReplyYes) return true;
    if (r == OFX::Message::eMessageReplyNo)  return false;
    return failedMeans;
}

void OpenDRTPlugin::setStringQuietly(OFX::StringParam* p, const std::string& v) {
    if (!p) return;
    try {
        std::string cur; p->getValue(cur);
        if (cur == v) return;
        ApplyGuard g(_isApplying);
        p->setValue(v);
    } catch (...) {}
}

void OpenDRTPlugin::setChoiceQuietly(OFX::ChoiceParam* p, int v) {
    if (!p) return;
    try {
        int cur = 0; p->getValue(cur);
        if (cur == v) return;
        ApplyGuard g(_isApplying);
        p->setValue(v);
    } catch (...) {}
}

void OpenDRTPlugin::setMyPresetIndexQuietly(int index) {
    _pendingMenuIndex = -1;
    setChoiceQuietly(_myPreset, index);
}

void OpenDRTPlugin::runPendingPresetFixup() {
    if (_pendingMenuIndex < 0) return;
    setMyPresetIndexQuietly(_pendingMenuIndex);
}

SelectedRef OpenDRTPlugin::readSelectedRef() const {
    SelectedRef r;
    if (!_pmSelected) return r;
    std::string s;
    try { _pmSelected->getValue(s); } catch (...) { return r; }
    if (!ldt::presets::decodeSelectedRef(s, r)) return SelectedRef();
    return r;
}

void OpenDRTPlugin::writeSelectedRef(const SelectedRef& r) {
    setStringQuietly(_pmSelected, r.empty() ? std::string() : ldt::presets::encodeSelectedRef(r));
}

int OpenDRTPlugin::selectedEntryIndex() const {
    const SelectedRef ref = readSelectedRef();
    if (ref.empty()) return -1;
    std::vector<Preset> list;
    list.reserve(_presetMenu.size());
    for (const auto& e : _presetMenu) list.push_back(e.preset);
    return ldt::presets::resolveSelectedRef(list, ref);
}

void OpenDRTPlugin::rebuildPresetMenu(double time, bool fixNow) {
    if (!_myPreset) return;
    std::vector<PresetMenuEntry> entries;
    try {
        ldt::presets::Store& st = odtPresetStore();
        st.reload();
        const int active = modelAt(time);
        // The active model's presets first, then the other models' presets.
        for (int pass = 0; pass < 2; ++pass) {
            for (int m = 0; m < odp::kModelCount; ++m) {
                if ((pass == 0) != (m == active)) continue;
                for (const Preset& p : st.list(odp::modelKeyFor(m))) {
                    PresetMenuEntry e; e.preset = p;
                    entries.push_back(e);
                }
            }
        }
    } catch (...) {}

    // A ref that matches nothing becomes a synthetic "(missing)" entry.
    const SelectedRef ref = readSelectedRef();
    int target = 0;
    if (!ref.empty()) {
        std::vector<Preset> list;
        for (const auto& e : entries) list.push_back(e.preset);
        const int hit = ldt::presets::resolveSelectedRef(list, ref);
        if (hit >= 0) {
            target = hit + 1;
        } else {
            PresetMenuEntry e;
            e.preset.family = ref.family; e.preset.model = ref.model;
            e.preset.id = ref.id; e.preset.name = ref.name;
            e.missing = true;
            entries.push_back(e);
            target = (int)entries.size();
        }
    }
    _presetMenu.swap(entries);

    try {
        _myPreset->resetOptions();
        _myPreset->appendOption("None");
        const int active = modelAt(time);
        for (const auto& e : _presetMenu) {
            std::string label;
            if (e.missing)
                label = ldt::presets::decorateLabel(ldt::presets::menuLabel(e.preset), false, true);
            else if (odp::modelIndexFor(e.preset.model) == active)
                label = ldt::presets::menuLabel(e.preset);
            else
                label = odp::crossModelLabel(e.preset);
            _myPreset->appendOption(label);
        }
    } catch (...) {}

    int cur = 0;
    try { _myPreset->getValue(cur); } catch (...) {}
    if (cur != target) {
        if (fixNow) setMyPresetIndexQuietly(target);
        else        _pendingMenuIndex = target;
    } else {
        _pendingMenuIndex = -1;
    }
    updatePresetButtons();
}

void OpenDRTPlugin::updatePresetButtons() {
    const int i = selectedEntryIndex();
    const bool real = i >= 0 && !_presetMenu[(size_t)i].missing;
    try {
        if (_pmUpdate) _pmUpdate->setEnabled(real);
        if (_pmRename) _pmRename->setEnabled(real);
        if (_pmDelete) _pmDelete->setEnabled(real);
    } catch (...) {}
}

const OpenDRTPlugin::PresetMenuEntry* OpenDRTPlugin::requireSelectedRecord() {
    const int i = selectedEntryIndex();
    if (i < 0) {
        presetMessage("Pick one of your presets first.");
        return nullptr;
    }
    const PresetMenuEntry& e = _presetMenu[(size_t)i];
    if (e.missing) {
        presetMessage("This preset isn't on this computer. The node keeps its settings. "
                      "Import the file to get it back.");
        return nullptr;
    }
    return &e;
}

std::map<std::string, double> OpenDRTPlugin::capturePresetLive(int model, double time) {
    std::map<std::string, double> live;
    for (const auto& d : odp::tableForModel(model)) {
        try {
            switch (d.type) {
            case ldt::presets::ParamType::Double: {
                double v = 0.0; fetchDoubleParam(d.id)->getValueAtTime(time, v); live[d.id] = v; break;
            }
            case ldt::presets::ParamType::Int: {
                int v = 0; fetchIntParam(d.id)->getValueAtTime(time, v); live[d.id] = (double)v; break;
            }
            case ldt::presets::ParamType::Bool: {
                bool v = false; fetchBooleanParam(d.id)->getValueAtTime(time, v); live[d.id] = v ? 1.0 : 0.0; break;
            }
            case ldt::presets::ParamType::Choice: {
                int v = 0; fetchChoiceParam(d.id)->getValueAtTime(time, v); live[d.id] = (double)v; break;
            }
            }
        } catch (...) {}
    }
    return live;
}

void OpenDRTPlugin::applyUserPreset(const Preset& p, double time) {
    const int model = odp::modelIndexFor(p.model);
    if (model < 0) return;
    const int active = modelAt(time);

    // Current and v1.0.1: a locked module (its lock, or the master lock) is
    // left alone, the same rule as applyPresetValues.
    ldt::presets::SkipFn skip;
    if (model == ODT_MODEL_CURRENT || model == ODT_MODEL_V101) {
        bool master = false;
        try { _lockStickshift->getValue(master); } catch (...) {}
        skip = [this, master](const std::string& id) {
            const char* lock = odp::lockFor(id);
            if (!lock) return false;
            if (master) return true;
            bool b = false;
            try { fetchBooleanParam(lock)->getValue(b); } catch (...) {}
            return b;
        };
    }
    ldt::presets::ResolveReport rep;
    const std::vector<ldt::presets::Resolved> rows =
        ldt::presets::resolveForApply(p, p.family, odp::tableForModel(model), skip, &rep);

    int keyed = 0;
    beginEditBlock("Apply preset " + p.name);
    try {
        ApplyGuard g(_isApplying);
        if (model != active) _model->setValue(model);
        for (const auto& r : rows) {
            try {
                OFX::ValueParam* vp = nullptr;
                switch (r.type) {
                case ldt::presets::ParamType::Double: {
                    OFX::DoubleParam* q = fetchDoubleParam(r.id); vp = q;
                    if (q->getNumKeys() > 0) { ++keyed; break; }
                    double cur = 0.0; q->getValueAtTime(time, cur);
                    if (cur != r.value) q->setValue(r.value);
                    break;
                }
                case ldt::presets::ParamType::Int: {
                    OFX::IntParam* q = fetchIntParam(r.id); vp = q;
                    if (q->getNumKeys() > 0) { ++keyed; break; }
                    const int v = (int)std::lround(r.value);
                    int cur = 0; q->getValueAtTime(time, cur);
                    if (cur != v) q->setValue(v);
                    break;
                }
                case ldt::presets::ParamType::Bool: {
                    OFX::BooleanParam* q = fetchBooleanParam(r.id); vp = q;
                    if (q->getNumKeys() > 0) { ++keyed; break; }
                    const bool v = r.value != 0.0;
                    bool cur = false; q->getValueAtTime(time, cur);
                    if (cur != v) q->setValue(v);
                    break;
                }
                case ldt::presets::ParamType::Choice: {
                    OFX::ChoiceParam* q = fetchChoiceParam(r.id); vp = q;
                    if (q->getNumKeys() > 0) { ++keyed; break; }
                    const int v = (int)std::lround(r.value);
                    int cur = 0; q->getValueAtTime(time, cur);
                    if (cur != v) q->setValue(v);
                    break;
                }
                }
                (void)vp;
            } catch (...) {}
        }
        // Save As and Name follow the pick so Update and Rename read naturally.
        int fam = 0; _pmFamily->getValue(fam);
        if (fam != (int)p.family) _pmFamily->setValue((int)p.family);
        std::string nm; _pmName->getValue(nm);
        if (nm != p.name) _pmName->setValue(p.name);
        _pmSelected->setValue(ldt::presets::encodeSelectedRef(refFor(p)));
    } catch (...) {}
    refreshVisibility();   // also runs setEnabledness on Current / v1.0.1
    endEditBlock();

    rebuildPresetMenu(time, true);

    if (keyed > 0) {
        presetMessage("Left " + std::to_string(keyed) + " keyframed control(s) unchanged. "
                      "Remove their keyframes to apply a preset to them.", true);
    }
}

void OpenDRTPlugin::onMyPresetPicked(double time) {
    int idx = 0;
    try { _myPreset->getValue(idx); } catch (...) {}
    _pendingMenuIndex = -1;

    if (idx <= 0 || idx > (int)_presetMenu.size()) {
        // None: forget the selection. No grade value changes.
        writeSelectedRef(SelectedRef());
        rebuildPresetMenu(time, true);
        return;
    }
    const PresetMenuEntry entry = _presetMenu[(size_t)idx - 1];
    // Put the menu back on the current selection (used on every early return).
    auto restore = [&] { rebuildPresetMenu(time, true); };

    if (entry.missing) {
        presetMessage("This preset isn't on this computer. The node keeps its settings. "
                      "Import the file to get it back.");
        restore();
        return;
    }
    OpResult r = odtPresetStore().loadById(entry.preset.id);
    if (!r.ok()) {
        presetMessage("This preset isn't on this computer any more. The node keeps its settings.");
        restore();
        return;
    }
    const Preset& p = r.preset;
    const int model = odp::modelIndexFor(p.model);
    if (model < 0) { restore(); return; }
    if (model != modelAt(time)) {
        const std::string label = odp::modelLabelFor(model);
        if (!presetAsk("This preset uses " + label + ". Switch to " + label + " and apply it?",
                       true)) {
            restore();
            return;
        }
    }
    applyUserPreset(p, time);
}

void OpenDRTPlugin::onPresetSaveNew(double time) {
    int famIdx = 0; std::string name;
    try { _pmFamily->getValue(famIdx); _pmName->getValue(name); } catch (...) {}
    const int model = modelAt(time);
    const Family fam = familyFromIndex(famIdx);
    // Saving ignores locks; locks only matter on apply.
    Preset p = ldt::presets::capturePreset(name, fam, odp::modelKeyFor(model),
                                           odp::tableForModel(model),
                                           capturePresetLive(model, time));
    ldt::presets::Store& st = odtPresetStore();
    st.reload();
    OpResult r = st.saveNew(p, ClashPolicy::Fail);
    if (r.status == Status::NameClash) {
        const bool replace = presetAsk(
            "You already have a preset called " + quotedName(r.preset.name) +
            ". Replace it? (No keeps both.)", false);
        r = st.saveNew(p, replace ? ClashPolicy::Replace : ClashPolicy::KeepBoth);
    }
    if (!r.ok()) {
        presetMessage(r.reason.empty() ? "Couldn't save the preset." : r.reason, true);
        return;
    }
    writeSelectedRef(refFor(r.preset));
    setStringQuietly(_pmName, r.preset.name);
    rebuildPresetMenu(time, true);
}

void OpenDRTPlugin::onPresetUpdate(double time) {
    const PresetMenuEntry* e = requireSelectedRecord();
    if (!e) return;
    const Preset rec = e->preset;
    const int model = odp::modelIndexFor(rec.model);
    if (model != modelAt(time)) {
        const std::string label = odp::modelLabelForKey(rec.model);
        presetMessage("This preset uses " + label + ". Switch to " + label + " to update it.");
        return;
    }
    if (!presetAsk("Overwrite " + quotedName(rec.name) + " with the current settings?", true)) return;
    // The record's own family, not Save As.
    const Preset fresh = ldt::presets::capturePreset(rec.name, rec.family, rec.model,
                                                     odp::tableForModel(model),
                                                     capturePresetLive(model, time));
    OpResult r = odtPresetStore().update(rec.id, fresh.values);
    if (!r.ok()) {
        presetMessage(r.reason.empty() ? "Couldn't update the preset." : r.reason, true);
        return;
    }
    writeSelectedRef(refFor(r.preset));
    rebuildPresetMenu(time, true);
}

void OpenDRTPlugin::onPresetRename(double time) {
    const PresetMenuEntry* e = requireSelectedRecord();
    if (!e) return;
    const Preset rec = e->preset;
    std::string name;
    try { _pmName->getValue(name); } catch (...) {}
    OpResult r = odtPresetStore().rename(rec.id, name);
    if (r.status == Status::NameClash) {
        presetMessage("You already have a preset called " + quotedName(r.preset.name) + ".");
        return;
    }
    if (!r.ok()) {
        presetMessage(r.reason.empty() ? "Couldn't rename the preset." : r.reason, true);
        return;
    }
    writeSelectedRef(refFor(r.preset));
    setStringQuietly(_pmName, r.preset.name);
    rebuildPresetMenu(time, true);
}

void OpenDRTPlugin::onPresetDelete(double time) {
    const PresetMenuEntry* e = requireSelectedRecord();
    if (!e) return;
    const Preset rec = e->preset;
    if (!presetAsk("Delete " + quotedName(rec.name) + "? Nodes using it keep their settings.", true))
        return;
    OpResult r = odtPresetStore().remove(rec.id);
    if (!r.ok() && r.status != Status::NotFound) {
        presetMessage(r.reason.empty() ? "Couldn't delete the preset." : r.reason, true);
        return;
    }
    writeSelectedRef(SelectedRef());
    rebuildPresetMenu(time, true);
}

void OpenDRTPlugin::onPresetExport(double time) {
    ldt::presets::Store& st = odtPresetStore();
    st.reload();
    const int i = selectedEntryIndex();
    const bool real = i >= 0 && !_presetMenu[(size_t)i].missing;

    Preset out;
    if (real) {
        OpResult r = st.loadById(_presetMenu[(size_t)i].preset.id);
        if (!r.ok()) {
            presetMessage("This preset isn't on this computer any more.", true);
            rebuildPresetMenu(time, true);
            return;
        }
        out = r.preset;
    } else {
        // None or missing: the live values of Save As under Name (a missing
        // entry keeps its own name and family, the easy way to recreate it).
        int famIdx = 0; std::string name;
        try { _pmFamily->getValue(famIdx); _pmName->getValue(name); } catch (...) {}
        Family fam = familyFromIndex(famIdx);
        if (i >= 0) { name = _presetMenu[(size_t)i].preset.name; fam = _presetMenu[(size_t)i].preset.family; }
        const int model = modelAt(time);
        out = ldt::presets::capturePreset(
            ldt::presets::sanitizePresetName(name, st.config().knownModels), fam,
            odp::modelKeyFor(model), odp::tableForModel(model), capturePresetLive(model, time));
        out.plugin = st.config().pluginId;
    }
    const std::string suggested =
        ldt::presets::sanitizePresetName(out.name, st.config().knownModels) + ".json";
    const std::string dest = ldt::dialog::chooseSavePath(
        "Export OpenDRT Preset", suggested, "json", ldt::dialog::defaultDocumentsDirectory());
    if (dest.empty()) return;   // cancelled
    OpResult w = ldt::presets::Store::exportPreset(out, std::filesystem::u8path(dest));
    if (!w.ok())
        presetMessage(w.reason.empty() ? "Couldn't write the file." : w.reason, true);
}

void OpenDRTPlugin::onPresetImport(double time) {
    const std::string src = ldt::dialog::chooseOpenPath(
        "Import OpenDRT Preset", "json", ldt::dialog::defaultDocumentsDirectory());
    if (src.empty()) return;   // cancelled
    ldt::presets::Store& st = odtPresetStore();
    st.reload();
    const std::filesystem::path path = std::filesystem::u8path(src);
    OpResult r = st.importFrom(path, ClashPolicy::Fail);
    if (r.status == Status::SameId) {
        if (!presetAsk("You already have " + quotedName(r.preset.name) +
                       ". Replace it with this file?", false))
            return;
        r = st.importFrom(path, ClashPolicy::Replace);
    } else if (r.status == Status::NameClash) {
        const bool replace = presetAsk(
            "You already have a preset called " + quotedName(r.preset.name) +
            ". Replace it? (No keeps both.)", false);
        r = st.importFrom(path, replace ? ClashPolicy::Replace : ClashPolicy::KeepBoth);
    }
    if (!r.ok()) {
        presetMessage(r.reason.empty() ? "Couldn't import the preset." : r.reason, true);
        return;
    }
    // Import adds the preset to the menu; it does not apply it.
    rebuildPresetMenu(time, true);
    std::string note = "Imported " + quotedName(r.preset.name) + ".";
    if (r.skipped > 0)
        note += " " + std::to_string(r.skipped) +
                " setting(s) were skipped because this version doesn't know them.";
    presetMessage(note);
}

void OpenDRTPlugin::onPresetShowFolder() {
    const std::string dir = odtPresetStore().directory().u8string();
    if (!ldt::dialog::revealFolder(dir))
        presetMessage("Couldn't open the preset folder: " + dir, true);
}

bool OpenDRTPlugin::handlePresetManager(const OFX::InstanceChangedArgs& args,
                                        const std::string& paramName) {
    const bool mine = paramName == kODT_MyPreset ||
                      paramName.compare(0, std::strlen(kODT_PmPrefix), kODT_PmPrefix) == 0;
    if (!mine) return false;
    // Restored values replay as plugin edits: never act on them.
    if (args.reason != OFX::eChangeUserEdit || !_projectLoaded || _isApplying) return true;

    if      (paramName == kODT_MyPreset)    onMyPresetPicked(args.time);
    else if (paramName == kODT_PmSaveNew)   onPresetSaveNew(args.time);
    else if (paramName == kODT_PmUpdate)    onPresetUpdate(args.time);
    else if (paramName == kODT_PmRename)    onPresetRename(args.time);
    else if (paramName == kODT_PmDelete)    onPresetDelete(args.time);
    else if (paramName == kODT_PmExport)    onPresetExport(args.time);
    else if (paramName == kODT_PmImport)    onPresetImport(args.time);
    else if (paramName == kODT_PmShowFolder) onPresetShowFolder();
    return true;
}

// Re-evaluate group visibility whenever an enable/beta toggle changes, and load
// preset values when the user picks a Look/Tonescale preset.
void OpenDRTPlugin::changedParam(const OFX::InstanceChangedArgs& args,
                                 const std::string& paramName) {
    if (_factory && _factory->handleChangedParam(args, paramName)) return;

    // Params written by the plugin itself (a preset apply, a quiet fix-up)
    // re-enter here; tool logic must not react to half-applied state.
    if (_isApplying) return;

    if (args.reason == OFX::eChangeUserEdit) {
        _projectLoaded = true;
        if (paramName != kODT_MyPreset) runPendingPresetFixup();
    }

    if (handlePresetManager(args, paramName)) return;

    if (paramName == kODT_Model) {
        refreshVisibility();
        // The menu lists the active model's presets first.
        rebuildPresetMenu(args.time, args.reason == OFX::eChangeUserEdit);
        return;
    }

    // v1.1 presets write their rows on explicit user edit only.
    if (args.reason == OFX::eChangeUserEdit) {
        if (paramName == kOdt11Desc[ODT11_I_LOOK].name) {
            applyV11Rows(ODT11_ROWS_LOOK, args.time); return;
        }
        if (paramName == kOdt11Desc[ODT11_I_TONESCALE].name) {
            applyV11Rows(ODT11_ROWS_TONESCALE, args.time); return;
        }
        if (paramName == kOdt11Desc[ODT11_I_ENCODING].name) {
            applyV11Rows(ODT11_ROWS_ENCODING, args.time); return;
        }
    }

    // A user edit to a module slider arms that module (hidden legacy flag).
    // Never the on switch or the armed flag itself.
    if (args.reason == OFX::eChangeUserEdit) {
        if (const char* lock = odp::lockFor(paramName)) {
            const std::string id(lock);
            OFX::BooleanParam* armed =
                id == kODT_LockHcon  ? _tnHconEnable :
                id == kODT_LockLcon  ? _tnLconEnable :
                id == kODT_LockPtl   ? _ptlEnable :
                id == kODT_LockPtm   ? _ptmEnable :
                id == kODT_LockBrl   ? _brlEnable :
                id == kODT_LockHsRgb ? _hsRgbEnable :
                id == kODT_LockHsCmy ? _hsCmyEnable :
                id == kODT_LockHc    ? _hcEnable : nullptr;
            const bool isSwitch = armed && (paramName == armed->getName() ||
                (paramName.size() > 3 && paramName.compare(paramName.size() - 3, 3, "_on") == 0));
            if (armed && !isSwitch) {
                bool b = false; armed->getValue(b);
                if (!b) { ApplyGuard g(_isApplying); armed->setValue(true); }
            }
        }
    }

    if (paramName == kODT_BetaFeatures || paramName == kODT_FilmicMode) {
        // The toggles only exist on the Current-model UI (Current and v1.0.1);
        // every other model keeps them hidden.
        const int m = modelAt(args.time);
        if (m == ODT_MODEL_CURRENT || m == ODT_MODEL_V101) setEnabledness();
    } else if ((paramName == kODT_LookPreset || paramName == kODT_TonescalePreset) &&
               args.reason == OFX::eChangeUserEdit) {
        // Only on explicit user edit - never during project restore, or the
        // setValue() calls would overwrite the user's saved slider state.
        // Guarded so the preset's own writes can never arm a module.
        ApplyGuard g(_isApplying);
        applyPresetValues(args);
    }
}

OpenDRTParams OpenDRTPlugin::buildParams(double time, int /*w*/, int /*h*/) const {
    OpenDRTParams p{};

    // -- Choice params --
    _inGamut->getValueAtTime(time, p.inGamut);
    _inOetf->getValueAtTime(time, p.inOetf);
    _displayGamut->getValueAtTime(time, p.displayGamut);
    _eotf->getValueAtTime(time, p.eotf);
    int lookPreset = 0;
    _lookPreset->getValueAtTime(time, lookPreset);
    int cwpChoice = 0;
    _cwp->getValueAtTime(time, cwpChoice);
    _filmicProjSim->getValueAtTime(time, p.filmicProjectorSim);

    // -- Bool params (-> int) --
    bool b = false;
    _clamp->getValueAtTime(time, b);           p.clamp              = b ? 1 : 0;
    _advHueContrast->getValueAtTime(time, b);  p.advHueContrast     = b ? 1 : 0;
    _betaFeatures->getValueAtTime(time, b);    p.betaFeaturesEnable = b ? 1 : 0;
    _filmicMode->getValueAtTime(time, b);      p.filmicMode         = b ? 1 : 0;
    _tonescaleMap->getValueAtTime(time, b);    p.tonescaleMap       = b ? 1 : 0;
    _diagnosticsMode->getValueAtTime(time, b); p.diagnosticsMode    = b ? 1 : 0;
    _rgbChips->getValueAtTime(time, b);        p.rgbChipsMode       = b ? 1 : 0;

    // -- UI enable gates (hidden armed flags; the on switches gate them below) --
    _tnHconEnable->getValueAtTime(time, b);    p.tnHconUIEnable = b ? 1 : 0;
    _tnLconEnable->getValueAtTime(time, b);    p.tnLconUIEnable = b ? 1 : 0;
    _ptlEnable->getValueAtTime(time, b);       p.ptlUIEnable    = b ? 1 : 0;
    _ptmEnable->getValueAtTime(time, b);       p.ptmUIEnable    = b ? 1 : 0;
    _brlEnable->getValueAtTime(time, b);       p.brlUIEnable    = b ? 1 : 0;
    _hsRgbEnable->getValueAtTime(time, b);     p.hsRgbUIEnable  = b ? 1 : 0;
    _hsCmyEnable->getValueAtTime(time, b);     p.hsCmyUIEnable  = b ? 1 : 0;
    _hcEnable->getValueAtTime(time, b);        p.hcUIEnable     = b ? 1 : 0;

    // -- Preset enable gates (from the current look preset) --
    if (lookPreset < 0 || lookPreset > 3) lookPreset = 0;
    const LookPreset& g = kLookPresets[lookPreset];
    p.tnHconPresetEnable = g.hcon_enable;
    p.tnLconPresetEnable = g.lcon_enable;
    p.ptlPresetEnable    = g.ptl_enable;
    p.ptmPresetEnable    = g.ptm_enable;
    p.brlPresetEnable    = g.brl_enable;
    p.hsRgbPresetEnable  = g.hs_rgb_enable;
    p.hsCmyPresetEnable  = g.hs_cmy_enable;
    p.hcPresetEnable     = g.hc_enable;

    // A picked tonescale preset (1-9) owns the contrast high / low gates, as
    // in upstream OpenDRT v1.0.0; "Use Look Preset" (0) keeps the look's.
    int tonescalePreset = 0;
    _tonescalePreset->getValueAtTime(time, tonescalePreset);
    if (tonescalePreset >= 1 && tonescalePreset <= 9) {
        p.tnHconPresetEnable = kTonescalePresetFlags[tonescalePreset - 1][0];
        p.tnLconPresetEnable = kTonescalePresetFlags[tonescalePreset - 1][1];
    }

    // -- Module on switches: off bypasses both the preset and the armed gate --
    auto gate = [&](OFX::BooleanParam* on, int& presetGate, int& uiGate) {
        bool isOn = true;
        on->getValueAtTime(time, isOn);
        presetGate = isOn ? presetGate : 0;
        uiGate = (isOn && uiGate) ? 1 : 0;
    };
    gate(_tnHconOn, p.tnHconPresetEnable, p.tnHconUIEnable);
    gate(_tnLconOn, p.tnLconPresetEnable, p.tnLconUIEnable);
    gate(_ptlOn,    p.ptlPresetEnable,    p.ptlUIEnable);
    gate(_ptmOn,    p.ptmPresetEnable,    p.ptmUIEnable);
    gate(_brlOn,    p.brlPresetEnable,    p.brlUIEnable);
    gate(_hsRgbOn,  p.hsRgbPresetEnable,  p.hsRgbUIEnable);
    gate(_hsCmyOn,  p.hsCmyPresetEnable,  p.hsCmyUIEnable);
    gate(_hcOn,     p.hcPresetEnable,     p.hcUIEnable);

    // -- Creative white: "Use Look Preset" (index 4) resolves to preset cwp --
    p.cwp = (cwpChoice == 4) ? g.cwp : cwpChoice;

    // -- Double params --
    double v;
    _tnLp->getValueAtTime(time, v);      p.tnLp      = (float)v;
    _tnGb->getValueAtTime(time, v);      p.tnGb      = (float)v;
    _ptHdr->getValueAtTime(time, v);     p.ptHdr     = (float)v;
    _tnLg->getValueAtTime(time, v);      p.tnLg      = (float)v;
    _tnCon->getValueAtTime(time, v);     p.tnCon     = (float)v;
    _tnSh->getValueAtTime(time, v);      p.tnSh      = (float)v;
    _tnToe->getValueAtTime(time, v);     p.tnToe     = (float)v;
    _tnOff->getValueAtTime(time, v);     p.tnOff     = (float)v;

    _tnHcon->getValueAtTime(time, v);    p.tnHcon    = (float)v;
    _tnHconPv->getValueAtTime(time, v);  p.tnHconPv  = (float)v;
    _tnHconSt->getValueAtTime(time, v);  p.tnHconSt  = (float)v;

    _tnLcon->getValueAtTime(time, v);    p.tnLcon    = (float)v;
    _tnLconW->getValueAtTime(time, v);   p.tnLconW   = (float)v;
    _tnLconPc->getValueAtTime(time, v);  p.tnLconPc  = (float)v;

    _cwpRng->getValueAtTime(time, v);    p.cwpRng    = (float)v;
    _rsSa->getValueAtTime(time, v);      p.rsSa      = (float)v;
    _rsRw->getValueAtTime(time, v);      p.rsRw      = (float)v;
    _rsBw->getValueAtTime(time, v);      p.rsBw      = (float)v;

    _ptR->getValueAtTime(time, v);       p.ptR       = (float)v;
    _ptG->getValueAtTime(time, v);       p.ptG       = (float)v;
    _ptB->getValueAtTime(time, v);       p.ptB       = (float)v;
    _ptRngLow->getValueAtTime(time, v);  p.ptRngLow  = (float)v;
    _ptRngHigh->getValueAtTime(time, v); p.ptRngHigh = (float)v;

    _ptmLow->getValueAtTime(time, v);    p.ptmLow    = (float)v;
    _ptmLowSt->getValueAtTime(time, v);  p.ptmLowSt  = (float)v;
    _ptmHigh->getValueAtTime(time, v);   p.ptmHigh   = (float)v;
    _ptmHighSt->getValueAtTime(time, v); p.ptmHighSt = (float)v;

    _brlR->getValueAtTime(time, v);      p.brlR      = (float)v;
    _brlG->getValueAtTime(time, v);      p.brlG      = (float)v;
    _brlB->getValueAtTime(time, v);      p.brlB      = (float)v;
    _brlC->getValueAtTime(time, v);      p.brlC      = (float)v;
    _brlM->getValueAtTime(time, v);      p.brlM      = (float)v;
    _brlY->getValueAtTime(time, v);      p.brlY      = (float)v;
    _brlRng->getValueAtTime(time, v);    p.brlRng    = (float)v;

    _hsR->getValueAtTime(time, v);       p.hsR       = (float)v;
    _hsG->getValueAtTime(time, v);       p.hsG       = (float)v;
    _hsB->getValueAtTime(time, v);       p.hsB       = (float)v;
    _hsRgbRng->getValueAtTime(time, v);  p.hsRgbRng  = (float)v;

    _hsC->getValueAtTime(time, v);       p.hsC       = (float)v;
    _hsM->getValueAtTime(time, v);       p.hsM       = (float)v;
    _hsY->getValueAtTime(time, v);       p.hsY       = (float)v;

    _hcR->getValueAtTime(time, v);       p.hcR       = (float)v;

    _advHcR->getValueAtTime(time, v);    p.advHcR    = (float)v;
    _advHcG->getValueAtTime(time, v);    p.advHcG    = (float)v;
    _advHcB->getValueAtTime(time, v);    p.advHcB    = (float)v;
    _advHcC->getValueAtTime(time, v);    p.advHcC    = (float)v;
    _advHcM->getValueAtTime(time, v);    p.advHcM    = (float)v;
    _advHcY->getValueAtTime(time, v);    p.advHcY    = (float)v;
    _advHcPower->getValueAtTime(time, v);p.advHcPower= (float)v;

    _filmicSrcStops->getValueAtTime(time, v); p.filmicSourceStops  = (float)v;
    _filmicTgtStops->getValueAtTime(time, v); p.filmicTargetStops  = (float)v;
    _filmicDynRange->getValueAtTime(time, v); p.filmicDynamicRange = (float)v;
    _filmicStrength->getValueAtTime(time, v); p.filmicStrength     = (float)v;

    // -- Hue Anchor Compression: not exposed in this build (no UI param).
    //    Left disabled (zeroed by the {} init), so the kernel skips that block.
    p.hueCompressionEnable = 0;
    // Current = 0; v1.0.1 = 1 (upstream purity softclip fix). v1.1 never
    // reaches here (render dispatches it to its own kernel).
    p.mathVariant = (modelAt(time) == ODT_MODEL_V101) ? 1 : 0;

    // -- Per-frame precompute: display-encoding remap + ts_* tonescale constants.
    //    Mutates p.displayGamut/p.eotf to the canonical pair the kernel reads. --
    opendrt_precompute(&p);

    return p;
}

bool OpenDRTPlugin::isIdentity(const OFX::IsIdentityArguments& /*args*/,
                               OFX::Clip*& /*identityClip*/, double& /*identityTime*/) {
    // OpenDRT is a display transform - never an identity (it always remaps).
    return false;
}

void OpenDRTPlugin::render(const OFX::RenderArguments& args) {
    if (_control) _control->onRender();

    std::unique_ptr<OFX::Image> srcImg(_srcClip->fetchImage(args.time));
    std::unique_ptr<OFX::Image> dstImg(_dstClip->fetchImage(args.time));
    if (!dstImg.get()) return;

    OfxRectI bounds = dstImg->getBounds();
    int width  = bounds.x2 - bounds.x1;
    int height = bounds.y2 - bounds.y1;

    int srcRowBytes = srcImg.get() ? srcImg->getRowBytes()
                                    : width * 4 * (int)sizeof(float);
    int dstRowBytes = dstImg->getRowBytes();

    // -- v1.1 model: own params and kernels, same buffer conventions --
    if (modelAt(args.time) == ODT_MODEL_V11) {
        const OpenDRTv11Params v11 = buildV11Params(args.time);
#ifdef LDT_HAS_CUDA
        if (args.pCudaStream && srcImg.get()) {
            const float* src = static_cast<const float*>(srcImg->getPixelAddress(bounds.x1, bounds.y1));
            float*       dst = static_cast<float*>(dstImg->getPixelAddress(bounds.x1, bounds.y1));
            RunOpenDRTv11CudaKernel(args.pCudaStream, width, height,
                                    srcRowBytes, dstRowBytes, &v11, src, dst);
            return;
        }
#endif
#ifdef LDT_HAS_OPENCL
        if (args.pOpenCLCmdQ && srcImg.get()) {
            const OfxRectI sb = srcImg->getBounds();
            RunOpenDRTv11OpenCLKernel(args.pOpenCLCmdQ, width, height,
                                      srcRowBytes, dstRowBytes, &v11,
                                      static_cast<const float*>(srcImg->getPixelData()),
                                      static_cast<float*>(dstImg->getPixelData()),
                                      bounds.x1 - sb.x1, bounds.y1 - sb.y1,
                                      sb.x2 - sb.x1, sb.y2 - sb.y1);
            return;
        }
        if (args.pOpenCLCmdQ) {
            RunOpenDRTv11OpenCLKernel(args.pOpenCLCmdQ, width, height,
                                      srcRowBytes, dstRowBytes, &v11,
                                      nullptr, static_cast<float*>(dstImg->getPixelData()));
            return;
        }
#endif
#ifdef __APPLE__
        if (args.pMetalCmdQ && srcImg.get()) {
            const float* src = static_cast<const float*>(srcImg->getPixelAddress(bounds.x1, bounds.y1));
            float*       dst = static_cast<float*>(dstImg->getPixelAddress(bounds.x1, bounds.y1));
            RunOpenDRTv11MetalKernel(args.pMetalCmdQ, width, height,
                                     srcRowBytes, dstRowBytes, &v11, src, dst);
            return;
        }
#endif
        OpenDRTv11Processor processor(*this);
        processor.setDstImg(dstImg.get());
        processor.setSrcImg(srcImg.get());
        processor.setRenderWindow(args.renderWindow);
        processor.setParams(v11);
        processor.process();
        return;
    }

    // -- Tesseract model: own params and kernels, same buffer conventions --
    if (modelAt(args.time) == ODT_MODEL_TESSERACT) {
        const TesseractParams mp = buildTessParams(args.time);
#ifdef LDT_HAS_CUDA
        if (args.pCudaStream && srcImg.get()) {
            const float* src = static_cast<const float*>(srcImg->getPixelAddress(bounds.x1, bounds.y1));
            float*       dst = static_cast<float*>(dstImg->getPixelAddress(bounds.x1, bounds.y1));
            RunTesseractCudaKernel(args.pCudaStream, width, height,
                               srcRowBytes, dstRowBytes, &mp, src, dst);
            return;
        }
#endif
#ifdef LDT_HAS_OPENCL
        if (args.pOpenCLCmdQ && srcImg.get()) {
            const OfxRectI sb = srcImg->getBounds();
            RunTesseractOpenCLKernel(args.pOpenCLCmdQ, width, height,
                                 srcRowBytes, dstRowBytes, &mp,
                                 static_cast<const float*>(srcImg->getPixelData()),
                                 static_cast<float*>(dstImg->getPixelData()),
                                 bounds.x1 - sb.x1, bounds.y1 - sb.y1,
                                 sb.x2 - sb.x1, sb.y2 - sb.y1);
            return;
        }
        if (args.pOpenCLCmdQ) {
            RunTesseractOpenCLKernel(args.pOpenCLCmdQ, width, height,
                                 srcRowBytes, dstRowBytes, &mp,
                                 nullptr, static_cast<float*>(dstImg->getPixelData()));
            return;
        }
#endif
#ifdef __APPLE__
        if (args.pMetalCmdQ && srcImg.get()) {
            const float* src = static_cast<const float*>(srcImg->getPixelAddress(bounds.x1, bounds.y1));
            float*       dst = static_cast<float*>(dstImg->getPixelAddress(bounds.x1, bounds.y1));
            RunTesseractMetalKernel(args.pMetalCmdQ, width, height,
                                srcRowBytes, dstRowBytes, &mp, src, dst);
            return;
        }
#endif
        TesseractProcessor processor(*this);
        processor.setDstImg(dstImg.get());
        processor.setSrcImg(srcImg.get());
        processor.setRenderWindow(args.renderWindow);
        processor.setParams(mp);
        processor.process();
        return;
    }

    // -- Chromagnon model: own params and kernels, same buffer conventions --
    if (modelAt(args.time) == ODT_MODEL_CHROMAGNON) {
        const ChromagnonParams mp = buildCmgParams(args.time);
#ifdef LDT_HAS_CUDA
        if (args.pCudaStream && srcImg.get()) {
            const float* src = static_cast<const float*>(srcImg->getPixelAddress(bounds.x1, bounds.y1));
            float*       dst = static_cast<float*>(dstImg->getPixelAddress(bounds.x1, bounds.y1));
            RunChromagnonCudaKernel(args.pCudaStream, width, height,
                               srcRowBytes, dstRowBytes, &mp, src, dst);
            return;
        }
#endif
#ifdef LDT_HAS_OPENCL
        if (args.pOpenCLCmdQ && srcImg.get()) {
            const OfxRectI sb = srcImg->getBounds();
            RunChromagnonOpenCLKernel(args.pOpenCLCmdQ, width, height,
                                 srcRowBytes, dstRowBytes, &mp,
                                 static_cast<const float*>(srcImg->getPixelData()),
                                 static_cast<float*>(dstImg->getPixelData()),
                                 bounds.x1 - sb.x1, bounds.y1 - sb.y1,
                                 sb.x2 - sb.x1, sb.y2 - sb.y1);
            return;
        }
        if (args.pOpenCLCmdQ) {
            RunChromagnonOpenCLKernel(args.pOpenCLCmdQ, width, height,
                                 srcRowBytes, dstRowBytes, &mp,
                                 nullptr, static_cast<float*>(dstImg->getPixelData()));
            return;
        }
#endif
#ifdef __APPLE__
        if (args.pMetalCmdQ && srcImg.get()) {
            const float* src = static_cast<const float*>(srcImg->getPixelAddress(bounds.x1, bounds.y1));
            float*       dst = static_cast<float*>(dstImg->getPixelAddress(bounds.x1, bounds.y1));
            RunChromagnonMetalKernel(args.pMetalCmdQ, width, height,
                                srcRowBytes, dstRowBytes, &mp, src, dst);
            return;
        }
#endif
        ChromagnonProcessor processor(*this);
        processor.setDstImg(dstImg.get());
        processor.setSrcImg(srcImg.get());
        processor.setRenderWindow(args.renderWindow);
        processor.setParams(mp);
        processor.process();
        return;
    }

    OpenDRTParams params = buildParams(args.time, width, height);

    // -- CUDA path -------------------------------------------------------------
#ifdef LDT_HAS_CUDA
    if (args.pCudaStream && srcImg.get()) {
        const float* src = static_cast<const float*>(srcImg->getPixelAddress(bounds.x1, bounds.y1));
        float*       dst = static_cast<float*>(dstImg->getPixelAddress(bounds.x1, bounds.y1));
        RunOpenDRTCudaKernel(args.pCudaStream, width, height,
                             srcRowBytes, dstRowBytes, &params, src, dst);
        return;
    }
#endif

    // -- OpenCL path -----------------------------------------------------------
    // The host hands over cl_mem handles: pass getPixelData() of both images
    // (never offset, never touched on the host) and say where the render rect
    // sits inside the source.
#ifdef LDT_HAS_OPENCL
    if (args.pOpenCLCmdQ && srcImg.get()) {
        const OfxRectI sb = srcImg->getBounds();
        RunOpenDRTOpenCLKernel(args.pOpenCLCmdQ, width, height,
                               srcRowBytes, dstRowBytes, &params,
                               static_cast<const float*>(srcImg->getPixelData()),
                               static_cast<float*>(dstImg->getPixelData()),
                               bounds.x1 - sb.x1, bounds.y1 - sb.y1,
                               sb.x2 - sb.x1, sb.y2 - sb.y1);
        return;
    }
    if (args.pOpenCLCmdQ) {
        // No source: the CPU processor below would dereference a cl_mem
        // handle, so the OpenCL host renders the (0, 0, 0, 1) input itself.
        RunOpenDRTOpenCLKernel(args.pOpenCLCmdQ, width, height,
                               srcRowBytes, dstRowBytes, &params,
                               nullptr, static_cast<float*>(dstImg->getPixelData()));
        return;
    }
#endif

    // -- Metal path ----------------------------------------------------------
#ifdef __APPLE__
    if (args.pMetalCmdQ && srcImg.get()) {
        const float* src = static_cast<const float*>(srcImg->getPixelAddress(bounds.x1, bounds.y1));
        float*       dst = static_cast<float*>(dstImg->getPixelAddress(bounds.x1, bounds.y1));
        RunOpenDRTMetalKernel(args.pMetalCmdQ, width, height,
                              srcRowBytes, dstRowBytes, &params, src, dst);
        return;
    }
#endif

    // -- CPU path --------------------------------------------------------------
    OpenDRTProcessor processor(*this);
    processor.setDstImg(dstImg.get());
    processor.setSrcImg(srcImg.get());
    processor.setRenderWindow(args.renderWindow);
    processor.setParams(params);
    processor.process();
}

// -- Factory -------------------------------------------------------------------
void OpenDRTFactory::describe(OFX::ImageEffectDescriptor& desc) {
    desc.setLabels(OPENDRT_PLUGIN_LABEL, OPENDRT_PLUGIN_LABEL, OPENDRT_PLUGIN_LABEL);
    desc.setPluginGrouping(OPENDRT_PLUGIN_GROUP);
    desc.setPluginDescription(kPluginDescription);
    desc.addSupportedContext(OFX::eContextFilter);
    desc.addSupportedContext(OFX::eContextGeneral);
    desc.addSupportedBitDepth(OFX::eBitDepthFloat);
    desc.setSupportsTiles(kSupportsTiles);
    desc.setSupportsMultiResolution(kSupportsMultiResolution);
    desc.setRenderThreadSafety(OFX::eRenderFullySafe);
#ifdef LDT_HAS_CUDA
    desc.setSupportsCudaRender(true);
#endif
#ifdef LDT_HAS_OPENCL
    desc.setSupportsOpenCLBuffersRender(true);
#endif
#ifdef __APPLE__
    desc.setSupportsMetalRender(true);
#endif
}

void OpenDRTFactory::describeInContext(OFX::ImageEffectDescriptor& desc,
                                       OFX::ContextEnum /*ctx*/) {
    using namespace OFX;

    ClipDescriptor* srcClip = desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    ClipDescriptor* dstClip = desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    PageParamDescriptor* page = desc.definePageParam("Controls");

    // -- Input & Output Settings group. Defined before Model so Model can be
    //    its first control. --
    GroupParamDescriptor* inputGroup        = desc.defineGroupParam("InputGroup");
    inputGroup->setLabels("Input & Output Settings", "Input & Output Settings", "Input & Output Settings");
    inputGroup->setOpen(false);

    // -- Model selector (first control in Input & Output Settings). Old
    //    projects have no value saved and load the default 0, the Current
    //    model, unchanged. It stays defined before every value param it
    //    governs. --
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kODT_Model);
        p->setLabels("Model", "Model", "Model");
        p->setScriptName(kODT_Model);
        p->setHint("Current: the original model existing grades were made on. "
                   "v1.0.1: same controls plus the upstream purity softclip fix. "
                   "v1.1: the newer upstream model with its own controls and defaults. "
                   "Tesseract and Chromagnon: alternate display transforms, each with "
                   "its own controls (Input Gamut is shared).");
        p->appendOption("Current");
        p->appendOption("v1.0.1");
        p->appendOption("v1.1");
        p->appendOption("Tesseract");
        p->appendOption("Chromagnon");
        p->setDefault(ODT_MODEL_CURRENT);
        p->setAnimates(false);
        p->setParent(*inputGroup);
        page->addChild(*p);
    }

    // -- Collapsible groups, defined in UI order. Stickshift Mode is a master
    //    group; the nine module groups are its sub-groups, each opening with
    //    its own Enable checkbox and always shown on the Current UI. --
    GroupParamDescriptor* presetGroup       = desc.defineGroupParam("PresetGroup");
    presetGroup->setLabels("Automatic Presets", "Automatic Presets", "Automatic Presets");
    GroupParamDescriptor* tonescaleGroup    = desc.defineGroupParam("TonescaleGroup");
    tonescaleGroup->setLabels("Tonescale", "Tonescale", "Tonescale");
    tonescaleGroup->setOpen(false);
    GroupParamDescriptor* contrastGroup     = desc.defineGroupParam("ContrastGroup");
    contrastGroup->setLabels("Basic Contrast", "Basic Contrast", "Basic Contrast");
    contrastGroup->setOpen(false);
    GroupParamDescriptor* colorGroup        = desc.defineGroupParam("ColorGroup");
    colorGroup->setLabels("Creative White Point", "Creative White Point", "Creative White Point");
    colorGroup->setOpen(false);
    GroupParamDescriptor* stickshiftGroup   = desc.defineGroupParam("StickshiftGroup");
    stickshiftGroup->setLabels("Stickshift Mode", "Stickshift Mode", "Stickshift Mode");
    stickshiftGroup->setOpen(false);
    GroupParamDescriptor* highContrastGroup = desc.defineGroupParam("HighContrastGroup");
    highContrastGroup->setLabels("High Contrast", "High Contrast", "High Contrast");
    highContrastGroup->setOpen(false);
    highContrastGroup->setParent(*stickshiftGroup);
    GroupParamDescriptor* lowContrastGroup  = desc.defineGroupParam("LowContrastGroup");
    lowContrastGroup->setLabels("Low Contrast", "Low Contrast", "Low Contrast");
    lowContrastGroup->setOpen(false);
    lowContrastGroup->setParent(*stickshiftGroup);
    GroupParamDescriptor* purityLowGroup    = desc.defineGroupParam("PurityLowGroup");
    purityLowGroup->setLabels("Purity Compress Low", "Purity Compress Low", "Purity Compress Low");
    purityLowGroup->setOpen(false);
    purityLowGroup->setParent(*stickshiftGroup);
    GroupParamDescriptor* midPurityGroup    = desc.defineGroupParam("MidPurityGroup");
    midPurityGroup->setLabels("Mid Purity", "Mid Purity", "Mid Purity");
    midPurityGroup->setOpen(false);
    midPurityGroup->setParent(*stickshiftGroup);
    GroupParamDescriptor* brillianceGroup   = desc.defineGroupParam("BrillianceGroup");
    brillianceGroup->setLabels("Brilliance", "Brilliance", "Brilliance");
    brillianceGroup->setOpen(false);
    brillianceGroup->setParent(*stickshiftGroup);
    GroupParamDescriptor* hueshiftRgbGroup  = desc.defineGroupParam("HueshiftRgbGroup");
    hueshiftRgbGroup->setLabels("Hueshift RGB", "Hueshift RGB", "Hueshift RGB");
    hueshiftRgbGroup->setOpen(false);
    hueshiftRgbGroup->setParent(*stickshiftGroup);
    GroupParamDescriptor* hueshiftCmyGroup  = desc.defineGroupParam("HueshiftCmyGroup");
    hueshiftCmyGroup->setLabels("Hueshift CMY", "Hueshift CMY", "Hueshift CMY");
    hueshiftCmyGroup->setOpen(false);
    hueshiftCmyGroup->setParent(*stickshiftGroup);
    GroupParamDescriptor* hueContrastGroup  = desc.defineGroupParam("HueContrastGroup");
    hueContrastGroup->setLabels("Hue Contrast", "Hue Contrast", "Hue Contrast");
    hueContrastGroup->setOpen(false);
    hueContrastGroup->setParent(*stickshiftGroup);
    GroupParamDescriptor* advHueContrastGroup = desc.defineGroupParam("AdvHueContrastGroup");
    advHueContrastGroup->setLabels("Advanced Hue Contrast", "Advanced Hue Contrast", "Advanced Hue Contrast");
    advHueContrastGroup->setOpen(false);
    advHueContrastGroup->setParent(*stickshiftGroup);
    GroupParamDescriptor* betaFeaturesGroup = desc.defineGroupParam("BetaFeaturesGroup");
    betaFeaturesGroup->setLabels("Beta Features", "Beta Features", "Beta Features");
    betaFeaturesGroup->setOpen(false);
    GroupParamDescriptor* diagnosticsGroup  = desc.defineGroupParam("DiagnosticsGroup");
    diagnosticsGroup->setLabels("Diagnostics", "Diagnostics", "Diagnostics");
    diagnosticsGroup->setOpen(false);
    GroupParamDescriptor* filmicDynamicRangeGroup = desc.defineGroupParam("FilmicDynamicRangeGroup");
    filmicDynamicRangeGroup->setLabels("Filmic Dynamic Range Beta", "Filmic Dynamic Range Beta", "Filmic Dynamic Range Beta");
    filmicDynamicRangeGroup->setOpen(false);
    GroupParamDescriptor* filmicProjectorSimGroup = desc.defineGroupParam("FilmicProjectorSimGroup");
    filmicProjectorSimGroup->setLabels("Filmic Projector Sim Beta", "Filmic Projector Sim Beta", "Filmic Projector Sim Beta");
    filmicProjectorSimGroup->setOpen(false);

    // v1.1 model groups (slot 0 is the shared InputGroup above).
    GroupParamDescriptor* v11Groups[ODT11_G_COUNT] = {};
    v11Groups[ODT11_G_INPUT] = inputGroup;
    for (int g = 1; g < ODT11_G_COUNT; ++g) {
        const Odt11GroupInfo& gi = kOdt11Groups[g];
        v11Groups[g] = desc.defineGroupParam(gi.name);
        v11Groups[g]->setLabels(gi.label, gi.label, gi.label);
        v11Groups[g]->setOpen(gi.open);
    }

    page->addChild(*inputGroup);
    page->addChild(*presetGroup);
    page->addChild(*tonescaleGroup);
    page->addChild(*contrastGroup);
    page->addChild(*colorGroup);
    page->addChild(*stickshiftGroup);
    page->addChild(*highContrastGroup);
    page->addChild(*lowContrastGroup);
    page->addChild(*purityLowGroup);
    page->addChild(*midPurityGroup);
    page->addChild(*brillianceGroup);
    page->addChild(*hueshiftRgbGroup);
    page->addChild(*hueshiftCmyGroup);
    page->addChild(*hueContrastGroup);
    page->addChild(*advHueContrastGroup);
    page->addChild(*betaFeaturesGroup);
    page->addChild(*diagnosticsGroup);
    page->addChild(*filmicDynamicRangeGroup);
    page->addChild(*filmicProjectorSimGroup);
    for (int g = 1; g < ODT11_G_COUNT; ++g) page->addChild(*v11Groups[g]);

    // -- Input gamut (16 options; all fully matrixed via odt_getInputMatrix) --
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kODT_InGamut);
        p->setLabel("Input Gamut");
        p->setHint("Input color gamut/primaries. All 16 options (XYZ / ACES2065-1 / "
                   "ACEScg / P3D65 / Rec.2020 / Rec.709 plus the camera wide gamuts) "
                   "use their exact input-to-XYZ matrix, matching canonical OpenDRT.");
        p->appendOption("XYZ");
        p->appendOption("ACES 2065-1");
        p->appendOption("ACEScg");
        p->appendOption("P3D65");
        p->appendOption("Rec.2020");
        p->appendOption("Rec.709");
        p->appendOption("Arri Wide Gamut 3");
        p->appendOption("Arri Wide Gamut 4");
        p->appendOption("Red Wide Gamut RGB");
        p->appendOption("Sony SGamut3");
        p->appendOption("Sony SGamut3Cine");
        p->appendOption("Panasonic V-Gamut");
        p->appendOption("Blackmagic Wide Gamut");
        p->appendOption("Filmlight E-Gamut");
        p->appendOption("Filmlight E-Gamut2");
        p->appendOption("DaVinci Wide Gamut");
        p->setDefault(15);
        p->setAnimates(true);
        p->setParent(*inputGroup);
        page->addChild(*p);
    }
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kODT_InOetf);
        p->setLabel("Input Transfer Function");
        p->setHint("Input transfer function (OETF). Defaults to Linear to match the "
                   "Linear-default convention of the other Look Dev Tools.");
        p->appendOption("Linear");
        p->appendOption("Davinci Intermediate");
        p->appendOption("Filmlight T-Log");
        p->appendOption("ACEScct");
        p->appendOption("Arri LogC3");
        p->appendOption("Arri LogC4");
        p->appendOption("RedLog3G10");
        p->appendOption("Panasonic V-Log");
        p->appendOption("Sony S-Log3");
        p->setDefault(0);   // Linear - suite-wide default (was 1 = DaVinci Intermediate)
        p->setAnimates(true);
        p->setParent(*inputGroup);
        page->addChild(*p);
    }
    // Display gamut + EOTF live in the Input group (governed-before pattern: the
    // host's display-encoding remap reads both, but as plain choices order is moot).
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kODT_DisplayGamut);
        p->setLabel("Display Gamut");
        p->setHint("Output display gamut");
        p->appendOption("Rec.709");
        p->appendOption("P3-D65");
        p->appendOption("Rec.2020 (P3 Limited)");
        p->setDefault(0);
        p->setAnimates(true);
        p->setParent(*inputGroup);
        page->addChild(*p);
    }
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kODT_Eotf);
        p->setLabel("Display EOTF");
        p->setHint("Output display transfer function (EOTF)");
        p->appendOption("Linear");
        p->appendOption("2.2 Power sRGB Display");
        p->appendOption("2.4 Power Rec.1886");
        p->appendOption("2.6 Power DCI");
        p->appendOption("ST 2084 PQ");
        p->appendOption("HLG");
        p->setDefault(2);
        p->setAnimates(true);
        p->setParent(*inputGroup);
        page->addChild(*p);
    }

    // -- Look preset (governs PresetEnable gates + cwp "Use Look Preset"). Defined
    //    BEFORE the modules/cwp it influences, per the OFX restore-order gotcha. --
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kODT_LookPreset);
        p->setLabel("Look Preset");
        p->setHint("Selects which transform modules are enabled by default "
                   "(High/Low Contrast, Purity, Brilliance, Hueshift, Hue Contrast) "
                   "and the creative white when 'Use Look Preset' is chosen.");
        p->appendOption("Default");
        p->appendOption("Colorful");
        p->appendOption("Umbra");
        p->appendOption("Base");
        p->setDefault(0);
        p->setAnimates(true);
        p->setParent(*presetGroup);
        page->addChild(*p);
    }
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kODT_TonescalePreset);
        p->setLabel("Tonescale Preset");
        p->setHint("Select a tonescale preset for quick adjustment (overrides the "
                   "look preset's tonescale when set to anything but 'Use Look Preset').");
        p->appendOption("Use Look Preset");      // 0
        p->appendOption("High-Contrast");        // 1
        p->appendOption("Low-Contrast");         // 2
        p->appendOption("ACES-1.x");             // 3
        p->appendOption("ACES-2.0");             // 4
        p->appendOption("Marvelous Tonescape");  // 5
        p->appendOption("Arriba Tonecall");      // 6
        p->appendOption("DaGrinchi Tonegroan");  // 7
        p->appendOption("Aery Tonescale");       // 8
        p->appendOption("Umbra Tonescale");      // 9
        p->setDefault(0);
        p->setAnimates(true);
        p->setParent(*presetGroup);
        page->addChild(*p);
    }
    defineBool(desc, kODT_LockStickshift, "Lock Manual Adjustments",
               "When enabled, prevents presets from overriding ANY manual stickshift "
               "adjustments (master lock over all per-module Persistent Values).",
               false, presetGroup, page);

    // -- Tonescale --
    defineDouble(desc, kODT_TnLp,  "Display Peak Luminance", "Peak luminance of the display in nits", 100.0, 100.0, 1000.0, 1.0, tonescaleGroup, page);
    defineDouble(desc, kODT_TnGb,  "HDR Grey Boost",         "Boost grey levels for HDR displays",     0.13,  0.0,   1.0,    0.01, tonescaleGroup, page);
    defineDouble(desc, kODT_PtHdr, "HDR Purity",             "Purity adjustment for HDR",              0.5,   0.0,   1.0,    0.01, tonescaleGroup, page);

    // -- Basic contrast --
    defineBool(desc, kODT_Clamp, "Clamp", "Enable tone curve clamping", true, contrastGroup, page);
    defineDouble(desc, kODT_TnLg,  "Grey Luminance", "Grey point luminance",          11.1,  4.0, 25.0, 0.1,   contrastGroup, page);
    defineDouble(desc, kODT_TnCon, "Contrast",       "Overall contrast adjustment",   1.4,   1.0, 2.0,  0.01,  contrastGroup, page);
    defineDouble(desc, kODT_TnSh,  "Shoulder Clip",  "Highlight shoulder clipping",   0.5,   0.0, 1.0,  0.01,  contrastGroup, page);
    defineDouble(desc, kODT_TnToe, "Toe",            "Shadow toe adjustment",         0.003, 0.0, 0.1,  0.001, contrastGroup, page);
    defineDouble(desc, kODT_TnOff, "Offset",         "Black point offset",            0.005, 0.0, 0.02, 0.001, contrastGroup, page);

    // -- High contrast --
    defineBool(desc, kODT_TnHconOn, "Enable Contrast High", "Quick on/off for this module. Untick to bypass it while you compare.", true, highContrastGroup, page);
    // Hidden armed flag: set by editing a slider, cleared by a preset.
    defineBool(desc, kODT_TnHconEnable, "Contrast High Armed (legacy)", "Internal. Set when a slider here is edited, cleared when a preset is applied.", false, highContrastGroup, page)->setIsSecret(true);
    defineBool(desc, kODT_LockHcon, "Lock (Persistent Values)", "Keep these values when switching presets; still manually adjustable.", false, highContrastGroup, page);
    defineDouble(desc, kODT_TnHcon,   "Contrast High",          "High frequency contrast",       0.0, -1.0, 1.0, 0.01, highContrastGroup, page);
    defineDouble(desc, kODT_TnHconPv, "Contrast High Pivot",    "Pivot point for high contrast", 1.0,  0.0, 4.0, 0.01, highContrastGroup, page);
    defineDouble(desc, kODT_TnHconSt, "Contrast High Strength", "Strength of high contrast",     4.0,  0.0, 4.0, 0.01, highContrastGroup, page);

    // -- Low contrast --
    defineBool(desc, kODT_TnLconOn, "Enable Contrast Low", "Quick on/off for this module. Untick to bypass it while you compare.", true, lowContrastGroup, page);
    // Hidden armed flag: set by editing a slider, cleared by a preset.
    defineBool(desc, kODT_TnLconEnable, "Contrast Low Armed (legacy)", "Internal. Set when a slider here is edited, cleared when a preset is applied.", false, lowContrastGroup, page)->setIsSecret(true);
    defineBool(desc, kODT_LockLcon, "Lock (Persistent Values)", "Keep these values when switching presets; still manually adjustable.", false, lowContrastGroup, page);
    defineDouble(desc, kODT_TnLcon,   "Contrast Low",            "Low frequency contrast",   1.0, 0.0, 3.0, 0.001, lowContrastGroup, page);
    defineDouble(desc, kODT_TnLconW,  "Contrast Low Width",      "Width of low contrast",    0.5, 0.0, 2.0, 0.001, lowContrastGroup, page);
    defineDouble(desc, kODT_TnLconPc, "Contrast Low Per-Channel","Per-channel low contrast", 1.0, 0.0, 1.0, 0.001, lowContrastGroup, page);

    // -- Creative white + render space --
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kODT_Cwp);
        p->setLabel("Creative White");
        p->setHint("Creative white point selection");
        p->appendOption("D65");
        p->appendOption("D60");
        p->appendOption("D55");
        p->appendOption("D50");
        p->appendOption("Use Look Preset");
        p->setDefault(4);
        p->setAnimates(true);
        p->setParent(*colorGroup);
        page->addChild(*p);
    }
    defineDouble(desc, kODT_CwpRng, "Creative White Range",     "Range of creative white effect", 0.5,  0.0, 1.0, 0.001, colorGroup, page);
    defineDouble(desc, kODT_RsSa,   "Render Space Strength",    "Strength of render space adj",   0.35, 0.0, 0.6, 0.001, colorGroup, page);
    defineDouble(desc, kODT_RsRw,   "Render Space Red Weight",  "Red channel weight",             0.25, 0.0, 0.8, 0.001, colorGroup, page);
    defineDouble(desc, kODT_RsBw,   "Render Space Blue Weight", "Blue channel weight",            0.55, 0.0, 0.8, 0.001, colorGroup, page);

    // -- Purity compress low --
    defineBool(desc, kODT_PtlOn, "Enable Purity Low", "Quick on/off for this module. Untick to bypass it while you compare.", true, purityLowGroup, page);
    // Hidden armed flag: set by editing a slider, cleared by a preset.
    defineBool(desc, kODT_PtlEnable, "Purity Low Armed (legacy)", "Internal. Set when a slider here is edited, cleared when a preset is applied.", false, purityLowGroup, page)->setIsSecret(true);
    defineBool(desc, kODT_LockPtl, "Lock (Persistent Values)", "Keep these values when switching presets; still manually adjustable.", false, purityLowGroup, page);
    defineDouble(desc, kODT_PtR,        "Purity Compress R", "Red purity compression",            0.5, 0.0, 4.0, 0.001, purityLowGroup, page);
    defineDouble(desc, kODT_PtG,        "Purity Compress G", "Green purity compression",          2.0, 0.0, 4.0, 0.001, purityLowGroup, page);
    defineDouble(desc, kODT_PtB,        "Purity Compress B", "Blue purity compression",           2.0, 0.0, 4.0, 0.001, purityLowGroup, page);
    defineDouble(desc, kODT_PtRngLow,   "Purity Range Low",  "Low range for purity compression",  0.2, 0.1, 0.6, 0.001, purityLowGroup, page);
    defineDouble(desc, kODT_PtRngHigh,  "Purity Range High", "High range for purity compression", 0.8, 0.25,2.0, 0.001, purityLowGroup, page);

    // -- Mid purity --
    defineBool(desc, kODT_PtmOn, "Enable Mid Purity", "Quick on/off for this module. Untick to bypass it while you compare.", true, midPurityGroup, page);
    // Hidden armed flag: set by editing a slider, cleared by a preset.
    defineBool(desc, kODT_PtmEnable, "Mid Purity Armed (legacy)", "Internal. Set when a slider here is edited, cleared when a preset is applied.", false, midPurityGroup, page)->setIsSecret(true);
    defineBool(desc, kODT_LockPtm, "Lock (Persistent Values)", "Keep these values when switching presets; still manually adjustable.", false, midPurityGroup, page);
    defineDouble(desc, kODT_PtmLow,    "Mid Purity Low",          "Mid purity low adjustment",   0.2,  0.0,  1.0, 0.001, midPurityGroup, page);
    defineDouble(desc, kODT_PtmLowSt,  "Mid Purity Low Strength", "Strength of mid purity low",  0.5,  0.1,  1.0, 0.001, midPurityGroup, page);
    defineDouble(desc, kODT_PtmHigh,   "Mid Purity High",         "Mid purity high adjustment", -0.8, -0.9,  0.0, 0.001, midPurityGroup, page);
    defineDouble(desc, kODT_PtmHighSt, "Mid Purity High Strength","Strength of mid purity high", 0.3,  0.2,  1.0, 0.001, midPurityGroup, page);

    // -- Brilliance --
    defineBool(desc, kODT_BrlOn, "Enable Brilliance", "Quick on/off for this module. Untick to bypass it while you compare.", true, brillianceGroup, page);
    // Hidden armed flag: set by editing a slider, cleared by a preset.
    defineBool(desc, kODT_BrlEnable, "Brilliance Armed (legacy)", "Internal. Set when a slider here is edited, cleared when a preset is applied.", false, brillianceGroup, page)->setIsSecret(true);
    defineBool(desc, kODT_LockBrl, "Lock (Persistent Values)", "Keep these values when switching presets; still manually adjustable.", false, brillianceGroup, page);
    defineDouble(desc, kODT_BrlR,   "Brilliance R",     "Red brilliance",     -0.5, -1.0, 1.0, 0.001, brillianceGroup, page);
    defineDouble(desc, kODT_BrlG,   "Brilliance G",     "Green brilliance",   -0.4, -1.0, 1.0, 0.001, brillianceGroup, page);
    defineDouble(desc, kODT_BrlB,   "Brilliance B",     "Blue brilliance",    -0.2, -1.0, 1.0, 0.001, brillianceGroup, page);
    defineDouble(desc, kODT_BrlC,   "Brilliance C",     "Cyan brilliance",     0.0, -1.0, 1.0, 0.001, brillianceGroup, page);
    defineDouble(desc, kODT_BrlM,   "Brilliance M",     "Magenta brilliance",  0.0, -1.0, 1.0, 0.001, brillianceGroup, page);
    defineDouble(desc, kODT_BrlY,   "Brilliance Y",     "Yellow brilliance",   0.0, -1.0, 1.0, 0.001, brillianceGroup, page);
    defineDouble(desc, kODT_BrlRng, "Brilliance Range", "Range of brilliance", 0.66, 0.0, 2.0, 0.001, brillianceGroup, page);

    // -- Hueshift RGB --
    defineBool(desc, kODT_HsRgbOn, "Enable Hueshift RGB", "Quick on/off for this module. Untick to bypass it while you compare.", true, hueshiftRgbGroup, page);
    // Hidden armed flag: set by editing a slider, cleared by a preset.
    defineBool(desc, kODT_HsRgbEnable, "Hueshift RGB Armed (legacy)", "Internal. Set when a slider here is edited, cleared when a preset is applied.", false, hueshiftRgbGroup, page)->setIsSecret(true);
    defineBool(desc, kODT_LockHsRgb, "Lock (Persistent Values)", "Keep these values when switching presets; still manually adjustable.", false, hueshiftRgbGroup, page);
    defineDouble(desc, kODT_HsR,      "Hueshift R",         "Red hue shifting",                0.35, -1.0, 1.0, 0.001, hueshiftRgbGroup, page);
    defineDouble(desc, kODT_HsG,      "Hueshift G",         "Green hue shifting",              0.25, -1.0, 1.0, 0.001, hueshiftRgbGroup, page);
    defineDouble(desc, kODT_HsB,      "Hueshift B",         "Blue hue shifting",               0.5,  -1.0, 1.0, 0.001, hueshiftRgbGroup, page);
    defineDouble(desc, kODT_HsRgbRng, "Hueshift RGB Range", "Range of RGB hue shifting",       0.6,   0.0, 2.0, 0.001, hueshiftRgbGroup, page);

    // -- Hueshift CMY --
    defineBool(desc, kODT_HsCmyOn, "Enable Hueshift CMY", "Quick on/off for this module. Untick to bypass it while you compare.", true, hueshiftCmyGroup, page);
    // Hidden armed flag: set by editing a slider, cleared by a preset.
    defineBool(desc, kODT_HsCmyEnable, "Hueshift CMY Armed (legacy)", "Internal. Set when a slider here is edited, cleared when a preset is applied.", false, hueshiftCmyGroup, page)->setIsSecret(true);
    defineBool(desc, kODT_LockHsCmy, "Lock (Persistent Values)", "Keep these values when switching presets; still manually adjustable.", false, hueshiftCmyGroup, page);
    defineDouble(desc, kODT_HsC, "Hueshift C", "Cyan hue shifting",    0.2, -1.0, 1.0, 0.001, hueshiftCmyGroup, page);
    defineDouble(desc, kODT_HsM, "Hueshift M", "Magenta hue shifting", 0.2, -1.0, 1.0, 0.001, hueshiftCmyGroup, page);
    defineDouble(desc, kODT_HsY, "Hueshift Y", "Yellow hue shifting",  0.2, -1.0, 1.0, 0.001, hueshiftCmyGroup, page);

    // -- Hue contrast --
    defineBool(desc, kODT_HcOn, "Enable Hue Contrast", "Quick on/off for this module. Untick to bypass it while you compare.", true, hueContrastGroup, page);
    // Hidden armed flag: set by editing a slider, cleared by a preset.
    defineBool(desc, kODT_HcEnable, "Hue Contrast Armed (legacy)", "Internal. Set when a slider here is edited, cleared when a preset is applied.", false, hueContrastGroup, page)->setIsSecret(true);
    defineBool(desc, kODT_LockHc, "Lock (Persistent Values)", "Keep these values when switching presets; still manually adjustable.", false, hueContrastGroup, page);
    defineDouble(desc, kODT_HcR, "Hue Contrast R", "Red hue contrast adjustment", 0.6, -1.0, 1.0, 0.001, hueContrastGroup, page);

    // -- Beta features (gates) --
    // The Advanced Hue Contrast Enable is defined here to keep the original
    // param definition order, but it lives in the Advanced Hue Contrast
    // sub-group of Stickshift Mode, not in Beta Features.
    defineBool(desc, kODT_FilmicMode,     "Enable Filmic Mode Beta",      "Enable filmic mode rendering",             false, betaFeaturesGroup,   page);
    defineBool(desc, kODT_AdvHueContrast, "Enable Advanced Hue Contrast", "Enable advanced hue contrast adjustments", false, advHueContrastGroup, page);
    defineBool(desc, kODT_BetaFeatures,   "Enable Beta Features",         "Turns on experimental controls",           false, betaFeaturesGroup,   page);

    // -- Advanced hue contrast --
    defineDouble(desc, kODT_AdvHcR,     "Red Contrast",     "Advanced red hue contrast",     1.0, 0.1, 4.0, 0.01, advHueContrastGroup, page);
    defineDouble(desc, kODT_AdvHcG,     "Green Contrast",   "Green hue contrast",            1.0, 0.1, 4.0, 0.01, advHueContrastGroup, page);
    defineDouble(desc, kODT_AdvHcB,     "Blue Contrast",    "Blue hue contrast",             1.0, 0.1, 4.0, 0.01, advHueContrastGroup, page);
    defineDouble(desc, kODT_AdvHcC,     "Cyan Contrast",    "Cyan hue contrast",             1.0, 0.1, 4.0, 0.01, advHueContrastGroup, page);
    defineDouble(desc, kODT_AdvHcM,     "Magenta Contrast", "Magenta hue contrast",          1.0, 0.1, 4.0, 0.01, advHueContrastGroup, page);
    defineDouble(desc, kODT_AdvHcY,     "Yellow Contrast",  "Yellow hue contrast",           1.0, 0.1, 4.0, 0.01, advHueContrastGroup, page);
    defineDouble(desc, kODT_AdvHcPower, "Power Strength",   "Blends adjusted vs unadjusted. 1.0 = full, 0.0 = off", 1.0, 0.0, 1.0, 0.01, advHueContrastGroup, page);

    // -- Diagnostics --
    defineBool(desc, kODT_TonescaleMap,    "Tonescale Curve", "Enable tonescale mapping visualization", false, diagnosticsGroup, page);
    defineBool(desc, kODT_DiagnosticsMode, "Grey Scale Ramp", "Enable grey scale ramp",                 false, diagnosticsGroup, page);
    defineBool(desc, kODT_RgbChips,        "RGB Chips",
               "Test chart: seven rows (R, Y, G, C, B, M, grey). Each row is an exposure ramp from black on the left, "
               "through full saturation, to the creative white point white on the right. Grey sits a third of the way across. "
               "Replaces the image and ignores the In Transfer Function.",
               false, diagnosticsGroup, page);

    // -- Filmic dynamic range --
    defineDouble(desc, kODT_FilmicSrcStops, "Original Camera Range", "Stops captured by the camera/scene",      14.0, 1.0, 20.0, 1.0,  filmicDynamicRangeGroup, page);
    defineDouble(desc, kODT_FilmicTgtStops, "Target Film Range",     "Stops to compress into to mimic film",    10.0, 1.0, 20.0, 1.0,  filmicDynamicRangeGroup, page);
    defineDouble(desc, kODT_FilmicDynRange, "Roll Off Characteristics","Highlight rolloff: lower=harder, higher=gentler", 5.0, 1.0, 10.0, 0.1, filmicDynamicRangeGroup, page);
    defineDouble(desc, kODT_FilmicStrength, "Strength",              "Blends tonescaled vs compressed result",  0.0,  0.0, 1.0,  0.01, filmicDynamicRangeGroup, page);

    // -- Filmic projector sim (carried; not read by the kernel) --
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kODT_FilmicProjSim);
        p->setLabel("Projector Simulation");
        p->setHint("Filmic projector simulation type (carried; not yet applied by the device path)");
        p->appendOption("None");
        p->appendOption("Xenon");
        p->appendOption("Tungsten");
        p->appendOption("LED");
        p->setDefault(0);
        p->setAnimates(true);
        p->setParent(*filmicProjectorSimGroup);
        page->addChild(*p);
    }

    // -- Claude control channel identity --
    // Hidden persistent uuid, defined LAST so no existing param's restore order
    // shifts. Defined in both the suite and the standalone build.
    ldt::control::Channel::describeUuidParam(desc, page);

    // -- v1.1 model params (hidden unless Model is v1.1). Defaults are exactly
    //    opendrt_v11_defaults, read through the host table. --
    {
        double def[ODT11_NDESC];
        odt11_default_values(def);
        for (int i = 0; i < ODT11_NDESC; ++i) {
            const Odt11Desc& d = kOdt11Desc[i];
            GroupParamDescriptor* parent = v11Groups[d.group];
            if (d.kind == ODT11_DOUBLE) {
                DoubleParamDescriptor* p = desc.defineDoubleParam(d.name);
                p->setLabels(d.label, d.label, d.label);
                p->setScriptName(d.name);
                p->setHint(d.hint);
                p->setDefault(def[i]);
                p->setRange(d.mn, d.mx);
                p->setDisplayRange(d.mn, d.mx);
                p->setIncrement(d.inc);
                p->setDoubleType(eDoubleTypePlain);
                p->setAnimates(true);
                if (parent) p->setParent(*parent);
                page->addChild(*p);
            } else if (d.kind == ODT11_BOOL) {
                BooleanParamDescriptor* p = desc.defineBooleanParam(d.name);
                p->setLabels(d.label, d.label, d.label);
                p->setScriptName(d.name);
                p->setHint(d.hint);
                p->setDefault(def[i] != 0.0);
                p->setAnimates(true);
                if (parent) p->setParent(*parent);
                page->addChild(*p);
            } else {
                ChoiceParamDescriptor* p = desc.defineChoiceParam(d.name);
                p->setLabels(d.label, d.label, d.label);
                p->setScriptName(d.name);
                p->setHint(d.hint);
                for (int o = 0; o < d.nopts; ++o) p->appendOption(d.opts[o]);
                p->setDefault((int)std::lround(def[i]));
                p->setAnimates(true);
                if (parent) p->setParent(*parent);
                page->addChild(*p);
            }
        }
    }

    // -- Tesseract and Chromagnon params and groups (hidden unless that model
    //    is picked). Defined after the v1.1 params so no existing restore order
    //    shifts. Defaults are exactly tesseract_defaults / chromagnon_defaults,
    //    read through the host tables. Group slot 0 is the shared InputGroup. --
    auto defineAltParam = [&](const char* name, const char* label, const char* hint,
                              int kind, double mn, double mx, double inc,
                              const char* const* opts, int nopts, double def,
                              GroupParamDescriptor* parent) {
        if (kind == 0) {            // double
            DoubleParamDescriptor* p = desc.defineDoubleParam(name);
            p->setLabels(label, label, label);
            p->setScriptName(name);
            p->setHint(hint);
            p->setDefault(def);
            p->setRange(mn, mx);
            p->setDisplayRange(mn, mx);
            p->setIncrement(inc);
            p->setDoubleType(eDoubleTypePlain);
            p->setAnimates(true);
            if (parent) p->setParent(*parent);
            page->addChild(*p);
        } else if (kind == 1) {     // bool
            BooleanParamDescriptor* p = desc.defineBooleanParam(name);
            p->setLabels(label, label, label);
            p->setScriptName(name);
            p->setHint(hint);
            p->setDefault(def != 0.0);
            p->setAnimates(true);
            if (parent) p->setParent(*parent);
            page->addChild(*p);
        } else {                    // choice
            ChoiceParamDescriptor* p = desc.defineChoiceParam(name);
            p->setLabels(label, label, label);
            p->setScriptName(name);
            p->setHint(hint);
            for (int o = 0; o < nopts; ++o) p->appendOption(opts[o]);
            p->setDefault((int)std::lround(def));
            p->setAnimates(true);
            if (parent) p->setParent(*parent);
            page->addChild(*p);
        }
    };
    static_assert(TESS_DOUBLE == 0 && TESS_BOOL == 1 && TESS_CHOICE == 2, "kind ids");
    static_assert(CMG_DOUBLE == 0 && CMG_BOOL == 1 && CMG_CHOICE == 2, "kind ids");
    {
        GroupParamDescriptor* tessGroups[TESS_G_COUNT] = {};
        tessGroups[TESS_G_INPUT] = inputGroup;
        for (int g = 1; g < TESS_G_COUNT; ++g) {
            const TessGroupInfo& gi = kTessGroups[g];
            tessGroups[g] = desc.defineGroupParam(gi.name);
            tessGroups[g]->setLabels(gi.label, gi.label, gi.label);
            tessGroups[g]->setOpen(gi.open);
            page->addChild(*tessGroups[g]);
        }
        double def[TESS_NDESC];
        tess_default_values(def);
        for (int i = 0; i < TESS_NDESC; ++i) {
            const TessDesc& d = kTessDesc[i];
            defineAltParam(d.name, d.label, d.hint, d.kind, d.mn, d.mx, d.inc,
                           d.opts, d.nopts, def[i], tessGroups[d.group]);
        }
    }
    {
        GroupParamDescriptor* cmgGroups[CMG_G_COUNT] = {};
        cmgGroups[CMG_G_INPUT] = inputGroup;
        for (int g = 1; g < CMG_G_COUNT; ++g) {
            const CmgGroupInfo& gi = kCmgGroups[g];
            cmgGroups[g] = desc.defineGroupParam(gi.name);
            cmgGroups[g]->setLabels(gi.label, gi.label, gi.label);
            cmgGroups[g]->setOpen(gi.open);
            page->addChild(*cmgGroups[g]);
        }
        double def[CMG_NDESC];
        cmg_default_values(def);
        for (int i = 0; i < CMG_NDESC; ++i) {
            const CmgDesc& d = kCmgDesc[i];
            defineAltParam(d.name, d.label, d.hint, d.kind, d.mn, d.mx, d.inc,
                           d.opts, d.nopts, def[i], cmgGroups[d.group]);
        }
    }

    // My Presets + Save and Share (design 3.9). Defined after every existing
    // param so no id moves, and before Factory Settings, which stays last.
    // my_preset is top-level, not in PresetGroup: PresetGroup is hidden on
    // the v1.1 / Tesseract / Chromagnon models, and user presets cover all five.
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kODT_MyPreset);
        p->setLabel("My Presets");
        p->setHint("Your saved presets. Pick one to apply it. Locked modules are left alone. "
                   "Save and manage them under Save and Share.");
        p->appendOption("None");
        p->setDefault(0);
        p->setAnimates(false);
        p->setEvaluateOnChange(false);
        // The hidden ref below is the truth; the index is rebuilt per machine.
        p->setIsPersistant(false);
        if (page) page->addChild(*p);
    }
    {
        StringParamDescriptor* p = desc.defineStringParam(kODT_PmSelected);
        p->setLabel("Selected Preset");
        p->setHint("Holds family|model|id|name of the last applied user preset.");
        p->setDefault("");
        p->setStringType(OFX::eStringTypeSingleLine);
        p->setIsSecret(true);
        p->setIsPersistant(true);
        p->setAnimates(false);
        p->setEvaluateOnChange(false);
        if (page) page->addChild(*p);
    }
    {
        GroupParamDescriptor* g = desc.defineGroupParam(kODT_PmGroup);
        g->setLabel("Save and Share");
        g->setHint("Save your own looks and tonescales, and move them between computers.");
        g->setOpen(false);
        if (page) page->addChild(*g);

        ChoiceParamDescriptor* fam = desc.defineChoiceParam(kODT_PmFamily);
        fam->setLabel("Save As");
        fam->setHint("What a new preset keeps. Full keeps everything, Look keeps the colour, "
                     "Tonescale keeps the contrast curve.");
        fam->appendOption("Full");        // append-only
        fam->appendOption("Look");
        fam->appendOption("Tonescale");
        fam->setDefault(0);
        fam->setAnimates(false);
        fam->setEvaluateOnChange(false);
        fam->setParent(*g);
        if (page) page->addChild(*fam);

        StringParamDescriptor* nm = desc.defineStringParam(kODT_PmName);
        nm->setLabel("Name");
        nm->setHint("Name for Save New and Rename.");
        nm->setDefault("My Preset");
        nm->setStringType(OFX::eStringTypeSingleLine);
        nm->setAnimates(false);
        nm->setEvaluateOnChange(false);
        nm->setParent(*g);
        if (page) page->addChild(*nm);

        struct Btn { const char* id; const char* label; const char* hint; };
        const Btn btns[] = {
            { kODT_PmSaveNew,    "Save New",           "Saves the current settings as a new preset." },
            { kODT_PmUpdate,     "Update Selected",    "Overwrites the selected preset with the current settings." },
            { kODT_PmRename,     "Rename Selected",    "Gives the selected preset the name above." },
            { kODT_PmDelete,     "Delete Selected",    "Removes the selected preset. Nodes using it keep their settings." },
            { kODT_PmExport,     "Export to File",     "Saves the selected preset (or the current settings) as a file you can share." },
            { kODT_PmImport,     "Import from File",   "Adds a preset file to My Presets." },
            { kODT_PmShowFolder, "Show Preset Folder", "Opens the folder your presets live in." },
        };
        for (const Btn& b : btns) {
            PushButtonParamDescriptor* p = desc.definePushButtonParam(b.id);
            p->setLabel(b.label);
            p->setHint(b.hint);
            p->setParent(*g);
            if (page) page->addChild(*p);
        }
    }

    // Factory Settings: user starting points. Always last.
    ldt::factory::Options factory;
    factory.pluginId = getID();
    factory.storeKey = "OpenDRT";
    factory.toolName = OPENDRT_PLUGIN_LABEL;
    // User preset state is per machine, never a factory starting point.
    factory.excludeIds = { kODT_DiagnosticsMode, kODT_RgbChips, kODT_TonescaleMap, kODT_MyPreset };
    factory.excludePrefixes = { kODT_PmPrefix };
    ldt::factory::describe(desc, page, factory);
}

OFX::ImageEffect* OpenDRTFactory::createInstance(OfxImageEffectHandle handle,
                                                 OFX::ContextEnum /*ctx*/) {
    return new OpenDRTPlugin(handle);
}
