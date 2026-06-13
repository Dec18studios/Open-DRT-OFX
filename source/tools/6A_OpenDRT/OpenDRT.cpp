// OpenDRT.cpp — "6A. OpenDRT" OFX plugin (factory + dispatch).
//
// Full OpenDRT display-rendering transform: input linearize + gamut → tonescale →
// per-hue purity/brilliance/hueshift/contrast → creative white point → display
// encode. The per-pixel math + host precompute live in core/OpenDRTAlgorithm.h
// (bit-exact vs the Metal golden — see that header's provenance note). This file
// only builds the param UI, populates OpenDRTParams on the host, runs
// opendrt_precompute(), and dispatches to CPU/CUDA/Metal.
//
// The params are ported VERBATIM (names, defaults, ranges, groups) from the
// canonical OpenDRT.cpp so a project authored against canonical OpenDRT round-
// trips. The canonical's preset-value override engine + dynamic-matrix
// (ColorMatrices.json / MatrixManager) layers are intentionally NOT reproduced;
// see the port-limitations note at the bottom of this file.

#ifndef NOMINMAX
  #define NOMINMAX
#endif

#include "OpenDRT.h"

#include <memory>

// ── Dual-build plugin identity ────────────────────────────────────────────────
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
    "OpenDRT display rendering transform — scene-linear to display. Tonescale, " \
    "creative white point, per-hue purity / brilliance / hueshift / contrast, and " \
    "display encoding (Rec.709/P3/Rec.2020, gamma/PQ/HLG). Ported from OpenDRT; " \
    "part of the Look Dev Tools suite."

#define kSupportsTiles            false
#define kSupportsMultiResolution  false
#define kSupportsMultipleClipPARs false

// ── Look-preset → PresetEnable + cwp table ────────────────────────────────────
// Mirrors OpenDRTPresets.h LOOK_PRESETS[0..3]. The kernel gates each module on
// (PresetEnable || UIEnable); the look_preset dropdown supplies PresetEnable
// exactly as canonical OpenDRT does. cwp "Use Look Preset" (index 4) resolves to
// the preset's creative-white index. The full per-preset VALUE override engine is
// out of scope (precompute never consumes it) — only the gates + cwp are honored.
namespace {
struct LookPresetGates {
    int hcon, lcon, ptl, ptm, brl, hsRgb, hsCmy, hc;
    int cwp;
};
// idx: 0 Default, 1 Colorful, 2 Umbra, 3 Base
static const LookPresetGates kLookGates[4] = {
    { 0, 1, 1, 1, 1, 1, 1, 1, 0 }, // Default
    { 0, 1, 1, 1, 1, 1, 1, 1, 0 }, // Colorful
    { 0, 1, 1, 1, 1, 1, 1, 1, 3 }, // Umbra
    { 0, 0, 1, 0, 0, 0, 0, 0, 0 }, // Base
};
} // namespace

// ── Param-definition helpers ──────────────────────────────────────────────────
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

// ── OpenDRTProcessor (CPU path) ───────────────────────────────────────────────
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

// ── OpenDRTPlugin ─────────────────────────────────────────────────────────────
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
    _tnHcon          = fetchDoubleParam(kODT_TnHcon);
    _tnHconPv        = fetchDoubleParam(kODT_TnHconPv);
    _tnHconSt        = fetchDoubleParam(kODT_TnHconSt);

    _tnLconEnable    = fetchBooleanParam(kODT_TnLconEnable);
    _tnLcon          = fetchDoubleParam(kODT_TnLcon);
    _tnLconW         = fetchDoubleParam(kODT_TnLconW);
    _tnLconPc        = fetchDoubleParam(kODT_TnLconPc);

    _cwp             = fetchChoiceParam(kODT_Cwp);
    _cwpRng          = fetchDoubleParam(kODT_CwpRng);
    _rsSa            = fetchDoubleParam(kODT_RsSa);
    _rsRw            = fetchDoubleParam(kODT_RsRw);
    _rsBw            = fetchDoubleParam(kODT_RsBw);

    _ptlEnable       = fetchBooleanParam(kODT_PtlEnable);
    _ptR             = fetchDoubleParam(kODT_PtR);
    _ptG             = fetchDoubleParam(kODT_PtG);
    _ptB             = fetchDoubleParam(kODT_PtB);
    _ptRngLow        = fetchDoubleParam(kODT_PtRngLow);
    _ptRngHigh       = fetchDoubleParam(kODT_PtRngHigh);

    _ptmEnable       = fetchBooleanParam(kODT_PtmEnable);
    _ptmLow          = fetchDoubleParam(kODT_PtmLow);
    _ptmLowSt        = fetchDoubleParam(kODT_PtmLowSt);
    _ptmHigh         = fetchDoubleParam(kODT_PtmHigh);
    _ptmHighSt       = fetchDoubleParam(kODT_PtmHighSt);

    _brlEnable       = fetchBooleanParam(kODT_BrlEnable);
    _brlR            = fetchDoubleParam(kODT_BrlR);
    _brlG            = fetchDoubleParam(kODT_BrlG);
    _brlB            = fetchDoubleParam(kODT_BrlB);
    _brlC            = fetchDoubleParam(kODT_BrlC);
    _brlM            = fetchDoubleParam(kODT_BrlM);
    _brlY            = fetchDoubleParam(kODT_BrlY);
    _brlRng          = fetchDoubleParam(kODT_BrlRng);

    _hsRgbEnable     = fetchBooleanParam(kODT_HsRgbEnable);
    _hsR             = fetchDoubleParam(kODT_HsR);
    _hsG             = fetchDoubleParam(kODT_HsG);
    _hsB             = fetchDoubleParam(kODT_HsB);
    _hsRgbRng        = fetchDoubleParam(kODT_HsRgbRng);

    _hsCmyEnable     = fetchBooleanParam(kODT_HsCmyEnable);
    _hsC             = fetchDoubleParam(kODT_HsC);
    _hsM             = fetchDoubleParam(kODT_HsM);
    _hsY             = fetchDoubleParam(kODT_HsY);

    _hcEnable        = fetchBooleanParam(kODT_HcEnable);
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
}

OpenDRTParams OpenDRTPlugin::buildParams(double time, int /*w*/, int /*h*/) const {
    OpenDRTParams p{};

    // ── Choice params ──
    _inGamut->getValueAtTime(time, p.inGamut);
    _inOetf->getValueAtTime(time, p.inOetf);
    _displayGamut->getValueAtTime(time, p.displayGamut);
    _eotf->getValueAtTime(time, p.eotf);
    int lookPreset = 0;
    _lookPreset->getValueAtTime(time, lookPreset);
    int cwpChoice = 0;
    _cwp->getValueAtTime(time, cwpChoice);
    _filmicProjSim->getValueAtTime(time, p.filmicProjectorSim);

    // ── Bool params (→ int) ──
    bool b = false;
    _clamp->getValueAtTime(time, b);           p.clamp              = b ? 1 : 0;
    _advHueContrast->getValueAtTime(time, b);  p.advHueContrast     = b ? 1 : 0;
    _betaFeatures->getValueAtTime(time, b);    p.betaFeaturesEnable = b ? 1 : 0;
    _filmicMode->getValueAtTime(time, b);      p.filmicMode         = b ? 1 : 0;
    _tonescaleMap->getValueAtTime(time, b);    p.tonescaleMap       = b ? 1 : 0;
    _diagnosticsMode->getValueAtTime(time, b); p.diagnosticsMode    = b ? 1 : 0;
    _rgbChips->getValueAtTime(time, b);        p.rgbChipsMode       = b ? 1 : 0;

    // ── UI enable gates (user checkboxes) ──
    _tnHconEnable->getValueAtTime(time, b);    p.tnHconUIEnable = b ? 1 : 0;
    _tnLconEnable->getValueAtTime(time, b);    p.tnLconUIEnable = b ? 1 : 0;
    _ptlEnable->getValueAtTime(time, b);       p.ptlUIEnable    = b ? 1 : 0;
    _ptmEnable->getValueAtTime(time, b);       p.ptmUIEnable    = b ? 1 : 0;
    _brlEnable->getValueAtTime(time, b);       p.brlUIEnable    = b ? 1 : 0;
    _hsRgbEnable->getValueAtTime(time, b);     p.hsRgbUIEnable  = b ? 1 : 0;
    _hsCmyEnable->getValueAtTime(time, b);     p.hsCmyUIEnable  = b ? 1 : 0;
    _hcEnable->getValueAtTime(time, b);        p.hcUIEnable     = b ? 1 : 0;

    // ── Preset enable gates (from the current look preset) ──
    if (lookPreset < 0 || lookPreset > 3) lookPreset = 0;
    const LookPresetGates& g = kLookGates[lookPreset];
    p.tnHconPresetEnable = g.hcon;
    p.tnLconPresetEnable = g.lcon;
    p.ptlPresetEnable    = g.ptl;
    p.ptmPresetEnable    = g.ptm;
    p.brlPresetEnable    = g.brl;
    p.hsRgbPresetEnable  = g.hsRgb;
    p.hsCmyPresetEnable  = g.hsCmy;
    p.hcPresetEnable     = g.hc;

    // ── Creative white: "Use Look Preset" (index 4) resolves to preset cwp ──
    p.cwp = (cwpChoice == 4) ? g.cwp : cwpChoice;

    // ── Double params ──
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

    // ── Hue Anchor Compression: not exposed in this build (no UI param).
    //    Left disabled (zeroed by the {} init), so the kernel skips that block.
    p.hueCompressionEnable = 0;

    // ── Per-frame precompute: display-encoding remap + ts_* tonescale constants.
    //    Mutates p.displayGamut/p.eotf to the canonical pair the kernel reads. ──
    opendrt_precompute(&p);

    return p;
}

bool OpenDRTPlugin::isIdentity(const OFX::IsIdentityArguments& /*args*/,
                               OFX::Clip*& /*identityClip*/, double& /*identityTime*/) {
    // OpenDRT is a display transform — never an identity (it always remaps).
    return false;
}

void OpenDRTPlugin::render(const OFX::RenderArguments& args) {
    std::unique_ptr<OFX::Image> srcImg(_srcClip->fetchImage(args.time));
    std::unique_ptr<OFX::Image> dstImg(_dstClip->fetchImage(args.time));
    if (!dstImg.get()) return;

    OfxRectI bounds = dstImg->getBounds();
    int width  = bounds.x2 - bounds.x1;
    int height = bounds.y2 - bounds.y1;

    int srcRowBytes = srcImg.get() ? srcImg->getRowBytes()
                                    : width * 4 * (int)sizeof(float);
    int dstRowBytes = dstImg->getRowBytes();

    OpenDRTParams params = buildParams(args.time, width, height);

    // ── CUDA path ─────────────────────────────────────────────────────────────
#ifdef LDT_HAS_CUDA
    if (args.pCudaStream && srcImg.get()) {
        const float* src = static_cast<const float*>(srcImg->getPixelAddress(bounds.x1, bounds.y1));
        float*       dst = static_cast<float*>(dstImg->getPixelAddress(bounds.x1, bounds.y1));
        RunOpenDRTCudaKernel(args.pCudaStream, width, height,
                             srcRowBytes, dstRowBytes, &params, src, dst);
        return;
    }
#endif

    // ── Metal path ──────────────────────────────────────────────────────────
#ifdef __APPLE__
    if (args.pMetalCmdQ && srcImg.get()) {
        const float* src = static_cast<const float*>(srcImg->getPixelAddress(bounds.x1, bounds.y1));
        float*       dst = static_cast<float*>(dstImg->getPixelAddress(bounds.x1, bounds.y1));
        RunOpenDRTMetalKernel(args.pMetalCmdQ, width, height,
                              srcRowBytes, dstRowBytes, &params, src, dst);
        return;
    }
#endif

    // ── CPU path ──────────────────────────────────────────────────────────────
    OpenDRTProcessor processor(*this);
    processor.setDstImg(dstImg.get());
    processor.setSrcImg(srcImg.get());
    processor.setRenderWindow(args.renderWindow);
    processor.setParams(params);
    processor.process();
}

// ── Factory ───────────────────────────────────────────────────────────────────
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

    // ── Collapsible groups (mirror canonical OpenDRT) ──
    GroupParamDescriptor* inputGroup        = desc.defineGroupParam("InputGroup");
    inputGroup->setLabels("Input & Output Settings", "Input & Output Settings", "Input & Output Settings");
    inputGroup->setOpen(false);
    GroupParamDescriptor* presetGroup       = desc.defineGroupParam("PresetGroup");
    presetGroup->setLabels("Automatic Presets", "Automatic Presets", "Automatic Presets");
    GroupParamDescriptor* stickshiftGroup   = desc.defineGroupParam("StickshiftGroup");
    stickshiftGroup->setLabels("Stickshift Mode", "Stickshift Mode", "Stickshift Mode");
    stickshiftGroup->setOpen(false);
    GroupParamDescriptor* tonescaleGroup    = desc.defineGroupParam("TonescaleGroup");
    tonescaleGroup->setLabels("Tonescale", "Tonescale", "Tonescale");
    tonescaleGroup->setOpen(false);
    GroupParamDescriptor* contrastGroup     = desc.defineGroupParam("ContrastGroup");
    contrastGroup->setLabels("Basic Contrast", "Basic Contrast", "Basic Contrast");
    contrastGroup->setOpen(false);
    GroupParamDescriptor* colorGroup        = desc.defineGroupParam("ColorGroup");
    colorGroup->setLabels("Creative White Point", "Creative White Point", "Creative White Point");
    colorGroup->setOpen(false);
    GroupParamDescriptor* highContrastGroup = desc.defineGroupParam("HighContrastGroup");
    highContrastGroup->setLabels("High Contrast", "High Contrast", "High Contrast");
    highContrastGroup->setOpen(false);
    GroupParamDescriptor* lowContrastGroup  = desc.defineGroupParam("LowContrastGroup");
    lowContrastGroup->setLabels("Low Contrast", "Low Contrast", "Low Contrast");
    lowContrastGroup->setOpen(false);
    GroupParamDescriptor* purityLowGroup    = desc.defineGroupParam("PurityLowGroup");
    purityLowGroup->setLabels("Purity Compress Low", "Purity Compress Low", "Purity Compress Low");
    purityLowGroup->setOpen(false);
    GroupParamDescriptor* midPurityGroup    = desc.defineGroupParam("MidPurityGroup");
    midPurityGroup->setLabels("Mid Purity", "Mid Purity", "Mid Purity");
    midPurityGroup->setOpen(false);
    GroupParamDescriptor* brillianceGroup   = desc.defineGroupParam("BrillianceGroup");
    brillianceGroup->setLabels("Brilliance", "Brilliance", "Brilliance");
    brillianceGroup->setOpen(false);
    GroupParamDescriptor* hueshiftRgbGroup  = desc.defineGroupParam("HueshiftRgbGroup");
    hueshiftRgbGroup->setLabels("Hueshift RGB", "Hueshift RGB", "Hueshift RGB");
    hueshiftRgbGroup->setOpen(false);
    GroupParamDescriptor* hueshiftCmyGroup  = desc.defineGroupParam("HueshiftCmyGroup");
    hueshiftCmyGroup->setLabels("Hueshift CMY", "Hueshift CMY", "Hueshift CMY");
    hueshiftCmyGroup->setOpen(false);
    GroupParamDescriptor* hueContrastGroup  = desc.defineGroupParam("HueContrastGroup");
    hueContrastGroup->setLabels("Hue Contrast", "Hue Contrast", "Hue Contrast");
    hueContrastGroup->setOpen(false);
    GroupParamDescriptor* diagnosticsGroup  = desc.defineGroupParam("DiagnosticsGroup");
    diagnosticsGroup->setLabels("Diagnostics", "Diagnostics", "Diagnostics");
    diagnosticsGroup->setOpen(false);
    GroupParamDescriptor* betaFeaturesGroup = desc.defineGroupParam("BetaFeaturesGroup");
    betaFeaturesGroup->setLabels("Beta Features", "Beta Features", "Beta Features");
    betaFeaturesGroup->setOpen(false);
    GroupParamDescriptor* advHueContrastGroup = desc.defineGroupParam("AdvHueContrastGroup");
    advHueContrastGroup->setLabels("Advanced Hue Contrast", "Advanced Hue Contrast", "Advanced Hue Contrast");
    advHueContrastGroup->setOpen(false);
    GroupParamDescriptor* filmicDynamicRangeGroup = desc.defineGroupParam("FilmicDynamicRangeGroup");
    filmicDynamicRangeGroup->setLabels("Filmic Dynamic Range Beta", "Filmic Dynamic Range Beta", "Filmic Dynamic Range Beta");
    filmicDynamicRangeGroup->setOpen(false);
    GroupParamDescriptor* filmicProjectorSimGroup = desc.defineGroupParam("FilmicProjectorSimGroup");
    filmicProjectorSimGroup->setLabels("Filmic Projector Sim Beta", "Filmic Projector Sim Beta", "Filmic Projector Sim Beta");
    filmicProjectorSimGroup->setOpen(false);

    page->addChild(*inputGroup);
    page->addChild(*presetGroup);
    page->addChild(*stickshiftGroup);
    page->addChild(*tonescaleGroup);
    page->addChild(*contrastGroup);
    page->addChild(*highContrastGroup);
    page->addChild(*lowContrastGroup);
    page->addChild(*colorGroup);
    page->addChild(*purityLowGroup);
    page->addChild(*midPurityGroup);
    page->addChild(*brillianceGroup);
    page->addChild(*hueshiftRgbGroup);
    page->addChild(*hueshiftCmyGroup);
    page->addChild(*hueContrastGroup);
    page->addChild(*betaFeaturesGroup);
    page->addChild(*diagnosticsGroup);
    page->addChild(*filmicDynamicRangeGroup);
    page->addChild(*filmicProjectorSimGroup);
    page->addChild(*advHueContrastGroup);

    // ── Input gamut (16 options; ONLY indices 0-5 fully matrixed — see port note) ──
    {
        ChoiceParamDescriptor* p = desc.defineChoiceParam(kODT_InGamut);
        p->setLabel("Input Gamut");
        p->setHint("Input color gamut/primaries. NOTE: indices 0-5 (XYZ / ACES2065-1 / "
                   "ACEScg / P3D65 / Rec.2020 / Rec.709) are fully supported; the camera "
                   "gamuts 6-15 currently fall back to Rec.709's matrix.");
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
        p->setHint("Input transfer function (OETF)");
        p->appendOption("Linear");
        p->appendOption("Davinci Intermediate");
        p->appendOption("Filmlight T-Log");
        p->appendOption("ACEScct");
        p->appendOption("Arri LogC3");
        p->appendOption("Arri LogC4");
        p->appendOption("RedLog3G10");
        p->appendOption("Panasonic V-Log");
        p->appendOption("Sony S-Log3");
        p->setDefault(1);
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

    // ── Look preset (governs PresetEnable gates + cwp "Use Look Preset"). Defined
    //    BEFORE the modules/cwp it influences, per the OFX restore-order gotcha. ──
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

    // ── Tonescale ──
    defineDouble(desc, kODT_TnLp,  "Display Peak Luminance", "Peak luminance of the display in nits", 100.0, 100.0, 1000.0, 1.0, tonescaleGroup, page);
    defineDouble(desc, kODT_TnGb,  "HDR Grey Boost",         "Boost grey levels for HDR displays",     0.13,  0.0,   1.0,    0.01, tonescaleGroup, page);
    defineDouble(desc, kODT_PtHdr, "HDR Purity",             "Purity adjustment for HDR",              0.5,   0.0,   1.0,    0.01, tonescaleGroup, page);

    // ── Basic contrast ──
    defineBool(desc, kODT_Clamp, "Clamp", "Enable tone curve clamping", true, contrastGroup, page);
    defineDouble(desc, kODT_TnLg,  "Grey Luminance", "Grey point luminance",          11.1,  4.0, 25.0, 0.1,   contrastGroup, page);
    defineDouble(desc, kODT_TnCon, "Contrast",       "Overall contrast adjustment",   1.4,   1.0, 2.0,  0.01,  contrastGroup, page);
    defineDouble(desc, kODT_TnSh,  "Shoulder Clip",  "Highlight shoulder clipping",   0.5,   0.0, 1.0,  0.01,  contrastGroup, page);
    defineDouble(desc, kODT_TnToe, "Toe",            "Shadow toe adjustment",         0.003, 0.0, 0.1,  0.001, contrastGroup, page);
    defineDouble(desc, kODT_TnOff, "Offset",         "Black point offset",            0.005, 0.0, 0.02, 0.001, contrastGroup, page);

    // ── High contrast ──
    defineBool(desc, kODT_TnHconEnable, "Enable Contrast High", "Enable high contrast adjustments", false, stickshiftGroup, page);
    defineDouble(desc, kODT_TnHcon,   "Contrast High",          "High frequency contrast",       0.0, -1.0, 1.0, 0.01, highContrastGroup, page);
    defineDouble(desc, kODT_TnHconPv, "Contrast High Pivot",    "Pivot point for high contrast", 1.0,  0.0, 4.0, 0.01, highContrastGroup, page);
    defineDouble(desc, kODT_TnHconSt, "Contrast High Strength", "Strength of high contrast",     4.0,  0.0, 4.0, 0.01, highContrastGroup, page);

    // ── Low contrast ──
    defineBool(desc, kODT_TnLconEnable, "Enable Contrast Low", "Enable low contrast adjustments", false, stickshiftGroup, page);
    defineDouble(desc, kODT_TnLcon,   "Contrast Low",            "Low frequency contrast",   1.0, 0.0, 3.0, 0.001, lowContrastGroup, page);
    defineDouble(desc, kODT_TnLconW,  "Contrast Low Width",      "Width of low contrast",    0.5, 0.0, 2.0, 0.001, lowContrastGroup, page);
    defineDouble(desc, kODT_TnLconPc, "Contrast Low Per-Channel","Per-channel low contrast", 1.0, 0.0, 1.0, 0.001, lowContrastGroup, page);

    // ── Creative white + render space ──
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

    // ── Purity compress low ──
    defineBool(desc, kODT_PtlEnable, "Enable Purity Low", "Enable purity compression low", false, stickshiftGroup, page);
    defineDouble(desc, kODT_PtR,        "Purity Compress R", "Red purity compression",            0.5, 0.0, 4.0, 0.001, purityLowGroup, page);
    defineDouble(desc, kODT_PtG,        "Purity Compress G", "Green purity compression",          2.0, 0.0, 4.0, 0.001, purityLowGroup, page);
    defineDouble(desc, kODT_PtB,        "Purity Compress B", "Blue purity compression",           2.0, 0.0, 4.0, 0.001, purityLowGroup, page);
    defineDouble(desc, kODT_PtRngLow,   "Purity Range Low",  "Low range for purity compression",  0.2, 0.1, 0.6, 0.001, purityLowGroup, page);
    defineDouble(desc, kODT_PtRngHigh,  "Purity Range High", "High range for purity compression", 0.8, 0.25,2.0, 0.001, purityLowGroup, page);

    // ── Mid purity ──
    defineBool(desc, kODT_PtmEnable, "Enable Mid Purity", "Enable mid purity adjustments", false, stickshiftGroup, page);
    defineDouble(desc, kODT_PtmLow,    "Mid Purity Low",          "Mid purity low adjustment",   0.2,  0.0,  1.0, 0.001, midPurityGroup, page);
    defineDouble(desc, kODT_PtmLowSt,  "Mid Purity Low Strength", "Strength of mid purity low",  0.5,  0.1,  1.0, 0.001, midPurityGroup, page);
    defineDouble(desc, kODT_PtmHigh,   "Mid Purity High",         "Mid purity high adjustment", -0.8, -0.9,  0.0, 0.001, midPurityGroup, page);
    defineDouble(desc, kODT_PtmHighSt, "Mid Purity High Strength","Strength of mid purity high", 0.3,  0.2,  1.0, 0.001, midPurityGroup, page);

    // ── Brilliance ──
    defineBool(desc, kODT_BrlEnable, "Enable Brilliance", "Enable brilliance adjustments", false, stickshiftGroup, page);
    defineDouble(desc, kODT_BrlR,   "Brilliance R",     "Red brilliance",     -0.5, -1.0, 1.0, 0.001, brillianceGroup, page);
    defineDouble(desc, kODT_BrlG,   "Brilliance G",     "Green brilliance",   -0.4, -1.0, 1.0, 0.001, brillianceGroup, page);
    defineDouble(desc, kODT_BrlB,   "Brilliance B",     "Blue brilliance",    -0.2, -1.0, 1.0, 0.001, brillianceGroup, page);
    defineDouble(desc, kODT_BrlC,   "Brilliance C",     "Cyan brilliance",     0.0, -1.0, 1.0, 0.001, brillianceGroup, page);
    defineDouble(desc, kODT_BrlM,   "Brilliance M",     "Magenta brilliance",  0.0, -1.0, 1.0, 0.001, brillianceGroup, page);
    defineDouble(desc, kODT_BrlY,   "Brilliance Y",     "Yellow brilliance",   0.0, -1.0, 1.0, 0.001, brillianceGroup, page);
    defineDouble(desc, kODT_BrlRng, "Brilliance Range", "Range of brilliance", 0.66, 0.0, 2.0, 0.001, brillianceGroup, page);

    // ── Hueshift RGB ──
    defineBool(desc, kODT_HsRgbEnable, "Enable Hueshift RGB", "Enable RGB hue shifting", false, stickshiftGroup, page);
    defineDouble(desc, kODT_HsR,      "Hueshift R",         "Red hue shifting",                0.35, -1.0, 1.0, 0.001, hueshiftRgbGroup, page);
    defineDouble(desc, kODT_HsG,      "Hueshift G",         "Green hue shifting",              0.25, -1.0, 1.0, 0.001, hueshiftRgbGroup, page);
    defineDouble(desc, kODT_HsB,      "Hueshift B",         "Blue hue shifting",               0.5,  -1.0, 1.0, 0.001, hueshiftRgbGroup, page);
    defineDouble(desc, kODT_HsRgbRng, "Hueshift RGB Range", "Range of RGB hue shifting",       0.6,   0.0, 2.0, 0.001, hueshiftRgbGroup, page);

    // ── Hueshift CMY ──
    defineBool(desc, kODT_HsCmyEnable, "Enable Hueshift CMY", "Enable CMY hue shifting", false, stickshiftGroup, page);
    defineDouble(desc, kODT_HsC, "Hueshift C", "Cyan hue shifting",    0.2, -1.0, 1.0, 0.001, hueshiftCmyGroup, page);
    defineDouble(desc, kODT_HsM, "Hueshift M", "Magenta hue shifting", 0.2, -1.0, 1.0, 0.001, hueshiftCmyGroup, page);
    defineDouble(desc, kODT_HsY, "Hueshift Y", "Yellow hue shifting",  0.2, -1.0, 1.0, 0.001, hueshiftCmyGroup, page);

    // ── Hue contrast ──
    defineBool(desc, kODT_HcEnable, "Enable Hue Contrast", "Enable hue contrast adjustments", false, stickshiftGroup, page);
    defineDouble(desc, kODT_HcR, "Hue Contrast R", "Red hue contrast adjustment", 0.6, -1.0, 1.0, 0.001, hueContrastGroup, page);

    // ── Beta features (gates) ──
    defineBool(desc, kODT_FilmicMode,     "Enable Filmic Mode Beta",  "Enable filmic mode rendering",            false, betaFeaturesGroup, page);
    defineBool(desc, kODT_AdvHueContrast, "Enable Adv Hue Contrast",  "Enable advanced hue contrast adjustments", false, betaFeaturesGroup, page);
    defineBool(desc, kODT_BetaFeatures,   "Enable Beta Features",     "Turns on experimental controls",          false, betaFeaturesGroup, page);

    // ── Advanced hue contrast ──
    defineDouble(desc, kODT_AdvHcR,     "Red Contrast",     "Advanced red hue contrast",     1.0, 0.1, 4.0, 0.01, advHueContrastGroup, page);
    defineDouble(desc, kODT_AdvHcG,     "Green Contrast",   "Green hue contrast",            1.0, 0.1, 4.0, 0.01, advHueContrastGroup, page);
    defineDouble(desc, kODT_AdvHcB,     "Blue Contrast",    "Blue hue contrast",             1.0, 0.1, 4.0, 0.01, advHueContrastGroup, page);
    defineDouble(desc, kODT_AdvHcC,     "Cyan Contrast",    "Cyan hue contrast",             1.0, 0.1, 4.0, 0.01, advHueContrastGroup, page);
    defineDouble(desc, kODT_AdvHcM,     "Magenta Contrast", "Magenta hue contrast",          1.0, 0.1, 4.0, 0.01, advHueContrastGroup, page);
    defineDouble(desc, kODT_AdvHcY,     "Yellow Contrast",  "Yellow hue contrast",           1.0, 0.1, 4.0, 0.01, advHueContrastGroup, page);
    defineDouble(desc, kODT_AdvHcPower, "Power Strength",   "Blends adjusted vs unadjusted. 1.0 = full, 0.0 = off", 1.0, 0.0, 1.0, 0.01, advHueContrastGroup, page);

    // ── Diagnostics ──
    defineBool(desc, kODT_TonescaleMap,    "Tonescale Curve", "Enable tonescale mapping visualization", false, diagnosticsGroup, page);
    defineBool(desc, kODT_DiagnosticsMode, "Grey Scale Ramp", "Enable grey scale ramp",                 false, diagnosticsGroup, page);
    defineBool(desc, kODT_RgbChips,        "RGB Chips",       "Enable RGB chips",                       false, diagnosticsGroup, page);

    // ── Filmic dynamic range ──
    defineDouble(desc, kODT_FilmicSrcStops, "Original Camera Range", "Stops captured by the camera/scene",      14.0, 1.0, 20.0, 1.0,  filmicDynamicRangeGroup, page);
    defineDouble(desc, kODT_FilmicTgtStops, "Target Film Range",     "Stops to compress into to mimic film",    10.0, 1.0, 20.0, 1.0,  filmicDynamicRangeGroup, page);
    defineDouble(desc, kODT_FilmicDynRange, "Roll Off Characteristics","Highlight rolloff: lower=harder, higher=gentler", 5.0, 1.0, 10.0, 0.1, filmicDynamicRangeGroup, page);
    defineDouble(desc, kODT_FilmicStrength, "Strength",              "Blends tonescaled vs compressed result",  0.0,  0.0, 1.0,  0.01, filmicDynamicRangeGroup, page);

    // ── Filmic projector sim (carried; not read by the kernel) ──
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
}

OFX::ImageEffect* OpenDRTFactory::createInstance(OfxImageEffectHandle handle,
                                                 OFX::ContextEnum /*ctx*/) {
    return new OpenDRTPlugin(handle);
}
