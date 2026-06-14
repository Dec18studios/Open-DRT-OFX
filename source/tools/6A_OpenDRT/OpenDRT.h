#pragma once
// OpenDRT.h — "6A. OpenDRT" OFX param names + class declarations.
//
// kODT_* are the OFX parameter names seen by the host; they mirror the canonical
// OpenDRT names (in_gamut, _tn_con, …) so a project authored against canonical
// OpenDRT round-trips. They must never change once shipped. OpenDRTProcessor is
// the CPU worker. OpenDRTPlugin owns the OFX param objects and dispatches render
// to CPU/CUDA/Metal. OpenDRTFactory registers the plugin. The per-pixel math and
// host precompute live in core/OpenDRTAlgorithm.h (bit-exact vs the Metal golden).

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"

#include "core/OpenDRTAlgorithm.h"

// ── GPU back-end host entry points ────────────────────────────────────────────
#ifdef LDT_HAS_CUDA
void RunOpenDRTCudaKernel(void* p_Stream, int p_Width, int p_Height,
                          int p_SrcRowBytes, int p_DstRowBytes,
                          const OpenDRTParams* p_Params,
                          const float* p_Input, float* p_Output);
#endif

#ifdef __APPLE__
void RunOpenDRTMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                           int p_SrcRowBytes, int p_DstRowBytes,
                           const OpenDRTParams* p_Params,
                           const float* p_Input, float* p_Output);
#endif

// ── OFX parameter name constants (canonical OpenDRT names; never change) ───────
// Input / Output
static const char* kODT_InGamut          = "in_gamut";
static const char* kODT_InOetf           = "in_oetf";
static const char* kODT_DisplayGamut     = "_display_gamut";
static const char* kODT_Eotf             = "_eotf";
// Look preset (governs *PresetEnable gates + cwp "Use Look Preset" resolution)
static const char* kODT_LookPreset       = "look_preset";
// Tonescale
static const char* kODT_TnLp             = "tn_Lp";
static const char* kODT_TnGb             = "tn_gb";
static const char* kODT_PtHdr            = "pt_hdr";
// Basic contrast
static const char* kODT_Clamp            = "_clamp";
static const char* kODT_TnLg             = "_tn_Lg";
static const char* kODT_TnCon            = "_tn_con";
static const char* kODT_TnSh             = "_tn_sh";
static const char* kODT_TnToe            = "_tn_toe";
static const char* kODT_TnOff            = "_tn_off";
// High contrast
static const char* kODT_TnHconEnable     = "_tn_hcon_enable";
static const char* kODT_TnHcon           = "_tn_hcon";
static const char* kODT_TnHconPv         = "_tn_hcon_pv";
static const char* kODT_TnHconSt         = "_tn_hcon_st";
// Low contrast
static const char* kODT_TnLconEnable     = "_tn_lcon_enable";
static const char* kODT_TnLcon           = "_tn_lcon";
static const char* kODT_TnLconW          = "_tn_lcon_w";
static const char* kODT_TnLconPc         = "_tn_lcon_pc";
// Creative white + render space
static const char* kODT_Cwp              = "_cwp";
static const char* kODT_CwpRng           = "_cwp_rng";
static const char* kODT_RsSa             = "_rs_sa";
static const char* kODT_RsRw             = "_rs_rw";
static const char* kODT_RsBw             = "_rs_bw";
// Purity compress low
static const char* kODT_PtlEnable        = "_ptl_enable";
static const char* kODT_PtR              = "_pt_r";
static const char* kODT_PtG              = "_pt_g";
static const char* kODT_PtB              = "_pt_b";
static const char* kODT_PtRngLow         = "_pt_rng_low";
static const char* kODT_PtRngHigh        = "_pt_rng_high";
// Mid purity
static const char* kODT_PtmEnable        = "_ptm_enable";
static const char* kODT_PtmLow           = "_ptm_low";
static const char* kODT_PtmLowSt         = "_ptm_low_st";
static const char* kODT_PtmHigh          = "_ptm_high";
static const char* kODT_PtmHighSt        = "_ptm_high_st";
// Brilliance
static const char* kODT_BrlEnable        = "_brl_enable";
static const char* kODT_BrlR             = "_brl_r";
static const char* kODT_BrlG             = "_brl_g";
static const char* kODT_BrlB             = "_brl_b";
static const char* kODT_BrlC             = "_brl_c";
static const char* kODT_BrlM             = "_brl_m";
static const char* kODT_BrlY             = "_brl_y";
static const char* kODT_BrlRng           = "_brl_rng";
// Hueshift RGB
static const char* kODT_HsRgbEnable      = "_hs_rgb_enable";
static const char* kODT_HsR              = "_hs_r";
static const char* kODT_HsG              = "_hs_g";
static const char* kODT_HsB              = "_hs_b";
static const char* kODT_HsRgbRng         = "_hs_rgb_rng";
// Hueshift CMY
static const char* kODT_HsCmyEnable      = "_hs_cmy_enable";
static const char* kODT_HsC              = "_hs_c";
static const char* kODT_HsM              = "_hs_m";
static const char* kODT_HsY              = "_hs_y";
// Hue contrast
static const char* kODT_HcEnable         = "_hc_enable";
static const char* kODT_HcR              = "_hc_r";
// Advanced hue contrast
static const char* kODT_AdvHueContrast   = "_adv_hue_contrast";
static const char* kODT_AdvHcR           = "_adv_hc_r";
static const char* kODT_AdvHcG           = "_adv_hc_g";
static const char* kODT_AdvHcB           = "_adv_hc_b";
static const char* kODT_AdvHcC           = "_adv_hc_c";
static const char* kODT_AdvHcM           = "_adv_hc_m";
static const char* kODT_AdvHcY           = "_adv_hc_y";
static const char* kODT_AdvHcPower       = "_adv_hc_power";
// Beta features + diagnostics
static const char* kODT_BetaFeatures     = "_beta_features_enable";
static const char* kODT_FilmicMode       = "_filmic_mode";
static const char* kODT_FilmicSrcStops   = "_filmic_source_stops";
static const char* kODT_FilmicTgtStops   = "_filmic_target_stops";
static const char* kODT_FilmicDynRange   = "_filmic_dynamic_range";
static const char* kODT_FilmicStrength   = "_filmic_strength";
static const char* kODT_FilmicProjSim    = "_filmic_projector_sim";
static const char* kODT_TonescaleMap     = "_tonescale_map";
static const char* kODT_DiagnosticsMode  = "_diagnostics_mode";
static const char* kODT_RgbChips         = "_rgbchips";

// ── OpenDRTProcessor (CPU path) ───────────────────────────────────────────────
class OpenDRTProcessor : public OFX::ImageProcessor {
public:
    explicit OpenDRTProcessor(OFX::ImageEffect& effect)
        : OFX::ImageProcessor(effect), _srcImg(nullptr) {}

    void setParams(const OpenDRTParams& p) { _params = p; }
    void setSrcImg(OFX::Image* v)          { _srcImg = v; }

    void multiThreadProcessImages(OfxRectI window) override;

private:
    OpenDRTParams _params;
    OFX::Image*   _srcImg;
};

// ── OpenDRTPlugin ─────────────────────────────────────────────────────────────
class OpenDRTPlugin : public OFX::ImageEffect {
public:
    explicit OpenDRTPlugin(OfxImageEffectHandle handle);

    void render(const OFX::RenderArguments& args) override;
    bool isIdentity(const OFX::IsIdentityArguments& args,
                    OFX::Clip*& identityClip, double& identityTime) override;

private:
    OpenDRTParams buildParams(double time, int width, int height) const;

    OFX::Clip* _srcClip = nullptr;
    OFX::Clip* _dstClip = nullptr;

    // Input / Output
    OFX::ChoiceParam*  _inGamut        = nullptr;
    OFX::ChoiceParam*  _inOetf         = nullptr;
    OFX::ChoiceParam*  _displayGamut   = nullptr;
    OFX::ChoiceParam*  _eotf           = nullptr;
    OFX::ChoiceParam*  _lookPreset     = nullptr;
    // Tonescale
    OFX::DoubleParam*  _tnLp           = nullptr;
    OFX::DoubleParam*  _tnGb           = nullptr;
    OFX::DoubleParam*  _ptHdr          = nullptr;
    // Basic contrast
    OFX::BooleanParam* _clamp          = nullptr;
    OFX::DoubleParam*  _tnLg           = nullptr;
    OFX::DoubleParam*  _tnCon          = nullptr;
    OFX::DoubleParam*  _tnSh           = nullptr;
    OFX::DoubleParam*  _tnToe          = nullptr;
    OFX::DoubleParam*  _tnOff          = nullptr;
    // High contrast
    OFX::BooleanParam* _tnHconEnable   = nullptr;
    OFX::DoubleParam*  _tnHcon         = nullptr;
    OFX::DoubleParam*  _tnHconPv       = nullptr;
    OFX::DoubleParam*  _tnHconSt       = nullptr;
    // Low contrast
    OFX::BooleanParam* _tnLconEnable   = nullptr;
    OFX::DoubleParam*  _tnLcon         = nullptr;
    OFX::DoubleParam*  _tnLconW        = nullptr;
    OFX::DoubleParam*  _tnLconPc       = nullptr;
    // Creative white + render space
    OFX::ChoiceParam*  _cwp            = nullptr;
    OFX::DoubleParam*  _cwpRng         = nullptr;
    OFX::DoubleParam*  _rsSa           = nullptr;
    OFX::DoubleParam*  _rsRw           = nullptr;
    OFX::DoubleParam*  _rsBw           = nullptr;
    // Purity compress low
    OFX::BooleanParam* _ptlEnable      = nullptr;
    OFX::DoubleParam*  _ptR            = nullptr;
    OFX::DoubleParam*  _ptG            = nullptr;
    OFX::DoubleParam*  _ptB            = nullptr;
    OFX::DoubleParam*  _ptRngLow       = nullptr;
    OFX::DoubleParam*  _ptRngHigh      = nullptr;
    // Mid purity
    OFX::BooleanParam* _ptmEnable      = nullptr;
    OFX::DoubleParam*  _ptmLow         = nullptr;
    OFX::DoubleParam*  _ptmLowSt       = nullptr;
    OFX::DoubleParam*  _ptmHigh        = nullptr;
    OFX::DoubleParam*  _ptmHighSt      = nullptr;
    // Brilliance
    OFX::BooleanParam* _brlEnable      = nullptr;
    OFX::DoubleParam*  _brlR           = nullptr;
    OFX::DoubleParam*  _brlG           = nullptr;
    OFX::DoubleParam*  _brlB           = nullptr;
    OFX::DoubleParam*  _brlC           = nullptr;
    OFX::DoubleParam*  _brlM           = nullptr;
    OFX::DoubleParam*  _brlY           = nullptr;
    OFX::DoubleParam*  _brlRng         = nullptr;
    // Hueshift RGB
    OFX::BooleanParam* _hsRgbEnable    = nullptr;
    OFX::DoubleParam*  _hsR            = nullptr;
    OFX::DoubleParam*  _hsG            = nullptr;
    OFX::DoubleParam*  _hsB            = nullptr;
    OFX::DoubleParam*  _hsRgbRng       = nullptr;
    // Hueshift CMY
    OFX::BooleanParam* _hsCmyEnable    = nullptr;
    OFX::DoubleParam*  _hsC            = nullptr;
    OFX::DoubleParam*  _hsM            = nullptr;
    OFX::DoubleParam*  _hsY            = nullptr;
    // Hue contrast
    OFX::BooleanParam* _hcEnable       = nullptr;
    OFX::DoubleParam*  _hcR            = nullptr;
    // Advanced hue contrast
    OFX::BooleanParam* _advHueContrast = nullptr;
    OFX::DoubleParam*  _advHcR         = nullptr;
    OFX::DoubleParam*  _advHcG         = nullptr;
    OFX::DoubleParam*  _advHcB         = nullptr;
    OFX::DoubleParam*  _advHcC         = nullptr;
    OFX::DoubleParam*  _advHcM         = nullptr;
    OFX::DoubleParam*  _advHcY         = nullptr;
    OFX::DoubleParam*  _advHcPower     = nullptr;
    // Beta features + diagnostics
    OFX::BooleanParam* _betaFeatures   = nullptr;
    OFX::BooleanParam* _filmicMode     = nullptr;
    OFX::DoubleParam*  _filmicSrcStops = nullptr;
    OFX::DoubleParam*  _filmicTgtStops = nullptr;
    OFX::DoubleParam*  _filmicDynRange = nullptr;
    OFX::DoubleParam*  _filmicStrength = nullptr;
    OFX::ChoiceParam*  _filmicProjSim  = nullptr;
    OFX::BooleanParam* _tonescaleMap   = nullptr;
    OFX::BooleanParam* _diagnosticsMode= nullptr;
    OFX::BooleanParam* _rgbChips       = nullptr;
};

// ── OpenDRTFactory ────────────────────────────────────────────────────────────
// The plugin identifier differs between the standalone (free) build and the
// Look Dev Tools suite build so the two co-install in Resolve without dedup.
#ifdef LDT_OPENDRT_STANDALONE
  #define OPENDRT_PLUGIN_ID "com.dec18studios.OpenDRT"
#else
  #define OPENDRT_PLUGIN_ID "com.dec18studios.lookdev.openDRT"
#endif

class OpenDRTFactory : public OFX::PluginFactoryHelper<OpenDRTFactory> {
public:
    OpenDRTFactory()
        : OFX::PluginFactoryHelper<OpenDRTFactory>(OPENDRT_PLUGIN_ID, 1, 0) {}

    void describe(OFX::ImageEffectDescriptor& desc) override;
    void describeInContext(OFX::ImageEffectDescriptor& desc,
                           OFX::ContextEnum ctx) override;
    OFX::ImageEffect* createInstance(OfxImageEffectHandle handle,
                                     OFX::ContextEnum ctx) override;
};
