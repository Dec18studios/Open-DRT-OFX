#pragma once
// OpenDRT.h - "6A. OpenDRT" OFX param names + class declarations.
//
// kODT_* are the OFX parameter names seen by the host; they mirror the canonical
// OpenDRT names (in_gamut, _tn_con, ...) so a project authored against canonical
// OpenDRT round-trips. They must never change once shipped. OpenDRTProcessor is
// the CPU worker. OpenDRTPlugin owns the OFX param objects and dispatches render
// to CPU/CUDA/Metal. OpenDRTFactory registers the plugin. The per-pixel math and
// host precompute live in core/OpenDRTAlgorithm.h (bit-exact vs the Metal golden).

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"

#include "core/OpenDRTAlgorithm.h"
#include "core/OpenDRTv11Algorithm.h"
#include "OpenDRTv11Runners.h"
#include "OpenDRTv11HostTable.h"
#include "AltDrtRunners.h"
#include "TesseractHostTable.h"
#include "ChromagnonHostTable.h"
#include "LdtControl.h"
#include "LdtFactorySettings.h"
#include "LdtPresetStore.h"
#include "OpenDRTPresetModels.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

// -- GPU back-end host entry points --------------------------------------------
#ifdef LDT_HAS_CUDA
void RunOpenDRTCudaKernel(void* p_Stream, int p_Width, int p_Height,
                          int p_SrcRowBytes, int p_DstRowBytes,
                          const OpenDRTParams* p_Params,
                          const float* p_Input, float* p_Output);
#endif

#ifdef LDT_HAS_OPENCL
// OpenCL (Windows / Linux; p_CmdQ is a cl_command_queue, p_Input / p_Output
// are cl_mem handles from getPixelData). p_SrcOffX / p_SrcOffY place the
// render rect inside the source (dst bounds minus src bounds), p_SrcW / p_SrcH
// give the source size (-1: covers the rect at offset 0). Returns false only
// when neither the kernel nor the CPU fallback could render.
bool RunOpenDRTOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height,
                            int p_SrcRowBytes, int p_DstRowBytes,
                            const OpenDRTParams* p_Params,
                            const float* p_Input, float* p_Output,
                            int p_SrcOffX = 0, int p_SrcOffY = 0,
                            int p_SrcW = -1, int p_SrcH = -1);
#endif

#ifdef __APPLE__
void RunOpenDRTMetalKernel(void* p_CmdQ, int p_Width, int p_Height,
                           int p_SrcRowBytes, int p_DstRowBytes,
                           const OpenDRTParams* p_Params,
                           const float* p_Input, float* p_Output);
#endif

// -- OFX parameter name constants (canonical OpenDRT names; never change) -------
// Model choice: 0 Current, 1 v1.0.1, 2 v1.1, 3 Tesseract, 4 Chromagnon. The
// v1.1 params use the odt11_ prefix (OpenDRTv11HostTable.h, kOdt11Desc), the
// Tesseract params tess_ (TesseractHostTable.h, kTessDesc) and the Chromagnon
// params cmg_ (ChromagnonHostTable.h, kCmgDesc).
static const char* kODT_Model            = "odt_model";
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
static const char* kODT_TnHconOn         = "_tn_hcon_on";
static const char* kODT_TnHcon           = "_tn_hcon";
static const char* kODT_TnHconPv         = "_tn_hcon_pv";
static const char* kODT_TnHconSt         = "_tn_hcon_st";
// Low contrast
static const char* kODT_TnLconEnable     = "_tn_lcon_enable";
static const char* kODT_TnLconOn         = "_tn_lcon_on";
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
static const char* kODT_PtlOn            = "_ptl_on";
static const char* kODT_PtR              = "_pt_r";
static const char* kODT_PtG              = "_pt_g";
static const char* kODT_PtB              = "_pt_b";
static const char* kODT_PtRngLow         = "_pt_rng_low";
static const char* kODT_PtRngHigh        = "_pt_rng_high";
// Mid purity
static const char* kODT_PtmEnable        = "_ptm_enable";
static const char* kODT_PtmOn            = "_ptm_on";
static const char* kODT_PtmLow           = "_ptm_low";
static const char* kODT_PtmLowSt         = "_ptm_low_st";
static const char* kODT_PtmHigh          = "_ptm_high";
static const char* kODT_PtmHighSt        = "_ptm_high_st";
// Brilliance
static const char* kODT_BrlEnable        = "_brl_enable";
static const char* kODT_BrlOn            = "_brl_on";
static const char* kODT_BrlR             = "_brl_r";
static const char* kODT_BrlG             = "_brl_g";
static const char* kODT_BrlB             = "_brl_b";
static const char* kODT_BrlC             = "_brl_c";
static const char* kODT_BrlM             = "_brl_m";
static const char* kODT_BrlY             = "_brl_y";
static const char* kODT_BrlRng           = "_brl_rng";
// Hueshift RGB
static const char* kODT_HsRgbEnable      = "_hs_rgb_enable";
static const char* kODT_HsRgbOn          = "_hs_rgb_on";
static const char* kODT_HsR              = "_hs_r";
static const char* kODT_HsG              = "_hs_g";
static const char* kODT_HsB              = "_hs_b";
static const char* kODT_HsRgbRng         = "_hs_rgb_rng";
// Hueshift CMY
static const char* kODT_HsCmyEnable      = "_hs_cmy_enable";
static const char* kODT_HsCmyOn          = "_hs_cmy_on";
static const char* kODT_HsC              = "_hs_c";
static const char* kODT_HsM              = "_hs_m";
static const char* kODT_HsY              = "_hs_y";
// Hue contrast
static const char* kODT_HcEnable         = "_hc_enable";
static const char* kODT_HcOn             = "_hc_on";
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
// Tonescale preset + per-module preset locks (canonical OpenDRT "Automatic Presets")
static const char* kODT_TonescalePreset  = "tonescale_preset";
static const char* kODT_LockStickshift   = "lockStickshift";
static const char* kODT_LockHcon         = "_lock_hcon";
static const char* kODT_LockLcon         = "_lock_lcon";
static const char* kODT_LockPtl          = "_lock_ptl";
static const char* kODT_LockPtm          = "_lock_ptm";
static const char* kODT_LockBrl          = "_lock_brl";
static const char* kODT_LockHsRgb        = "_lock_hs_rgb";
static const char* kODT_LockHsCmy        = "_lock_hs_cmy";
static const char* kODT_LockHc           = "_lock_hc";
// My Presets (user preset manager). Defined after every other tool param and
// before Factory Settings. UI state only: never in a preset, never in a
// starting point, never on the control whitelist, never read by the render.
static const char* kODT_MyPreset         = "my_preset";
static const char* kODT_PmSelected       = "preset_manager_selected";
static const char* kODT_PmGroup          = "PresetManagerGroup";
static const char* kODT_PmFamily         = "preset_manager_family";
static const char* kODT_PmName           = "preset_manager_name";
static const char* kODT_PmSaveNew        = "preset_manager_save_new";
static const char* kODT_PmUpdate         = "preset_manager_update_current";
static const char* kODT_PmRename         = "preset_manager_rename_current";
static const char* kODT_PmDelete         = "preset_manager_delete_current";
static const char* kODT_PmExport         = "preset_manager_export";
static const char* kODT_PmImport         = "preset_manager_import";
static const char* kODT_PmShowFolder     = "preset_manager_show_folder";
static const char* kODT_PmPrefix         = "preset_manager_";

// -- Claude control channel whitelist ------------------------------------------
// The params a remote controller may drive, built from the kODT_* constants
// above so the JSON keys can never drift from the OFX names. Every min/max here
// mirrors the setRange() in describeInContext; Choice entries are 0..count-1.
// Diagnostics (_diagnostics_mode, _rgbchips) are deliberately NOT controllable.
static const ldt::control::ParamSpec kODT_ControlSpecs[] = {
    // Model choice (v1.1, tess_ and cmg_ params are not on the whitelist this pass)
    { kODT_Model,          ldt::control::ParamSpec::Choice, 0.0,   4.0  },
    // Input / output
    { kODT_InGamut,        ldt::control::ParamSpec::Choice, 0.0,  15.0  },
    { kODT_InOetf,         ldt::control::ParamSpec::Choice, 0.0,   8.0  },
    { kODT_DisplayGamut,   ldt::control::ParamSpec::Choice, 0.0,   2.0  },
    { kODT_Eotf,           ldt::control::ParamSpec::Choice, 0.0,   5.0  },
    // Presets + per-module preset locks
    { kODT_LookPreset,      ldt::control::ParamSpec::Choice, 0.0,  3.0  },
    { kODT_TonescalePreset, ldt::control::ParamSpec::Choice, 0.0,  9.0  },
    { kODT_LockStickshift,  ldt::control::ParamSpec::Bool,   0.0,  1.0  },
    { kODT_LockHcon,        ldt::control::ParamSpec::Bool,   0.0,  1.0  },
    { kODT_LockLcon,        ldt::control::ParamSpec::Bool,   0.0,  1.0  },
    { kODT_LockPtl,         ldt::control::ParamSpec::Bool,   0.0,  1.0  },
    { kODT_LockPtm,         ldt::control::ParamSpec::Bool,   0.0,  1.0  },
    { kODT_LockBrl,         ldt::control::ParamSpec::Bool,   0.0,  1.0  },
    { kODT_LockHsRgb,       ldt::control::ParamSpec::Bool,   0.0,  1.0  },
    { kODT_LockHsCmy,       ldt::control::ParamSpec::Bool,   0.0,  1.0  },
    { kODT_LockHc,          ldt::control::ParamSpec::Bool,   0.0,  1.0  },
    // Tonescale
    { kODT_TnLp,           ldt::control::ParamSpec::Double, 100.0, 1000.0 },
    { kODT_TnGb,           ldt::control::ParamSpec::Double,   0.0,    1.0 },
    { kODT_PtHdr,          ldt::control::ParamSpec::Double,   0.0,    1.0 },
    // Basic contrast
    { kODT_Clamp,          ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_TnLg,           ldt::control::ParamSpec::Double,   4.0,   25.0 },
    { kODT_TnCon,          ldt::control::ParamSpec::Double,   1.0,    2.0 },
    { kODT_TnSh,           ldt::control::ParamSpec::Double,   0.0,    1.0 },
    { kODT_TnToe,          ldt::control::ParamSpec::Double,   0.0,    0.1 },
    { kODT_TnOff,          ldt::control::ParamSpec::Double,   0.0,    0.02 },
    // High contrast
    { kODT_TnHconEnable,   ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_TnHcon,         ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    { kODT_TnHconPv,       ldt::control::ParamSpec::Double,   0.0,    4.0 },
    { kODT_TnHconSt,       ldt::control::ParamSpec::Double,   0.0,    4.0 },
    // Low contrast
    { kODT_TnLconEnable,   ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_TnLcon,         ldt::control::ParamSpec::Double,   0.0,    3.0 },
    { kODT_TnLconW,        ldt::control::ParamSpec::Double,   0.0,    2.0 },
    { kODT_TnLconPc,       ldt::control::ParamSpec::Double,   0.0,    1.0 },
    // Creative white + render space
    { kODT_Cwp,            ldt::control::ParamSpec::Choice,   0.0,    4.0 },
    { kODT_CwpRng,         ldt::control::ParamSpec::Double,   0.0,    1.0 },
    { kODT_RsSa,           ldt::control::ParamSpec::Double,   0.0,    0.6 },
    { kODT_RsRw,           ldt::control::ParamSpec::Double,   0.0,    0.8 },
    { kODT_RsBw,           ldt::control::ParamSpec::Double,   0.0,    0.8 },
    // Purity compress low
    { kODT_PtlEnable,      ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_PtR,            ldt::control::ParamSpec::Double,   0.0,    4.0 },
    { kODT_PtG,            ldt::control::ParamSpec::Double,   0.0,    4.0 },
    { kODT_PtB,            ldt::control::ParamSpec::Double,   0.0,    4.0 },
    { kODT_PtRngLow,       ldt::control::ParamSpec::Double,   0.1,    0.6 },
    { kODT_PtRngHigh,      ldt::control::ParamSpec::Double,   0.25,   2.0 },
    // Mid purity
    { kODT_PtmEnable,      ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_PtmLow,         ldt::control::ParamSpec::Double,   0.0,    1.0 },
    { kODT_PtmLowSt,       ldt::control::ParamSpec::Double,   0.1,    1.0 },
    { kODT_PtmHigh,        ldt::control::ParamSpec::Double,  -0.9,    0.0 },
    { kODT_PtmHighSt,      ldt::control::ParamSpec::Double,   0.2,    1.0 },
    // Brilliance
    { kODT_BrlEnable,      ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_BrlR,           ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    { kODT_BrlG,           ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    { kODT_BrlB,           ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    { kODT_BrlC,           ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    { kODT_BrlM,           ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    { kODT_BrlY,           ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    { kODT_BrlRng,         ldt::control::ParamSpec::Double,   0.0,    2.0 },
    // Hueshift RGB
    { kODT_HsRgbEnable,    ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_HsR,            ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    { kODT_HsG,            ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    { kODT_HsB,            ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    { kODT_HsRgbRng,       ldt::control::ParamSpec::Double,   0.0,    2.0 },
    // Hueshift CMY
    { kODT_HsCmyEnable,    ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_HsC,            ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    { kODT_HsM,            ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    { kODT_HsY,            ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    // Hue contrast
    { kODT_HcEnable,       ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_HcR,            ldt::control::ParamSpec::Double,  -1.0,    1.0 },
    // Advanced hue contrast (beta-gated)
    { kODT_AdvHueContrast, ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_AdvHcR,         ldt::control::ParamSpec::Double,   0.1,    4.0 },
    { kODT_AdvHcG,         ldt::control::ParamSpec::Double,   0.1,    4.0 },
    { kODT_AdvHcB,         ldt::control::ParamSpec::Double,   0.1,    4.0 },
    { kODT_AdvHcC,         ldt::control::ParamSpec::Double,   0.1,    4.0 },
    { kODT_AdvHcM,         ldt::control::ParamSpec::Double,   0.1,    4.0 },
    { kODT_AdvHcY,         ldt::control::ParamSpec::Double,   0.1,    4.0 },
    { kODT_AdvHcPower,     ldt::control::ParamSpec::Double,   0.0,    1.0 },
    // Beta features + filmic (beta-gated)
    { kODT_BetaFeatures,   ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_FilmicMode,     ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_FilmicSrcStops, ldt::control::ParamSpec::Double,   1.0,   20.0 },
    { kODT_FilmicTgtStops, ldt::control::ParamSpec::Double,   1.0,   20.0 },
    { kODT_FilmicDynRange, ldt::control::ParamSpec::Double,   1.0,   10.0 },
    { kODT_FilmicStrength, ldt::control::ParamSpec::Double,   0.0,    1.0 },
    { kODT_FilmicProjSim,  ldt::control::ParamSpec::Choice,   0.0,    3.0 },
    // Tonescale curve overlay (the one diagnostic that is a look-neutral aid)
    { kODT_TonescaleMap,   ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    // Module quick on/off switches (appended)
    { kODT_TnHconOn,      ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_TnLconOn,      ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_PtlOn,         ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_PtmOn,         ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_BrlOn,         ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_HsRgbOn,       ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_HsCmyOn,       ldt::control::ParamSpec::Bool,     0.0,    1.0 },
    { kODT_HcOn,          ldt::control::ParamSpec::Bool,     0.0,    1.0 },
};
static const size_t kODT_ControlSpecCount =
    sizeof(kODT_ControlSpecs) / sizeof(kODT_ControlSpecs[0]);

// -- OpenDRTProcessor (CPU path) -----------------------------------------------
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

// CPU worker for the v1.1 model (Model choice index 2).
class OpenDRTv11Processor : public OFX::ImageProcessor {
public:
    explicit OpenDRTv11Processor(OFX::ImageEffect& effect)
        : OFX::ImageProcessor(effect), _srcImg(nullptr) {}

    void setParams(const OpenDRTv11Params& p) { _params = p; }
    void setSrcImg(OFX::Image* v)            { _srcImg = v; }

    void multiThreadProcessImages(OfxRectI window) override;

private:
    OpenDRTv11Params _params;
    OFX::Image*      _srcImg;
};

// CPU worker for the Tesseract model (Model choice index 3).
class TesseractProcessor : public OFX::ImageProcessor {
public:
    explicit TesseractProcessor(OFX::ImageEffect& effect)
        : OFX::ImageProcessor(effect), _srcImg(nullptr) {}

    void setParams(const TesseractParams& p) { _params = p; }
    void setSrcImg(OFX::Image* v)           { _srcImg = v; }

    void multiThreadProcessImages(OfxRectI window) override;

private:
    TesseractParams _params;
    OFX::Image*     _srcImg;
};

// CPU worker for the Chromagnon model (Model choice index 4).
class ChromagnonProcessor : public OFX::ImageProcessor {
public:
    explicit ChromagnonProcessor(OFX::ImageEffect& effect)
        : OFX::ImageProcessor(effect), _srcImg(nullptr) {}

    void setParams(const ChromagnonParams& p) { _params = p; }
    void setSrcImg(OFX::Image* v)            { _srcImg = v; }

    void multiThreadProcessImages(OfxRectI window) override;

private:
    ChromagnonParams _params;
    OFX::Image*      _srcImg;
};

// -- OpenDRTPlugin -------------------------------------------------------------
class OpenDRTPlugin : public OFX::ImageEffect {
public:
    explicit OpenDRTPlugin(OfxImageEffectHandle handle);
    ~OpenDRTPlugin() override;

    void render(const OFX::RenderArguments& args) override;
    bool isIdentity(const OFX::IsIdentityArguments& args,
                    OFX::Clip*& identityClip, double& identityTime) override;
    void changedParam(const OFX::InstanceChangedArgs& args,
                      const std::string& paramName) override;
    // The user opened the inspector: latch project-loaded and run a deferred
    // my_preset index fix-up.
    void beginEdit() override;

private:
    // -- My Presets (user preset manager) --
    // One my_preset menu entry (option index = entry index + 1; 0 is None).
    struct PresetMenuEntry {
        ldt::presets::Preset preset;
        bool missing = false;    // the hidden ref matches nothing in the store
    };
    // Re-scan the store and rebuild the my_preset options. Only option-list
    // calls; the index fix-up runs now when `fixNow`, else it is deferred.
    void rebuildPresetMenu(double time, bool fixNow);
    void runPendingPresetFixup();
    void setMyPresetIndexQuietly(int index);
    // Menu entry the hidden ref points at, -1 when none.
    int  selectedEntryIndex() const;
    ldt::presets::SelectedRef readSelectedRef() const;
    void writeSelectedRef(const ldt::presets::SelectedRef& r);
    void updatePresetButtons();
    // Dispatch for my_preset and every preset_manager_* param. True when handled.
    bool handlePresetManager(const OFX::InstanceChangedArgs& args, const std::string& paramName);
    void onMyPresetPicked(double time);
    void onPresetSaveNew(double time);
    void onPresetUpdate(double time);
    void onPresetRename(double time);
    void onPresetDelete(double time);
    void onPresetExport(double time);
    void onPresetImport(double time);
    void onPresetShowFolder();
    // Live values of one model's preset table (bool 0/1, choice index).
    std::map<std::string, double> capturePresetLive(int model, double time);
    // Switch the model if needed and write the preset in one edit block.
    void applyUserPreset(const ldt::presets::Preset& p, double time);
    // A real (not missing) selected record, or nullptr after telling the user.
    const PresetMenuEntry* requireSelectedRecord();
    void presetMessage(const std::string& msg, bool warning = false);
    // Yes/No question. A host that cannot ask returns `failedMeans`.
    bool presetAsk(const std::string& msg, bool failedMeans);
    void setStringQuietly(OFX::StringParam* p, const std::string& v);
    void setChoiceQuietly(OFX::ChoiceParam* p, int v);

    OpenDRTParams buildParams(double time, int width, int height) const;
    // Hide/show the feature-detail groups based on their enable toggles + beta
    // gate, mirroring canonical OpenDRT's setEnabledness().
    void setEnabledness();
    // Write the selected Look/Tonescale preset's values into the sliders, gated
    // by the per-module preset locks. Mirrors canonical applyPresetValues().
    void applyPresetValues(const OFX::InstanceChangedArgs& args);
    // Model aware visibility: Current and v1.0.1 show today's UI (via
    // setEnabledness) and hide every other model's items; v1.1, Tesseract and
    // Chromagnon each show their own params plus the shared ones and hide
    // every Current-only item.
    void refreshVisibility();
    int  modelAt(double time) const;
    // v1.1: read every odt11_ value, then build + precompute the kernel params.
    OpenDRTv11Params buildV11Params(double time) const;
    void readV11Values(double time, double* v) const;
    // v1.1: write one preset's rows into the sliders (user edit only).
    void applyV11Rows(int which, double time);
    // Tesseract / Chromagnon: read every tess_ / cmg_ value, then build +
    // precompute the kernel params (no presets upstream, so no row writes).
    void readTessValues(double time, double* v) const;
    TesseractParams buildTessParams(double time) const;
    void readCmgValues(double time, double* v) const;
    ChromagnonParams buildCmgParams(double time) const;

    OFX::Clip* _srcClip = nullptr;
    OFX::Clip* _dstClip = nullptr;

    OFX::ChoiceParam*  _model          = nullptr;
    // v1.1 params, one per kOdt11Desc entry; only the pointer matching the
    // entry kind is set.
    OFX::DoubleParam*  _v11Double[ODT11_NDESC] = {};
    OFX::BooleanParam* _v11Bool[ODT11_NDESC]   = {};
    OFX::ChoiceParam*  _v11Choice[ODT11_NDESC] = {};
    OFX::GroupParam*   _v11Groups[ODT11_G_COUNT]   = {};
    // Tesseract and Chromagnon params, same layout as the v1.1 arrays.
    OFX::DoubleParam*  _tessDouble[TESS_NDESC] = {};
    OFX::BooleanParam* _tessBool[TESS_NDESC]   = {};
    OFX::ChoiceParam*  _tessChoice[TESS_NDESC] = {};
    OFX::GroupParam*   _tessGroups[TESS_G_COUNT] = {};
    OFX::DoubleParam*  _cmgDouble[CMG_NDESC] = {};
    OFX::BooleanParam* _cmgBool[CMG_NDESC]   = {};
    OFX::ChoiceParam*  _cmgChoice[CMG_NDESC] = {};
    OFX::GroupParam*   _cmgGroups[CMG_G_COUNT] = {};
    // Diagnostics group: shown for Current, v1.0.1 and v1.1 only.
    OFX::GroupParam*   _diagnosticsGroup = nullptr;
    // Current-only groups hidden on v1.1 (the feature groups below are also
    // driven by setEnabledness on Current / v1.0.1).
    OFX::GroupParam* _presetGroup       = nullptr;
    OFX::GroupParam* _stickshiftGroup   = nullptr;
    OFX::GroupParam* _tonescaleGroup    = nullptr;
    OFX::GroupParam* _contrastGroup     = nullptr;
    OFX::GroupParam* _colorGroup        = nullptr;
    OFX::GroupParam* _betaFeaturesGroup = nullptr;

    // Input / Output
    OFX::ChoiceParam*  _inGamut        = nullptr;
    OFX::ChoiceParam*  _inOetf         = nullptr;
    OFX::ChoiceParam*  _displayGamut   = nullptr;
    OFX::ChoiceParam*  _eotf           = nullptr;
    OFX::ChoiceParam*  _lookPreset     = nullptr;
    OFX::ChoiceParam*  _tonescalePreset= nullptr;
    // Per-module preset locks ("Persistent Values")
    OFX::BooleanParam* _lockStickshift = nullptr;
    OFX::BooleanParam* _lockHcon       = nullptr;
    OFX::BooleanParam* _lockLcon       = nullptr;
    OFX::BooleanParam* _lockPtl        = nullptr;
    OFX::BooleanParam* _lockPtm        = nullptr;
    OFX::BooleanParam* _lockBrl        = nullptr;
    OFX::BooleanParam* _lockHsRgb      = nullptr;
    OFX::BooleanParam* _lockHsCmy      = nullptr;
    OFX::BooleanParam* _lockHc         = nullptr;
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
    OFX::BooleanParam* _tnHconOn      = nullptr;
    OFX::DoubleParam*  _tnHcon         = nullptr;
    OFX::DoubleParam*  _tnHconPv       = nullptr;
    OFX::DoubleParam*  _tnHconSt       = nullptr;
    // Low contrast
    OFX::BooleanParam* _tnLconEnable   = nullptr;
    OFX::BooleanParam* _tnLconOn      = nullptr;
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
    OFX::BooleanParam* _ptlOn         = nullptr;
    OFX::DoubleParam*  _ptR            = nullptr;
    OFX::DoubleParam*  _ptG            = nullptr;
    OFX::DoubleParam*  _ptB            = nullptr;
    OFX::DoubleParam*  _ptRngLow       = nullptr;
    OFX::DoubleParam*  _ptRngHigh      = nullptr;
    // Mid purity
    OFX::BooleanParam* _ptmEnable      = nullptr;
    OFX::BooleanParam* _ptmOn         = nullptr;
    OFX::DoubleParam*  _ptmLow         = nullptr;
    OFX::DoubleParam*  _ptmLowSt       = nullptr;
    OFX::DoubleParam*  _ptmHigh        = nullptr;
    OFX::DoubleParam*  _ptmHighSt      = nullptr;
    // Brilliance
    OFX::BooleanParam* _brlEnable      = nullptr;
    OFX::BooleanParam* _brlOn         = nullptr;
    OFX::DoubleParam*  _brlR           = nullptr;
    OFX::DoubleParam*  _brlG           = nullptr;
    OFX::DoubleParam*  _brlB           = nullptr;
    OFX::DoubleParam*  _brlC           = nullptr;
    OFX::DoubleParam*  _brlM           = nullptr;
    OFX::DoubleParam*  _brlY           = nullptr;
    OFX::DoubleParam*  _brlRng         = nullptr;
    // Hueshift RGB
    OFX::BooleanParam* _hsRgbEnable    = nullptr;
    OFX::BooleanParam* _hsRgbOn       = nullptr;
    OFX::DoubleParam*  _hsR            = nullptr;
    OFX::DoubleParam*  _hsG            = nullptr;
    OFX::DoubleParam*  _hsB            = nullptr;
    OFX::DoubleParam*  _hsRgbRng       = nullptr;
    // Hueshift CMY
    OFX::BooleanParam* _hsCmyEnable    = nullptr;
    OFX::BooleanParam* _hsCmyOn       = nullptr;
    OFX::DoubleParam*  _hsC            = nullptr;
    OFX::DoubleParam*  _hsM            = nullptr;
    OFX::DoubleParam*  _hsY            = nullptr;
    // Hue contrast
    OFX::BooleanParam* _hcEnable       = nullptr;
    OFX::BooleanParam* _hcOn          = nullptr;
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

    // -- Feature-detail groups (toggled secret by setEnabledness) --
    OFX::GroupParam* _highContrastGroup      = nullptr;
    OFX::GroupParam* _lowContrastGroup       = nullptr;
    OFX::GroupParam* _purityLowGroup         = nullptr;
    OFX::GroupParam* _midPurityGroup         = nullptr;
    OFX::GroupParam* _brillianceGroup        = nullptr;
    OFX::GroupParam* _hueshiftRgbGroup       = nullptr;
    OFX::GroupParam* _hueshiftCmyGroup       = nullptr;
    OFX::GroupParam* _hueContrastGroup       = nullptr;
    OFX::GroupParam* _advHueContrastGroup    = nullptr;
    OFX::GroupParam* _filmicDynamicRangeGroup= nullptr;
    OFX::GroupParam* _filmicProjectorSimGroup= nullptr;

    // -- My Presets --
    OFX::ChoiceParam*     _myPreset     = nullptr;
    OFX::StringParam*     _pmSelected   = nullptr;
    OFX::ChoiceParam*     _pmFamily     = nullptr;
    OFX::StringParam*     _pmName       = nullptr;
    OFX::PushButtonParam* _pmUpdate     = nullptr;
    OFX::PushButtonParam* _pmRename     = nullptr;
    OFX::PushButtonParam* _pmDelete     = nullptr;
    std::vector<PresetMenuEntry> _presetMenu;
    int  _pendingMenuIndex = -1;   // deferred my_preset index fix-up, -1 none
    bool _projectLoaded = false;   // latched on the first user edit or beginEdit
    bool _isApplying    = false;   // set while the plugin writes params itself

    // Claude control channel (inert until ~/.ofx-lookdevtools/armed exists).
    std::unique_ptr<ldt::control::Channel> _control;
    // Factory Settings (user starting points). Built before _control binds.
    std::unique_ptr<ldt::factory::Controller> _factory;
};

// -- OpenDRTFactory ------------------------------------------------------------
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
