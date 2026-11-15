#pragma once
// OpenDRTPresetModels.h - host-free model tables for the OpenDRT preset
// manager (My Presets). No OFX includes: tools/control/ldt_preset_selftest.cpp
// builds this with plain clang++ next to shared/LdtPresetStore.cpp.
//
// A preset is saved for one model and only ever holds that model's own params:
//   index 0 Current    key "default"     Current-model ids (_tn_*, _pt_*, ...)
//   index 1 v1.0.1     key "v101"        the same ids as Current
//   index 2 v1.1       key "v11"         odt11_* ids from the v1.1 host table
//   index 3 Tesseract  key "tesseract"   tess_* ids from the Tesseract host table
//   index 4 Chromagnon key "chromagnon"  cmg_* ids from the Chromagnon host table
//
// Never in any table: input gamut and transfer, display gamut and EOTF,
// peak luminance and the HDR grey and purity controls, clamp, the
// diagnostics toggles, every lock, the control uuid, my_preset,
// preset_manager_* and factory_*. A preset is a look, not a delivery setup.
//
// The Current-model id strings are repeated here (OpenDRT.h pulls in OFX).
// The selftest reads OpenDRT.h and checks every one of them is a real id.

#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "LdtPresetStore.h"
#include "OpenDRTv11HostTable.h"
#include "TesseractHostTable.h"
#include "ChromagnonHostTable.h"

namespace ldt {
namespace opendrt_presets {

using ldt::presets::Family;
using ldt::presets::ParamDef;
using ldt::presets::ParamType;
using ldt::presets::kInLook;
using ldt::presets::kInTonescale;
using ldt::presets::kFullOnly;

static const int kModelCount = 5;

// Matches the odt_model choice order. Append-only.
inline const char* modelKeyFor(int index)
{
    static const char* const keys[kModelCount] = { "default", "v101", "v11", "tesseract", "chromagnon" };
    return (index >= 0 && index < kModelCount) ? keys[index] : nullptr;
}

inline const char* modelLabelFor(int index)
{
    static const char* const labels[kModelCount] = { "Current", "v1.0.1", "v1.1", "Tesseract", "Chromagnon" };
    return (index >= 0 && index < kModelCount) ? labels[index] : "";
}

// -1 when the key is not one of the five.
inline int modelIndexFor(const std::string& key)
{
    for (int i = 0; i < kModelCount; ++i)
        if (key == modelKeyFor(i)) return i;
    return -1;
}

inline const char* modelLabelForKey(const std::string& key)
{
    const int i = modelIndexFor(key);
    return i >= 0 ? modelLabelFor(i) : key.c_str();
}

namespace detail {

inline ParamDef dbl(const char* id, double def, double lo, double hi, unsigned fam)
{
    ParamDef d; d.id = id; d.type = ParamType::Double; d.def = def; d.lo = lo; d.hi = hi; d.families = fam;
    return d;
}
inline ParamDef flag(const char* id, bool def, unsigned fam)
{
    ParamDef d; d.id = id; d.type = ParamType::Bool; d.def = def ? 1.0 : 0.0; d.families = fam;
    return d;
}
inline ParamDef choice(const char* id, int def, std::vector<std::string> opts, unsigned fam)
{
    ParamDef d; d.id = id; d.type = ParamType::Choice; d.def = def; d.options = std::move(opts);
    d.lo = 0.0; d.hi = (double)d.options.size() - 1.0; d.families = fam;
    return d;
}

// Shared shape of the three host table descriptors.
template <typename Desc>
ParamDef fromDesc(const Desc& d, double def, unsigned fam, int kindDouble, int kindBool)
{
    ParamDef p;
    p.id = d.name;
    p.def = def;
    p.families = fam;
    if (d.kind == kindDouble) {
        p.type = ParamType::Double;
        p.lo = d.mn; p.hi = d.mx;
    } else if (d.kind == kindBool) {
        p.type = ParamType::Bool;
    } else {
        p.type = ParamType::Choice;
        for (int o = 0; o < d.nopts; ++o) p.options.push_back(d.opts[o]);
        p.lo = 0.0; p.hi = (double)d.nopts - 1.0;
    }
    return p;
}

inline bool sameId(const char* a, const char* b) { return std::string(a) == b; }

} // namespace detail

// -- Current and v1.0.1 -------------------------------------------------------
// Design section 3.1. Table order is apply order.
inline const std::vector<ParamDef>& currentModelTable()
{
    using namespace detail;
    static const std::vector<ParamDef> t = [] {
        const unsigned L = kInLook, T = kInTonescale, F = kFullOnly;
        std::vector<ParamDef> v;
        // Look
        v.push_back(choice("look_preset", 0, { "Default", "Colorful", "Umbra", "Base" }, L));
        v.push_back(choice("_cwp", 4, { "D65", "D60", "D55", "D50", "Use Look Preset" }, L));
        v.push_back(dbl("_cwp_rng", 0.5, 0.0, 1.0, L));
        v.push_back(dbl("_rs_sa", 0.35, 0.0, 0.6, L));
        v.push_back(dbl("_rs_rw", 0.25, 0.0, 0.8, L));
        v.push_back(dbl("_rs_bw", 0.55, 0.0, 0.8, L));
        v.push_back(flag("_ptl_on", true, L));
        v.push_back(flag("_ptl_enable", false, L));
        v.push_back(dbl("_pt_r", 0.5, 0.0, 4.0, L));
        v.push_back(dbl("_pt_g", 2.0, 0.0, 4.0, L));
        v.push_back(dbl("_pt_b", 2.0, 0.0, 4.0, L));
        v.push_back(dbl("_pt_rng_low", 0.2, 0.1, 0.6, L));
        v.push_back(dbl("_pt_rng_high", 0.8, 0.25, 2.0, L));
        v.push_back(flag("_ptm_on", true, L));
        v.push_back(flag("_ptm_enable", false, L));
        v.push_back(dbl("_ptm_low", 0.2, 0.0, 1.0, L));
        v.push_back(dbl("_ptm_low_st", 0.5, 0.1, 1.0, L));
        v.push_back(dbl("_ptm_high", -0.8, -0.9, 0.0, L));
        v.push_back(dbl("_ptm_high_st", 0.3, 0.2, 1.0, L));
        v.push_back(flag("_brl_on", true, L));
        v.push_back(flag("_brl_enable", false, L));
        v.push_back(dbl("_brl_r", -0.5, -1.0, 1.0, L));
        v.push_back(dbl("_brl_g", -0.4, -1.0, 1.0, L));
        v.push_back(dbl("_brl_b", -0.2, -1.0, 1.0, L));
        v.push_back(dbl("_brl_c", 0.0, -1.0, 1.0, L));
        v.push_back(dbl("_brl_m", 0.0, -1.0, 1.0, L));
        v.push_back(dbl("_brl_y", 0.0, -1.0, 1.0, L));
        v.push_back(dbl("_brl_rng", 0.66, 0.0, 2.0, L));
        v.push_back(flag("_hs_rgb_on", true, L));
        v.push_back(flag("_hs_rgb_enable", false, L));
        v.push_back(dbl("_hs_r", 0.35, -1.0, 1.0, L));
        v.push_back(dbl("_hs_g", 0.25, -1.0, 1.0, L));
        v.push_back(dbl("_hs_b", 0.5, -1.0, 1.0, L));
        v.push_back(dbl("_hs_rgb_rng", 0.6, 0.0, 2.0, L));
        v.push_back(flag("_hs_cmy_on", true, L));
        v.push_back(flag("_hs_cmy_enable", false, L));
        v.push_back(dbl("_hs_c", 0.2, -1.0, 1.0, L));
        v.push_back(dbl("_hs_m", 0.2, -1.0, 1.0, L));
        v.push_back(dbl("_hs_y", 0.2, -1.0, 1.0, L));
        v.push_back(flag("_hc_on", true, L));
        v.push_back(flag("_hc_enable", false, L));
        v.push_back(dbl("_hc_r", 0.6, -1.0, 1.0, L));
        v.push_back(flag("_adv_hue_contrast", false, L));
        v.push_back(dbl("_adv_hc_r", 1.0, 0.1, 4.0, L));
        v.push_back(dbl("_adv_hc_g", 1.0, 0.1, 4.0, L));
        v.push_back(dbl("_adv_hc_b", 1.0, 0.1, 4.0, L));
        v.push_back(dbl("_adv_hc_c", 1.0, 0.1, 4.0, L));
        v.push_back(dbl("_adv_hc_m", 1.0, 0.1, 4.0, L));
        v.push_back(dbl("_adv_hc_y", 1.0, 0.1, 4.0, L));
        v.push_back(dbl("_adv_hc_power", 1.0, 0.0, 1.0, L));
        // Tonescale
        v.push_back(choice("tonescale_preset", 0, { "Use Look Preset", "High-Contrast", "Low-Contrast",
                                                    "ACES-1.x", "ACES-2.0", "Marvelous Tonescape",
                                                    "Arriba Tonecall", "DaGrinchi Tonegroan",
                                                    "Aery Tonescale", "Umbra Tonescale" }, T));
        v.push_back(dbl("_tn_Lg", 11.1, 4.0, 25.0, T));
        v.push_back(dbl("_tn_con", 1.4, 1.0, 2.0, T));
        v.push_back(dbl("_tn_sh", 0.5, 0.0, 1.0, T));
        v.push_back(dbl("_tn_toe", 0.003, 0.0, 0.1, T));
        v.push_back(dbl("_tn_off", 0.005, 0.0, 0.02, T));
        v.push_back(flag("_tn_hcon_on", true, T));
        v.push_back(flag("_tn_hcon_enable", false, T));
        v.push_back(dbl("_tn_hcon", 0.0, -1.0, 1.0, T));
        v.push_back(dbl("_tn_hcon_pv", 1.0, 0.0, 4.0, T));
        v.push_back(dbl("_tn_hcon_st", 4.0, 0.0, 4.0, T));
        v.push_back(flag("_tn_lcon_on", true, T));
        v.push_back(flag("_tn_lcon_enable", false, T));
        v.push_back(dbl("_tn_lcon", 1.0, 0.0, 3.0, T));
        v.push_back(dbl("_tn_lcon_w", 0.5, 0.0, 2.0, T));
        v.push_back(dbl("_tn_lcon_pc", 1.0, 0.0, 1.0, T));
        // Full only (beta block)
        v.push_back(flag("_beta_features_enable", false, F));
        v.push_back(flag("_filmic_mode", false, F));
        v.push_back(dbl("_filmic_source_stops", 14.0, 1.0, 20.0, F));
        v.push_back(dbl("_filmic_target_stops", 10.0, 1.0, 20.0, F));
        v.push_back(dbl("_filmic_dynamic_range", 5.0, 1.0, 10.0, F));
        v.push_back(dbl("_filmic_strength", 0.0, 0.0, 1.0, F));
        v.push_back(choice("_filmic_projector_sim", 0, { "None", "Xenon", "Tungsten", "LED" }, F));
        return v;
    }();
    return t;
}

// Current-model lock toggles and the params each one keeps. The master
// lockStickshift keeps every listed member. Apply leaves a locked member alone.
struct LockGroup { const char* lockId; std::vector<std::string> members; };

inline const char* masterLockId() { return "lockStickshift"; }

inline const std::vector<LockGroup>& currentModelLocks()
{
    static const std::vector<LockGroup> g = {
        { "_lock_hcon",   { "_tn_hcon_on", "_tn_hcon_enable", "_tn_hcon", "_tn_hcon_pv", "_tn_hcon_st" } },
        { "_lock_lcon",   { "_tn_lcon_on", "_tn_lcon_enable", "_tn_lcon", "_tn_lcon_w", "_tn_lcon_pc" } },
        { "_lock_ptl",    { "_ptl_on", "_ptl_enable", "_pt_r", "_pt_g", "_pt_b", "_pt_rng_low", "_pt_rng_high" } },
        { "_lock_ptm",    { "_ptm_on", "_ptm_enable", "_ptm_low", "_ptm_low_st", "_ptm_high", "_ptm_high_st" } },
        { "_lock_brl",    { "_brl_on", "_brl_enable", "_brl_r", "_brl_g", "_brl_b", "_brl_c", "_brl_m", "_brl_y", "_brl_rng" } },
        { "_lock_hs_rgb", { "_hs_rgb_on", "_hs_rgb_enable", "_hs_r", "_hs_g", "_hs_b", "_hs_rgb_rng" } },
        { "_lock_hs_cmy", { "_hs_cmy_on", "_hs_cmy_enable", "_hs_c", "_hs_m", "_hs_y" } },
        { "_lock_hc",     { "_hc_on", "_hc_enable", "_hc_r" } },
    };
    return g;
}

// The lock id that keeps `paramId`, or nullptr when no lock covers it.
inline const char* lockFor(const std::string& paramId)
{
    for (const auto& g : currentModelLocks())
        for (const auto& m : g.members)
            if (m == paramId) return g.lockId;
    return nullptr;
}

// -- v1.1 ---------------------------------------------------------------------
// Everything but the Input and Display groups and the peak / HDR / encoding
// entries of the Preset group. tn_Lg and tonescale_preset sit in the Preset
// group but belong to the tonescale.
inline const std::vector<ParamDef>& v11Table()
{
    using namespace detail;
    static const std::vector<ParamDef> t = [] {
        std::vector<ParamDef> v;
        double def[ODT11_NDESC];
        odt11_default_values(def);
        for (int i = 0; i < ODT11_NDESC; ++i) {
            const Odt11Desc& d = kOdt11Desc[i];
            if (d.group == ODT11_G_INPUT || d.group == ODT11_G_DISPLAY) continue;
            if (sameId(d.name, "odt11_tn_Lp") || sameId(d.name, "odt11_tn_gb") ||
                sameId(d.name, "odt11_pt_hdr") || sameId(d.name, "odt11_display_encoding_preset"))
                continue;
            unsigned fam = kInLook;
            if (d.group == ODT11_G_TONESCALE || sameId(d.name, "odt11_tn_Lg") ||
                sameId(d.name, "odt11_tonescale_preset"))
                fam = kInTonescale;
            v.push_back(fromDesc(d, def[i], fam, ODT11_DOUBLE, ODT11_BOOL));
        }
        return v;
    }();
    return t;
}

// -- Tesseract ----------------------------------------------------------------
inline const std::vector<ParamDef>& tesseractTable()
{
    using namespace detail;
    static const std::vector<ParamDef> t = [] {
        std::vector<ParamDef> v;
        double def[TESS_NDESC];
        tess_default_values(def);
        for (int i = 0; i < TESS_NDESC; ++i) {
            const TessDesc& d = kTessDesc[i];
            if (d.group == TESS_G_INPUT || d.group == TESS_G_DISPLAY) continue;
            if (sameId(d.name, "tess_Lp")) continue;
            const unsigned fam = (d.group == TESS_G_TONESCALE) ? kInTonescale : kInLook;
            v.push_back(fromDesc(d, def[i], fam, TESS_DOUBLE, TESS_BOOL));
        }
        return v;
    }();
    return t;
}

// -- Chromagnon ---------------------------------------------------------------
inline const std::vector<ParamDef>& chromagnonTable()
{
    using namespace detail;
    static const std::vector<ParamDef> t = [] {
        std::vector<ParamDef> v;
        double def[CMG_NDESC];
        cmg_default_values(def);
        for (int i = 0; i < CMG_NDESC; ++i) {
            const CmgDesc& d = kCmgDesc[i];
            if (d.group == CMG_G_INPUT || d.group == CMG_G_DISPLAY) continue;
            if (sameId(d.name, "cmg_Lp") || sameId(d.name, "cmg_Lgb")) continue;
            const unsigned fam = (d.group == CMG_G_TONESCALE) ? kInTonescale : kInLook;
            v.push_back(fromDesc(d, def[i], fam, CMG_DOUBLE, CMG_BOOL));
        }
        return v;
    }();
    return t;
}

// The table for a model index (0 and 1 share one). Empty for anything else.
inline const std::vector<ParamDef>& tableForModel(int index)
{
    static const std::vector<ParamDef> none;
    switch (index) {
        case 0: case 1: return currentModelTable();
        case 2: return v11Table();
        case 3: return tesseractTable();
        case 4: return chromagnonTable();
        default: return none;
    }
}

inline const std::vector<ParamDef>& tableForModelKey(const std::string& key)
{
    return tableForModel(modelIndexFor(key));
}

inline const ParamDef* findDef(const std::vector<ParamDef>& table, const std::string& id)
{
    for (const auto& d : table) if (d.id == id) return &d;
    return nullptr;
}

// Import validation run by the store before anything is written: no path
// pieces in the name, a model this build has, and only that model's ids.
inline bool checkImport(const ldt::presets::Preset& p, std::string& reason)
{
    const std::string& raw = p.sourceName;
    if (raw.find('/') != std::string::npos || raw.find('\\') != std::string::npos ||
        raw.find("..") != std::string::npos) {
        reason = "This preset file has a name that isn't allowed.";
        return false;
    }
    const int m = modelIndexFor(p.model);
    if (m < 0) {
        reason = "This preset needs a model this version doesn't have (" + p.model + ").";
        return false;
    }
    const std::vector<ParamDef>& table = tableForModel(m);
    if (p.values.size() > table.size()) {
        reason = "This preset file has more settings than OpenDRT knows.";
        return false;
    }
    for (const auto& kv : p.values) {
        if (!findDef(table, kv.first)) {
            reason = "This preset file has a setting OpenDRT doesn't know (" + kv.first + ").";
            return false;
        }
    }
    return true;
}

// The OpenDRT store config with all five model keys and the import check.
inline ldt::presets::StoreConfig modelConfig()
{
    ldt::presets::StoreConfig c = ldt::presets::openDrtConfig();
    c.knownModels.clear();
    for (int i = 0; i < kModelCount; ++i) c.knownModels.push_back(modelKeyFor(i));
    c.importCheck = &checkImport;
    return c;
}

// Menu text for a preset of another model: "<name> (<model label>)".
inline std::string crossModelLabel(const ldt::presets::Preset& p)
{
    return ldt::presets::menuLabel(p) + " (" + modelLabelForKey(p.model) + ")";
}

} // namespace opendrt_presets
} // namespace ldt
