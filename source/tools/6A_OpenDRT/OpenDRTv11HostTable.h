#pragma once
// OpenDRTv11HostTable.h - host side parameter table for the v1.1 OpenDRT model
// (Model choice index 2). OFX free, so the host self test in
// harness/opendrt_models/ can include it directly.
//
// Port of the upstream OpenDRT v1.1.0 (GPL-3.0) UI. Upstream exposes a preset
// UI (look, tonescale, creative white, display encoding) and a commented
// stickshift UI with every module slider. The plugin shows both at once:
//   - the sliders are the source of truth (odt11_build reads only the values)
//   - picking a preset writes that preset's rows into the sliders
//     (odt11_write_rows), so after a preset pick the sliders hold exactly what
//     upstream preset mode would compute.
//
// Values travel as one double per descriptor (the OFX value type). Float
// fields round trip float -> double -> float exactly, so a preset written to
// the sliders and read back yields the same bytes as opendrt_v11_fill_presets.
//
// Not exposed:
//   in_gamut, crv_enable  shared with the Current model (Input Gamut, Tonescale
//                         Curve), passed to odt11_build by the caller
//   pt_enable             upstream never reads it; derived from the look row
//   cwp, cwp_lm           derived exactly as upstream: from the look row, then
//                         the Creative White override (ui_cwp, ui_cwp_lm)

#include <cmath>
#include <cstddef>
#include <cstring>
#include "core/OpenDRTv11Params.h"
#include "core/OpenDRTv11Presets.h"

enum Odt11Kind { ODT11_DOUBLE = 0, ODT11_BOOL = 1, ODT11_CHOICE = 2 };

// Group slots. ODT11_G_INPUT is the shared Input group of the plugin.
enum Odt11Group {
    ODT11_G_INPUT = 0,
    ODT11_G_PRESET,
    ODT11_G_TONESCALE,
    ODT11_G_RENDER,
    ODT11_G_PURITY,
    ODT11_G_BRILLIANCE,
    ODT11_G_HUECON,
    ODT11_G_HUESHIFT,
    ODT11_G_DISPLAY,
    ODT11_G_COUNT
};

struct Odt11GroupInfo { const char* name; const char* label; bool open; };
static const Odt11GroupInfo kOdt11Groups[ODT11_G_COUNT] = {
    { "InputGroup",            "Input & Output Settings", false }, // existing group, not defined here
    { "ODT11PresetGroup",      "Presets",                 true  },
    { "ODT11TonescaleGroup",   "Tonescale",               false },
    { "ODT11RenderSpaceGroup", "Render Space",            false },
    { "ODT11PurityGroup",      "Purity",                  false },
    { "ODT11BrillianceGroup",  "Brilliance",              false },
    { "ODT11HueContrastGroup", "Hue Contrast",            false },
    { "ODT11HueShiftGroup",    "Hue Shift",               false },
    { "ODT11DisplayGroup",     "Display",                 false },
};

struct Odt11Desc {
    const char* name;      // OFX param id
    const char* label;
    const char* hint;
    int         group;     // Odt11Group
    int         kind;      // Odt11Kind; DOUBLE = float field, BOOL/CHOICE = int field
    double      mn, mx, inc;
    size_t      off;       // offsetof(OpenDRTv11Params, field)
    const char* const* opts;
    int         nopts;
};

// Choice labels (upstream combo labels; Input Transfer uses the Current
// model's labels plus the v1.1 F-Log2 entry).
static const char* const kOdt11InOetfLabels[10] = {
    "Linear", "Davinci Intermediate", "Filmlight T-Log", "ACEScct", "Arri LogC3",
    "Arri LogC4", "RedLog3G10", "Panasonic V-Log", "Sony S-Log3", "Fujifilm F-Log2"
};
static const char* const kOdt11EotfLabels[6] = {
    "Linear", "2.2 Power sRGB Display", "2.4 Power Rec.1886", "2.6 Power DCI",
    "ST 2084 PQ", "HLG"
};

#define ODT11_OFF(f) offsetof(OpenDRTv11Params, f)
#define ODT11_D(id, lab, hint, g, mn, mx, inc, f) \
    { "odt11_" id, lab, hint, g, ODT11_DOUBLE, mn, mx, inc, ODT11_OFF(f), 0, 0 }
#define ODT11_B(id, lab, hint, g, f) \
    { "odt11_" id, lab, hint, g, ODT11_BOOL, 0.0, 1.0, 1.0, ODT11_OFF(f), 0, 0 }
#define ODT11_C(id, lab, hint, g, f, opts, n) \
    { "odt11_" id, lab, hint, g, ODT11_CHOICE, 0.0, (double)((n) - 1), 1.0, ODT11_OFF(f), opts, n }

// Definition (and UI) order. Index constants below must follow this table.
static const Odt11Desc kOdt11Desc[] = {
    // Input (shared Input group)
    ODT11_C("in_oetf", "Input Transfer Function", "Input transfer function (OETF) for the v1.1 model.", ODT11_G_INPUT, in_oetf, kOdt11InOetfLabels, 10),

    // Presets and top-level controls
    ODT11_C("look_preset", "Look Preset", "Writes every module slider from the chosen look. The sliders stay editable.", ODT11_G_PRESET, look_preset, kOdt11LookNames, ODT11_LOOK_VISIBLE_COUNT),
    ODT11_C("tonescale_preset", "Tonescale Preset", "Writes the tonescale sliders. Use Look Preset takes them from the look.", ODT11_G_PRESET, tonescale_preset, kOdt11TonescaleNames, ODT11_TONESCALE_COUNT),
    ODT11_C("cwp", "Creative White", "Creative white point. Use Look Preset takes it from the look.", ODT11_G_PRESET, ui_cwp, kOdt11CwpUiNames, ODT11_CWP_UI_COUNT),
    ODT11_D("cwp_lm", "Creative White Limit", "Creative white limit, used when Creative White is not Use Look Preset.", ODT11_G_PRESET, 0.0, 1.0, 0.01, ui_cwp_lm),
    ODT11_C("display_encoding_preset", "Display Encoding Preset", "Writes Surround, Display Gamut and Display EOTF.", ODT11_G_PRESET, display_encoding_preset, kOdt11EncodingNames, ODT11_ENCODING_COUNT),
    ODT11_D("tn_Lp", "Display Peak Luminance", "Peak luminance of the display in nits.", ODT11_G_PRESET, 100.0, 1000.0, 1.0, tn_Lp),
    ODT11_D("tn_gb", "HDR Grey Boost", "Boost grey levels for HDR displays.", ODT11_G_PRESET, 0.0, 1.0, 0.001, tn_gb),
    ODT11_D("pt_hdr", "HDR Purity", "Purity adjustment for HDR.", ODT11_G_PRESET, 0.0, 1.0, 0.01, pt_hdr),
    ODT11_D("tn_Lg", "Display Grey Luminance", "Grey point luminance in nits.", ODT11_G_PRESET, 3.0, 25.0, 0.1, tn_Lg),

    // Tonescale
    ODT11_D("tn_con", "Contrast", "Tonescale contrast.", ODT11_G_TONESCALE, 1.0, 2.0, 0.01, tn_con),
    ODT11_D("tn_sh", "Shoulder Clip", "Highlight shoulder clip.", ODT11_G_TONESCALE, 0.0, 1.0, 0.01, tn_sh),
    ODT11_D("tn_toe", "Toe", "Shadow toe.", ODT11_G_TONESCALE, 0.0, 0.1, 0.001, tn_toe),
    ODT11_D("tn_off", "Offset", "Black point offset.", ODT11_G_TONESCALE, 0.0, 0.02, 0.0002, tn_off),
    ODT11_B("tn_hcon_enable", "Enable Contrast High", "Enable the high contrast adjustment.", ODT11_G_TONESCALE, tn_hcon_enable),
    ODT11_D("tn_hcon", "Contrast High", "High contrast amount.", ODT11_G_TONESCALE, -1.0, 1.0, 0.02, tn_hcon),
    ODT11_D("tn_hcon_pv", "Contrast High Pivot", "High contrast pivot.", ODT11_G_TONESCALE, 0.0, 4.0, 0.04, tn_hcon_pv),
    ODT11_D("tn_hcon_st", "Contrast High Strength", "High contrast strength.", ODT11_G_TONESCALE, 0.0, 4.0, 0.04, tn_hcon_st),
    ODT11_B("tn_lcon_enable", "Enable Contrast Low", "Enable the low contrast adjustment.", ODT11_G_TONESCALE, tn_lcon_enable),
    ODT11_D("tn_lcon", "Contrast Low", "Low contrast amount.", ODT11_G_TONESCALE, 0.0, 3.0, 0.03, tn_lcon),
    ODT11_D("tn_lcon_w", "Contrast Low Width", "Low contrast width.", ODT11_G_TONESCALE, 0.0, 2.0, 0.02, tn_lcon_w),

    // Render space
    ODT11_D("rs_sa", "Render Space Strength", "Render space strength.", ODT11_G_RENDER, 0.0, 0.6, 0.006, rs_sa),
    ODT11_D("rs_rw", "Render Space Weight R", "Render space red weight.", ODT11_G_RENDER, 0.0, 0.8, 0.008, rs_rw),
    ODT11_D("rs_bw", "Render Space Weight B", "Render space blue weight.", ODT11_G_RENDER, 0.0, 0.8, 0.008, rs_bw),

    // Purity
    ODT11_D("pt_lml", "Purity Limit Low", "Purity limit low.", ODT11_G_PURITY, 0.0, 1.0, 0.01, pt_lml),
    ODT11_D("pt_lml_r", "Purity Limit Low R", "Purity limit low, red.", ODT11_G_PURITY, 0.0, 1.0, 0.01, pt_lml_r),
    ODT11_D("pt_lml_g", "Purity Limit Low G", "Purity limit low, green.", ODT11_G_PURITY, 0.0, 1.0, 0.01, pt_lml_g),
    ODT11_D("pt_lml_b", "Purity Limit Low B", "Purity limit low, blue.", ODT11_G_PURITY, 0.0, 1.0, 0.01, pt_lml_b),
    ODT11_D("pt_lmh", "Purity Limit High", "Purity limit high.", ODT11_G_PURITY, 0.0, 1.0, 0.01, pt_lmh),
    ODT11_D("pt_lmh_r", "Purity Limit High R", "Purity limit high, red.", ODT11_G_PURITY, 0.0, 1.0, 0.01, pt_lmh_r),
    ODT11_D("pt_lmh_b", "Purity Limit High B", "Purity limit high, blue.", ODT11_G_PURITY, 0.0, 1.0, 0.01, pt_lmh_b),
    ODT11_B("ptl_enable", "Enable Purity Softclip", "Enable the purity softclip.", ODT11_G_PURITY, ptl_enable),
    ODT11_D("ptl_c", "Purity Softclip C", "Purity softclip, cyan.", ODT11_G_PURITY, 0.0, 0.25, 0.0025, ptl_c),
    ODT11_D("ptl_m", "Purity Softclip M", "Purity softclip, magenta.", ODT11_G_PURITY, 0.0, 0.25, 0.0025, ptl_m),
    ODT11_D("ptl_y", "Purity Softclip Y", "Purity softclip, yellow.", ODT11_G_PURITY, 0.0, 0.25, 0.0025, ptl_y),
    ODT11_B("ptm_enable", "Enable Mid Purity", "Enable mid purity.", ODT11_G_PURITY, ptm_enable),
    ODT11_D("ptm_low", "Mid Purity Low", "Mid purity low.", ODT11_G_PURITY, 0.0, 2.0, 0.02, ptm_low),
    ODT11_D("ptm_low_rng", "Mid Purity Low Range", "Mid purity low range.", ODT11_G_PURITY, 0.0, 1.0, 0.01, ptm_low_rng),
    ODT11_D("ptm_low_st", "Mid Purity Low Strength", "Mid purity low strength.", ODT11_G_PURITY, 0.1, 1.0, 0.01, ptm_low_st),
    ODT11_D("ptm_high", "Mid Purity High", "Mid purity high.", ODT11_G_PURITY, -0.9, 0.0, 0.01, ptm_high),
    ODT11_D("ptm_high_rng", "Mid Purity High Range", "Mid purity high range.", ODT11_G_PURITY, 0.0, 1.0, 0.01, ptm_high_rng),
    ODT11_D("ptm_high_st", "Mid Purity High Strength", "Mid purity high strength.", ODT11_G_PURITY, 0.1, 1.0, 0.01, ptm_high_st),

    // Brilliance
    ODT11_B("brl_enable", "Enable Brilliance", "Enable brilliance.", ODT11_G_BRILLIANCE, brl_enable),
    ODT11_D("brl", "Brilliance", "Brilliance.", ODT11_G_BRILLIANCE, -6.0, 2.0, 0.08, brl),
    ODT11_D("brl_r", "Brilliance R", "Brilliance, red.", ODT11_G_BRILLIANCE, -6.0, 2.0, 0.08, brl_r),
    ODT11_D("brl_g", "Brilliance G", "Brilliance, green.", ODT11_G_BRILLIANCE, -6.0, 2.0, 0.08, brl_g),
    ODT11_D("brl_b", "Brilliance B", "Brilliance, blue.", ODT11_G_BRILLIANCE, -6.0, 2.0, 0.08, brl_b),
    ODT11_D("brl_rng", "Brilliance Range", "Brilliance range.", ODT11_G_BRILLIANCE, 0.0, 1.0, 0.01, brl_rng),
    ODT11_D("brl_st", "Brilliance Strength", "Brilliance strength.", ODT11_G_BRILLIANCE, 0.0, 1.0, 0.01, brl_st),
    ODT11_B("brlp_enable", "Enable Post Brilliance", "Enable post brilliance.", ODT11_G_BRILLIANCE, brlp_enable),
    ODT11_D("brlp", "Brilliance Post", "Post brilliance.", ODT11_G_BRILLIANCE, -1.0, 0.0, 0.01, brlp),
    ODT11_D("brlp_r", "Post Brilliance R", "Post brilliance, red.", ODT11_G_BRILLIANCE, -3.0, 0.0, 0.03, brlp_r),
    ODT11_D("brlp_g", "Post Brilliance G", "Post brilliance, green.", ODT11_G_BRILLIANCE, -3.0, 0.0, 0.03, brlp_g),
    ODT11_D("brlp_b", "Post Brilliance B", "Post brilliance, blue.", ODT11_G_BRILLIANCE, -3.0, 0.0, 0.03, brlp_b),

    // Hue contrast
    ODT11_B("hc_enable", "Enable Hue Contrast", "Enable hue contrast.", ODT11_G_HUECON, hc_enable),
    ODT11_D("hc_r", "Hue Contrast R", "Hue contrast, red.", ODT11_G_HUECON, 0.0, 2.0, 0.02, hc_r),
    ODT11_D("hc_r_rng", "Hue Contrast R Range", "Hue contrast red range.", ODT11_G_HUECON, 0.0, 1.0, 0.01, hc_r_rng),

    // Hue shift
    ODT11_B("hs_rgb_enable", "Enable Hueshift RGB", "Enable the RGB hue shift.", ODT11_G_HUESHIFT, hs_rgb_enable),
    ODT11_D("hs_r", "Hueshift R", "Hue shift, red.", ODT11_G_HUESHIFT, 0.0, 1.0, 0.01, hs_r),
    ODT11_D("hs_r_rng", "Hueshift R Range", "Hue shift red range.", ODT11_G_HUESHIFT, 0.0, 2.0, 0.02, hs_r_rng),
    ODT11_D("hs_g", "Hueshift G", "Hue shift, green.", ODT11_G_HUESHIFT, 0.0, 1.0, 0.01, hs_g),
    ODT11_D("hs_g_rng", "Hueshift G Range", "Hue shift green range.", ODT11_G_HUESHIFT, 0.0, 2.0, 0.02, hs_g_rng),
    ODT11_D("hs_b", "Hueshift B", "Hue shift, blue.", ODT11_G_HUESHIFT, 0.0, 1.0, 0.01, hs_b),
    ODT11_D("hs_b_rng", "Hueshift B Range", "Hue shift blue range.", ODT11_G_HUESHIFT, 0.0, 4.0, 0.02, hs_b_rng),
    ODT11_B("hs_cmy_enable", "Enable Hueshift CMY", "Enable the CMY hue shift.", ODT11_G_HUESHIFT, hs_cmy_enable),
    ODT11_D("hs_c", "Hueshift C", "Hue shift, cyan.", ODT11_G_HUESHIFT, 0.0, 1.0, 0.01, hs_c),
    ODT11_D("hs_c_rng", "Hueshift C Range", "Hue shift cyan range.", ODT11_G_HUESHIFT, 0.0, 1.0, 0.01, hs_c_rng),
    ODT11_D("hs_m", "Hueshift M", "Hue shift, magenta.", ODT11_G_HUESHIFT, 0.0, 1.0, 0.01, hs_m),
    ODT11_D("hs_m_rng", "Hueshift M Range", "Hue shift magenta range.", ODT11_G_HUESHIFT, 0.0, 1.0, 0.01, hs_m_rng),
    ODT11_D("hs_y", "Hueshift Y", "Hue shift, yellow.", ODT11_G_HUESHIFT, 0.0, 1.0, 0.01, hs_y),
    ODT11_D("hs_y_rng", "Hueshift Y Range", "Hue shift yellow range.", ODT11_G_HUESHIFT, 0.0, 1.0, 0.01, hs_y_rng),

    // Display
    ODT11_C("tn_su", "Surround", "Viewing surround.", ODT11_G_DISPLAY, tn_su, kOdt11SurroundNames, 3),
    ODT11_C("display_gamut", "Display Gamut", "Output display gamut.", ODT11_G_DISPLAY, display_gamut, kOdt11DisplayGamutNames, 6),
    ODT11_C("eotf", "Display EOTF", "Output display transfer function (EOTF).", ODT11_G_DISPLAY, eotf, kOdt11EotfLabels, 6),
    ODT11_B("clamp", "Clamp", "Clamp the output to the display range.", ODT11_G_DISPLAY, clamp),
};

#undef ODT11_D
#undef ODT11_B
#undef ODT11_C

static const int ODT11_NDESC = (int)(sizeof(kOdt11Desc) / sizeof(kOdt11Desc[0]));

// Index constants into kOdt11Desc (checked by the self test).
enum {
    ODT11_I_IN_OETF = 0,
    ODT11_I_LOOK = 1,
    ODT11_I_TONESCALE = 2,
    ODT11_I_CWP = 3,
    ODT11_I_CWP_LM = 4,
    ODT11_I_ENCODING = 5
};

// Which preset row set to write.
enum Odt11Rows {
    ODT11_ROWS_LOOK = 0,       // look row, then the tonescale row if one is picked
    ODT11_ROWS_TONESCALE = 1,  // tonescale row, or the look's tonescale for Use Look Preset
    ODT11_ROWS_ENCODING = 2,   // surround, display gamut, EOTF
    ODT11_ROWS_ALL = 3         // all of the above plus clamp, as upstream preset mode
};

static inline bool odt11_is_int_kind(int kind) { return kind != ODT11_DOUBLE; }

static inline void odt11_params_from_values(const double* v, OpenDRTv11Params* P) {
    unsigned char* base = reinterpret_cast<unsigned char*>(P);
    for (int i = 0; i < ODT11_NDESC; ++i) {
        const Odt11Desc& d = kOdt11Desc[i];
        if (odt11_is_int_kind(d.kind)) {
            const int iv = (int)std::lround(v[i]);
            std::memcpy(base + d.off, &iv, sizeof(int));
        } else {
            const float fv = (float)v[i];
            std::memcpy(base + d.off, &fv, sizeof(float));
        }
    }
}

static inline void odt11_values_from_params(const OpenDRTv11Params* P, double* v) {
    const unsigned char* base = reinterpret_cast<const unsigned char*>(P);
    for (int i = 0; i < ODT11_NDESC; ++i) {
        const Odt11Desc& d = kOdt11Desc[i];
        if (odt11_is_int_kind(d.kind)) {
            int iv = 0;
            std::memcpy(&iv, base + d.off, sizeof(int));
            v[i] = (double)iv;
        } else {
            float fv = 0.0f;
            std::memcpy(&fv, base + d.off, sizeof(float));
            v[i] = (double)fv;
        }
    }
}

// Describe-time defaults: exactly opendrt_v11_defaults, read through the table.
static inline void odt11_default_values(double* v) {
    OpenDRTv11Params P;
    opendrt_v11_defaults(&P);
    odt11_values_from_params(&P, v);
}

static inline int odt11_clamp_index(int i, int n) { return (i >= 0 && i < n) ? i : 0; }

// Write a preset's rows into the slider values. Selectors are read from v.
// Look also sets Clamp on, since upstream preset mode always clamps.
static inline void odt11_write_rows(double* v, int which) {
    OpenDRTv11Params P;
    std::memset(&P, 0, sizeof(P));
    odt11_params_from_values(v, &P);
    const int li = odt11_clamp_index(P.look_preset, ODT11_LOOK_COUNT);
    const int ts = P.tonescale_preset;
    const bool tsRow = (ts >= 1 && ts < ODT11_TONESCALE_COUNT);

    if (which == ODT11_ROWS_LOOK || which == ODT11_ROWS_ALL) {
        odt11_apply_look_row(&P, kOdt11LookTable[li]);
        if (tsRow) odt11_apply_tonescale_row(&P, kOdt11TonescaleTable[ts]);
        P.clamp = 1;
    }
    if (which == ODT11_ROWS_TONESCALE) {
        odt11_apply_tonescale_row(&P, tsRow ? kOdt11TonescaleTable[ts] : kOdt11LookTable[li]);
    }
    if (which == ODT11_ROWS_ENCODING || which == ODT11_ROWS_ALL) {
        const int ei = odt11_clamp_index(P.display_encoding_preset, ODT11_ENCODING_COUNT);
        P.tn_su = kOdt11EncodingTable[ei][0];
        P.display_gamut = kOdt11EncodingTable[ei][1];
        P.eotf = kOdt11EncodingTable[ei][2];
    }
    odt11_values_from_params(&P, v);
}

// Build the kernel params from the slider values (the source of truth) plus
// the two params shared with the Current model, then run the precompute.
static inline void odt11_build(const double* v, int in_gamut, int crv_enable, OpenDRTv11Params* P) {
    std::memset(P, 0, sizeof(*P));
    odt11_params_from_values(v, P);
    P->in_gamut = in_gamut;
    P->crv_enable = crv_enable ? 1 : 0;

    const int li = odt11_clamp_index(P->look_preset, ODT11_LOOK_COUNT);
    const float* L = kOdt11LookTable[li];
    P->cwp = (int)L[11];
    P->cwp_lm = L[12];
    P->pt_enable = odt11_flag(L[16]);
    if (P->ui_cwp != 0) P->cwp_lm = P->ui_cwp_lm;
    if (P->ui_cwp >= 1 && P->ui_cwp <= 6) P->cwp = P->ui_cwp - 1;

    opendrt_v11_precompute(P);
}
