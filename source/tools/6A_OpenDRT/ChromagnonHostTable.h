#pragma once
// ChromagnonHostTable.h - host side parameter table for the Chromagnon display
// transform (Model choice index 4). OFX free, so the host self test in
// harness/opendrt_models/ can include it directly.
//
// Port of the upstream Chromagnon View (GPL-3.0) UI. Upstream has no presets:
// every control is a plain slider, checkbox or combo, and the kernel params are
// built from those values alone (cmg_build).
//
// Values travel as one double per descriptor (the OFX value type). Float
// fields round trip float -> double -> float exactly.
//
// Invert: upstream "invert" checkbox (params field direction). The inverse
// takes display encoded input and returns scene linear in the input gamut,
// exactly as upstream; no re-encode is offered.
//
// Input Transfer Function: upstream Chromagnon has none (it is scene linear
// in). The core carries the 6A in_oetf addition, so the plugin offers the same
// 8 entries as Tesseract, default Linear (identical to upstream). It applies
// to the forward direction only.
//
// Not exposed:
//   in_gamut   shared with the other models (Input Gamut), passed to
//              cmg_build by the caller

#include <cmath>
#include <cstddef>
#include <cstring>
#include "core/ChromagnonParams.h"
#include "core/ChromagnonAlgorithm.h"

enum CmgKind { CMG_DOUBLE = 0, CMG_BOOL = 1, CMG_CHOICE = 2 };

// Group slots. CMG_G_INPUT is the shared Input group of the plugin.
enum CmgGroup {
    CMG_G_INPUT = 0,
    CMG_G_TONESCALE,
    CMG_G_NEUTRAL,
    CMG_G_PURITY,
    CMG_G_RENDERMTX,
    CMG_G_LOOKMTX,
    CMG_G_DISPLAY,
    CMG_G_COUNT
};

struct CmgGroupInfo { const char* name; const char* label; bool open; };
static const CmgGroupInfo kCmgGroups[CMG_G_COUNT] = {
    { "InputGroup",        "Input & Output Settings", false }, // existing group, not defined here
    { "CmgGroupTonescale", "Tonescale",               true  },
    { "CmgGroupNeutral",   "White and Grey",          false },
    { "CmgGroupPurity",    "Purity Compress",         false },
    { "CmgGroupRenderMtx", "Render Matrix",           false },
    { "CmgGroupLookMtx",   "Look Matrix",             false },
    { "CmgGroupDisplay",   "Display",                 false },
};

struct CmgDesc {
    const char* name;      // OFX param id
    const char* label;
    const char* hint;
    int         group;     // CmgGroup
    int         kind;      // CmgKind; DOUBLE = float field, BOOL/CHOICE = int field
    double      mn, mx, inc;
    size_t      off;       // offsetof(ChromagnonParams, field)
    const char* const* opts;
    int         nopts;
};

// Upstream combo labels (input transfer: the Tesseract list, see above).
static const char* const kCmgInOetfLabels[8] = {
    "Linear", "Davinci Intermediate", "Filmlight T-Log", "Arri LogC3",
    "Arri LogC4", "Panasonic V-Log", "Sony S-Log3", "Fuji F-Log"
};
// UI input transfer index -> 6A in_oetf index.
static const int kCmgInOetfTo6A[8] = { 0, 1, 2, 4, 5, 7, 8, 10 };
static const char* const kCmgDisplayGamutLabels[3] = { "Rec.709", "P3 D65", "Rec.2020" };
static const char* const kCmgEotfLabels[6] = {
    "Linear", "2.2 Power - sRGB Display", "2.4 Power - Rec .1886",
    "2.6 Power - DCI", "ST 2084 PQ", "HLG"
};

#define CMG_OFF(f) offsetof(ChromagnonParams, f)
#define CMG_D(id, lab, hint, g, mn, mx, inc, f) \
    { "cmg_" id, lab, hint, g, CMG_DOUBLE, mn, mx, inc, CMG_OFF(f), 0, 0 }
#define CMG_B(id, lab, hint, g, f) \
    { "cmg_" id, lab, hint, g, CMG_BOOL, 0.0, 1.0, 1.0, CMG_OFF(f), 0, 0 }
#define CMG_C(id, lab, hint, g, f, opts, n) \
    { "cmg_" id, lab, hint, g, CMG_CHOICE, 0.0, (double)((n) - 1), 1.0, CMG_OFF(f), opts, n }

// Definition (and UI) order. Index constants below must follow this table.
static const CmgDesc kCmgDesc[] = {
    // Input (shared Input group)
    CMG_C("in_oetf", "In Transfer Function", "Input transfer function for the Chromagnon model. Forward only.", CMG_G_INPUT, in_oetf, kCmgInOetfLabels, 8),

    // Tonescale
    CMG_D("Lp",   "Lp",            "Display peak luminance in nits.",  CMG_G_TONESCALE, 100.0, 1000.0, 10.0,   Lp),
    CMG_D("Lg",   "Lg",            "Display grey luminance in nits.",  CMG_G_TONESCALE, 4.0,   20.0,   0.1,    Lg),
    CMG_D("Lgb",  "Lg Boost",      "Grey luminance boost for HDR.",    CMG_G_TONESCALE, 0.0,   0.5,    0.005,  Lgb),
    CMG_D("off",  "Offset",        "Black point offset.",              CMG_G_TONESCALE, 0.0,   0.02,   0.0002, off),
    CMG_D("con",  "Contrast",      "Tonescale contrast.",              CMG_G_TONESCALE, 1.0,   3.0,    0.02,   con),
    CMG_D("conh", "High Contrast", "Highlight contrast.",              CMG_G_TONESCALE, 0.0,   0.5,    0.005,  conh),
    CMG_D("w0",   "High Clip",     "Scene value that maps to peak.",   CMG_G_TONESCALE, 16.0,  256.0,  1.0,    w0),

    // White and grey
    CMG_D("wtm", "White Temp", "White point temperature.", CMG_G_NEUTRAL, -0.6, 0.6, 0.01, wtm),
    CMG_D("wtn", "White Tint", "White point tint.",        CMG_G_NEUTRAL, -1.0, 1.0, 0.01, wtn),
    CMG_D("gtm", "Grey Temp",  "Grey temperature.",        CMG_G_NEUTRAL, -0.6, 0.6, 0.01, gtm),
    CMG_D("gtn", "Grey Tint",  "Grey tint.",               CMG_G_NEUTRAL, -1.0, 1.0, 0.01, gtn),

    // Purity compress
    CMG_D("pc_p",  "Slope",     "Purity compress slope.",     CMG_G_PURITY, 0.0, 1.0, 0.01,  pc_p),
    CMG_D("pc_x0", "Amount",    "Purity compress amount.",    CMG_G_PURITY, 0.0, 0.5, 0.005, pc_x0),
    CMG_D("pc_t0", "Threshold", "Purity compress threshold.", CMG_G_PURITY, 0.0, 1.0, 0.01,  pc_t0),

    // Render matrix
    CMG_D("rs_rnd", "Scale R",  "Render matrix scale, red.",    CMG_G_RENDERMTX, 0.0,  0.3, 0.003, rs_rnd),
    CMG_D("gs_rnd", "Scale G",  "Render matrix scale, green.",  CMG_G_RENDERMTX, 0.0,  0.3, 0.003, gs_rnd),
    CMG_D("bs_rnd", "Scale B",  "Render matrix scale, blue.",   CMG_G_RENDERMTX, 0.0,  0.3, 0.003, bs_rnd),
    CMG_D("rr_rnd", "Rotate R", "Render matrix rotate, red.",   CMG_G_RENDERMTX, -0.1, 0.1, 0.002, rr_rnd),
    CMG_D("gr_rnd", "Rotate G", "Render matrix rotate, green.", CMG_G_RENDERMTX, -0.1, 0.1, 0.002, gr_rnd),
    CMG_D("br_rnd", "Rotate B", "Render matrix rotate, blue.",  CMG_G_RENDERMTX, -0.1, 0.1, 0.002, br_rnd),

    // Look matrix
    CMG_D("rs_lk", "R Dist", "Look matrix distance, red.",   CMG_G_LOOKMTX, 0.0, 0.3, 0.003, rs_lk),
    CMG_D("gs_lk", "G Dist", "Look matrix distance, green.", CMG_G_LOOKMTX, 0.0, 0.3, 0.003, gs_lk),
    CMG_D("bs_lk", "B Dist", "Look matrix distance, blue.",  CMG_G_LOOKMTX, 0.0, 0.3, 0.003, bs_lk),

    // Display
    CMG_B("clamp",  "Clamp",  "Clamp the output to the display range.", CMG_G_DISPLAY, clamp),
    CMG_B("invert", "Invert", "Inverse transform: display encoded in, scene linear in the input gamut out.", CMG_G_DISPLAY, direction),
    CMG_C("display_gamut", "Display Gamut", "Output display gamut.", CMG_G_DISPLAY, display_gamut, kCmgDisplayGamutLabels, 3),
    CMG_C("eotf", "Display EOTF", "Output display transfer function (EOTF).", CMG_G_DISPLAY, eotf, kCmgEotfLabels, 6),
};

#undef CMG_D
#undef CMG_B
#undef CMG_C

static const int CMG_NDESC = (int)(sizeof(kCmgDesc) / sizeof(kCmgDesc[0]));

// Index constants into kCmgDesc (checked by the self test).
enum {
    CMG_I_IN_OETF = 0,
    CMG_I_LP = 1,
    CMG_I_CLAMP = 24,
    CMG_I_INVERT = 25,
    CMG_I_DISPLAY_GAMUT = 26,
    CMG_I_EOTF = 27
};

static inline bool cmg_is_int_kind(int kind) { return kind != CMG_DOUBLE; }

static inline int cmg_clamp_index(int i, int n) { return (i >= 0 && i < n) ? i : 0; }

// 6A in_oetf index -> UI index (0 for the 6A extras, which the UI does not
// offer).
static inline int cmg_in_oetf_to_ui(int v6a) {
    for (int i = 0; i < 8; ++i) if (kCmgInOetfTo6A[i] == v6a) return i;
    return 0;
}

static inline void cmg_params_from_values(const double* v, ChromagnonParams* P) {
    unsigned char* base = reinterpret_cast<unsigned char*>(P);
    for (int i = 0; i < CMG_NDESC; ++i) {
        const CmgDesc& d = kCmgDesc[i];
        if (cmg_is_int_kind(d.kind)) {
            int iv = (int)std::lround(v[i]);
            if (i == CMG_I_IN_OETF) iv = kCmgInOetfTo6A[cmg_clamp_index(iv, 8)];
            std::memcpy(base + d.off, &iv, sizeof(int));
        } else {
            const float fv = (float)v[i];
            std::memcpy(base + d.off, &fv, sizeof(float));
        }
    }
}

static inline void cmg_values_from_params(const ChromagnonParams* P, double* v) {
    const unsigned char* base = reinterpret_cast<const unsigned char*>(P);
    for (int i = 0; i < CMG_NDESC; ++i) {
        const CmgDesc& d = kCmgDesc[i];
        if (cmg_is_int_kind(d.kind)) {
            int iv = 0;
            std::memcpy(&iv, base + d.off, sizeof(int));
            if (i == CMG_I_IN_OETF) iv = cmg_in_oetf_to_ui(iv);
            v[i] = (double)iv;
        } else {
            float fv = 0.0f;
            std::memcpy(&fv, base + d.off, sizeof(float));
            v[i] = (double)fv;
        }
    }
}

// Describe-time defaults: exactly chromagnon_defaults, read through the table.
static inline void cmg_default_values(double* v) {
    ChromagnonParams P;
    chromagnon_defaults(&P);
    cmg_values_from_params(&P, v);
}

// Build the kernel params from the control values plus the shared Input
// Gamut, then run the precompute.
static inline void cmg_build(const double* v, int in_gamut, ChromagnonParams* P) {
    chromagnon_defaults(P);
    cmg_params_from_values(v, P);
    P->in_gamut = in_gamut;
    chromagnon_precompute(P);
}
