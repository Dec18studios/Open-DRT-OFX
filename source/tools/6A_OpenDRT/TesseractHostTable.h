#pragma once
// TesseractHostTable.h - host side parameter table for the Tesseract display
// transform (Model choice index 3). OFX free, so the host self test in
// harness/opendrt_models/ can include it directly.
//
// Port of the upstream Tesseract v0.0.3 (GPL-3.0) UI. Upstream has no presets:
// every control is a plain slider, checkbox or combo, and the kernel params are
// built from those values alone (tess_build).
//
// Values travel as one double per descriptor (the OFX value type). Float
// fields round trip float -> double -> float exactly.
//
// Input Transfer Function: the combo shows the upstream 8 entries in upstream
// order. The params struct holds the 6A index (see core/TesseractParams.h), so
// tess_params_from_values maps UI -> 6A and tess_values_from_params maps back.
//
// Not exposed:
//   in_gamut   shared with the other models (Input Gamut), passed to
//              tess_build by the caller

#include <cmath>
#include <cstddef>
#include <cstring>
#include "core/TesseractParams.h"
#include "core/TesseractAlgorithm.h"

enum TessKind { TESS_DOUBLE = 0, TESS_BOOL = 1, TESS_CHOICE = 2 };

// Group slots. TESS_G_INPUT is the shared Input group of the plugin.
enum TessGroup {
    TESS_G_INPUT = 0,
    TESS_G_TONESCALE,
    TESS_G_COLOR,
    TESS_G_DISPLAY,
    TESS_G_COUNT
};

struct TessGroupInfo { const char* name; const char* label; bool open; };
static const TessGroupInfo kTessGroups[TESS_G_COUNT] = {
    { "InputGroup",         "Input & Output Settings", false }, // existing group, not defined here
    { "TessGroupTonescale", "Tonescale",               true  },
    { "TessGroupColor",     "Purity and Hue",          false },
    { "TessGroupDisplay",   "Display",                 false },
};

struct TessDesc {
    const char* name;      // OFX param id
    const char* label;
    const char* hint;
    int         group;     // TessGroup
    int         kind;      // TessKind; DOUBLE = float field, BOOL/CHOICE = int field
    double      mn, mx, inc;
    size_t      off;       // offsetof(TesseractParams, field)
    const char* const* opts;
    int         nopts;
};

// Upstream combo labels.
static const char* const kTessInOetfLabels[8] = {
    "Linear", "Davinci Intermediate", "Filmlight T-Log", "Arri LogC3",
    "Arri LogC4", "Panasonic V-Log", "Sony S-Log3", "Fuji F-Log"
};
// Upstream input transfer index -> 6A in_oetf index.
static const int kTessInOetfTo6A[8] = { 0, 1, 2, 4, 5, 7, 8, 10 };
static const char* const kTessDisplayGamutLabels[3] = { "Rec.709", "P3 D65", "Rec.2020" };
static const char* const kTessEotfLabels[6] = {
    "Linear", "2.2 Power - sRGB Display", "2.4 Power - Rec .1886",
    "2.6 Power - DCI", "ST 2084 PQ", "HLG"
};

#define TESS_OFF(f) offsetof(TesseractParams, f)
#define TESS_D(id, lab, hint, g, mn, mx, inc, f) \
    { "tess_" id, lab, hint, g, TESS_DOUBLE, mn, mx, inc, TESS_OFF(f), 0, 0 }
#define TESS_B(id, lab, hint, g, f) \
    { "tess_" id, lab, hint, g, TESS_BOOL, 0.0, 1.0, 1.0, TESS_OFF(f), 0, 0 }
#define TESS_C(id, lab, hint, g, f, opts, n) \
    { "tess_" id, lab, hint, g, TESS_CHOICE, 0.0, (double)((n) - 1), 1.0, TESS_OFF(f), opts, n }

// Definition (and UI) order. Index constants below must follow this table.
static const TessDesc kTessDesc[] = {
    // Input (shared Input group)
    TESS_C("in_oetf", "In Transfer Function", "Input transfer function for the Tesseract model.", TESS_G_INPUT, in_oetf, kTessInOetfLabels, 8),

    // Tonescale
    TESS_D("Lp",  "Lp",       "Display peak luminance in nits.",  TESS_G_TONESCALE, 100.0, 1000.0, 10.0,   Lp),
    TESS_D("Lg",  "Lg",       "Display grey luminance in nits.",  TESS_G_TONESCALE, 4.0,   20.0,   0.1,    Lg),
    TESS_D("off", "Offset",   "Black point offset.",              TESS_G_TONESCALE, 0.0,   0.02,   0.0002, off),
    TESS_D("con", "Contrast", "Tonescale contrast.",              TESS_G_TONESCALE, 1.0,   2.0,    0.01,   con),

    // Purity and hue
    TESS_D("pb",   "Purity",     "Purity.",                 TESS_G_COLOR, 0.0,  1.0, 0.01, pb),
    TESS_D("hl_r", "Hue Low R",  "Low range hue shift, red.",   TESS_G_COLOR, -1.0, 1.0, 0.01, hl_r),
    TESS_D("hl_g", "Hue Low G",  "Low range hue shift, green.", TESS_G_COLOR, -1.0, 1.0, 0.01, hl_g),
    TESS_D("hl_b", "Hue Low B",  "Low range hue shift, blue.",  TESS_G_COLOR, -1.0, 1.0, 0.01, hl_b),
    TESS_D("hh_r", "Hue High R", "High range hue shift, red.",   TESS_G_COLOR, -1.0, 1.0, 0.01, hh_r),
    TESS_D("hh_g", "Hue High G", "High range hue shift, green.", TESS_G_COLOR, -1.0, 1.0, 0.01, hh_g),
    TESS_D("hh_b", "Hue High B", "High range hue shift, blue.",  TESS_G_COLOR, -1.0, 1.0, 0.01, hh_b),

    // Display
    TESS_B("clamp", "Clamp", "Clamp the output to the display range.", TESS_G_DISPLAY, clamp),
    TESS_C("display_gamut", "Display Gamut", "Output display gamut.", TESS_G_DISPLAY, display_gamut, kTessDisplayGamutLabels, 3),
    TESS_C("eotf", "Display EOTF", "Output display transfer function (EOTF).", TESS_G_DISPLAY, eotf, kTessEotfLabels, 6),
};

#undef TESS_D
#undef TESS_B
#undef TESS_C

static const int TESS_NDESC = (int)(sizeof(kTessDesc) / sizeof(kTessDesc[0]));

// Index constants into kTessDesc (checked by the self test).
enum {
    TESS_I_IN_OETF = 0,
    TESS_I_LP = 1,
    TESS_I_CLAMP = 12,
    TESS_I_DISPLAY_GAMUT = 13,
    TESS_I_EOTF = 14
};

static inline bool tess_is_int_kind(int kind) { return kind != TESS_DOUBLE; }

static inline int tess_clamp_index(int i, int n) { return (i >= 0 && i < n) ? i : 0; }

// 6A in_oetf index -> upstream UI index (0 for the 6A extras, which the UI
// does not offer).
static inline int tess_in_oetf_to_ui(int v6a) {
    for (int i = 0; i < 8; ++i) if (kTessInOetfTo6A[i] == v6a) return i;
    return 0;
}

static inline void tess_params_from_values(const double* v, TesseractParams* P) {
    unsigned char* base = reinterpret_cast<unsigned char*>(P);
    for (int i = 0; i < TESS_NDESC; ++i) {
        const TessDesc& d = kTessDesc[i];
        if (tess_is_int_kind(d.kind)) {
            int iv = (int)std::lround(v[i]);
            if (i == TESS_I_IN_OETF) iv = kTessInOetfTo6A[tess_clamp_index(iv, 8)];
            std::memcpy(base + d.off, &iv, sizeof(int));
        } else {
            const float fv = (float)v[i];
            std::memcpy(base + d.off, &fv, sizeof(float));
        }
    }
}

static inline void tess_values_from_params(const TesseractParams* P, double* v) {
    const unsigned char* base = reinterpret_cast<const unsigned char*>(P);
    for (int i = 0; i < TESS_NDESC; ++i) {
        const TessDesc& d = kTessDesc[i];
        if (tess_is_int_kind(d.kind)) {
            int iv = 0;
            std::memcpy(&iv, base + d.off, sizeof(int));
            if (i == TESS_I_IN_OETF) iv = tess_in_oetf_to_ui(iv);
            v[i] = (double)iv;
        } else {
            float fv = 0.0f;
            std::memcpy(&fv, base + d.off, sizeof(float));
            v[i] = (double)fv;
        }
    }
}

// Describe-time defaults: exactly tesseract_defaults, read through the table.
static inline void tess_default_values(double* v) {
    TesseractParams P;
    tesseract_defaults(&P);
    tess_values_from_params(&P, v);
}

// Build the kernel params from the control values plus the shared Input
// Gamut, then run the precompute.
static inline void tess_build(const double* v, int in_gamut, TesseractParams* P) {
    tesseract_defaults(P);
    tess_params_from_values(v, P);
    P->in_gamut = in_gamut;
    tesseract_precompute(P);
}
