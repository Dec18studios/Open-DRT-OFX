#pragma once
// OpenDRTv11Presets.h
// Host-only preset tables and parameter setup for the second OpenDRT model
// ("model 1"): a port of upstream OpenDRT v1.1.0 (GPL-3.0), commit af68332.
// Mirrors the DCTL "Parameter Setup" block (preset mode, INIT_PARAMS undefined)
// and the tonescale constant block at the top of the DCTL transform.
//
// Order of operations in opendrt_v11_fill_presets, same as the DCTL:
//   1. look preset writes every look field (tonescale, cwp, cwp_lm, render
//      space, purity, brilliance, hue contrast, hue shift)
//   2. tonescale preset (if nonzero) overwrites the 11 tonescale fields
//   3. clamp = 1
//   4. creative white override: if ui_cwp != 0, cwp_lm = ui_cwp_lm and
//      cwp = ui_cwp - 1
//   5. display encoding preset writes tn_su, display_gamut, eotf
//
// Deviation: the DCTL leaves the look and encoding variables uninitialized for
// an out-of-range preset index. Here an out-of-range look falls back to 0
// (Standard) and an out-of-range encoding falls back to 0 (Rec.1886).
// Look index 7 ("Base") exists in the DCTL body but is hidden from its combo
// box; it is kept here and reachable by index.
//
// Stickshift mode (the DCTL INIT_PARAMS path) needs no function: every field
// of OpenDRTv11Params can be written directly, followed by
// opendrt_v11_precompute.

#if !defined(__CUDACC__) && !defined(__METAL_VERSION__)

#include <cmath>
#include "OpenDRTv11Params.h"

// ---- Names (upstream combo labels) -----------------------------------------
static const int ODT11_LOOK_COUNT = 8;        // 7 visible + Base (hidden)
static const int ODT11_LOOK_VISIBLE_COUNT = 7;
static const char* const kOdt11LookNames[ODT11_LOOK_COUNT] = {
    "Standard", "Arriba", "Sylvan", "Colorful", "Aery", "Dystopic", "Umbra", "Base"
};

static const int ODT11_TONESCALE_COUNT = 14;  // 0 = use look preset
static const char* const kOdt11TonescaleNames[ODT11_TONESCALE_COUNT] = {
    "USE LOOK PRESET", "Low Contrast", "Medium Contrast", "High Contrast",
    "Arriba Tonescale", "Sylvan Tonescale", "Colorful Tonescale", "Aery Tonescale",
    "Dystopic Tonescale", "Umbra Tonescale", "ACES-1.x", "ACES-2.0",
    "Marvelous Tonescape", "DaGrinchi ToneGroan"
};

static const int ODT11_ENCODING_COUNT = 9;
static const char* const kOdt11EncodingNames[ODT11_ENCODING_COUNT] = {
    "Rec.1886 - 2.4 Power / Rec.709",
    "sRGB Display - 2.2 Power / Rec.709",
    "Display P3 - 2.2 Power / P3-D65",
    "DCI - 2.6 Power / P3-D60",
    "DCI - 2.6 Power / P3-DCI",
    "DCI - 2.6 Power / XYZ",
    "Rec.2100 - PQ / Rec.2020 (P3 Limited)",
    "Rec.2100 - HLG / Rec.2020 (P3 Limited)",
    "Dolby - PQ / P3-D65"
};

static const int ODT11_CWP_UI_COUNT = 7;      // 0 = use look preset
static const char* const kOdt11CwpUiNames[ODT11_CWP_UI_COUNT] = {
    "USE LOOK PRESET", "D93", "D75", "D65", "D60", "D55", "D50"
};

static const char* const kOdt11CwpNames[6] = { "D93", "D75", "D65", "D60", "D55", "D50" };
static const char* const kOdt11SurroundNames[3] = { "Dark", "Dim", "Bright" };
static const char* const kOdt11DisplayGamutNames[6] = {
    "Rec.709", "P3-D65", "Rec.2020 (P3 Limited)", "P3-D60", "P3-DCI", "XYZ"
};
static const char* const kOdt11EotfNames[6] = { "Linear", "2.2 Power", "2.4 Power", "2.6 Power", "ST 2084 PQ", "HLG" };

// ---- Look preset table ------------------------------------------------------
// 64 values per row, in this field order (enables stored as 0/1):
//  0 tn_con  1 tn_sh  2 tn_toe  3 tn_off  4 tn_hcon_enable  5 tn_hcon
//  6 tn_hcon_pv  7 tn_hcon_st  8 tn_lcon_enable  9 tn_lcon  10 tn_lcon_w
// 11 cwp  12 cwp_lm  13 rs_sa  14 rs_rw  15 rs_bw
// 16 pt_enable  17 pt_lml  18 pt_lml_r  19 pt_lml_g  20 pt_lml_b
// 21 pt_lmh  22 pt_lmh_r  23 pt_lmh_b
// 24 ptl_enable  25 ptl_c  26 ptl_m  27 ptl_y
// 28 ptm_enable  29 ptm_low  30 ptm_low_rng  31 ptm_low_st
// 32 ptm_high  33 ptm_high_rng  34 ptm_high_st
// 35 brl_enable  36 brl  37 brl_r  38 brl_g  39 brl_b  40 brl_rng  41 brl_st
// 42 brlp_enable  43 brlp  44 brlp_r  45 brlp_g  46 brlp_b
// 47 hc_enable  48 hc_r  49 hc_r_rng
// 50 hs_rgb_enable  51 hs_r  52 hs_r_rng  53 hs_g  54 hs_g_rng  55 hs_b  56 hs_b_rng
// 57 hs_cmy_enable  58 hs_c  59 hs_c_rng  60 hs_m  61 hs_m_rng  62 hs_y  63 hs_y_rng
static const int ODT11_LOOK_FIELDS = 64;
static const float kOdt11LookTable[ODT11_LOOK_COUNT][ODT11_LOOK_FIELDS] = {
    // 0 Standard
    { 1.66f, 0.5f, 0.003f, 0.005f, 0, 0.0f, 1.0f, 4.0f, 0, 0.0f, 0.5f,
      2, 0.25f, 0.35f, 0.25f, 0.55f,
      1, 0.25f, 0.5f, 0.0f, 0.1f, 0.25f, 0.5f, 0.0f,
      1, 0.06f, 0.08f, 0.06f,
      1, 0.4f, 0.25f, 0.5f, -0.8f, 0.35f, 0.4f,
      1, 0.0f, -2.5f, -1.5f, -1.5f, 0.5f, 0.35f,
      1, -0.5f, -1.25f, -1.25f, -0.25f,
      1, 1.0f, 0.3f,
      1, 0.6f, 0.6f, 0.35f, 1.0f, 0.66f, 1.0f,
      1, 0.25f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f },
    // 1 Arriba
    { 1.05f, 0.5f, 0.1f, 0.01f, 0, 0.0f, 1.0f, 4.0f, 1, 1.5f, 0.2f,
      2, 0.25f, 0.35f, 0.25f, 0.55f,
      1, 0.25f, 0.45f, 0.0f, 0.1f, 0.25f, 0.25f, 0.0f,
      1, 0.06f, 0.08f, 0.06f,
      1, 1.0f, 0.4f, 0.5f, -0.8f, 0.66f, 0.6f,
      1, 0.0f, -2.5f, -1.5f, -1.5f, 0.5f, 0.35f,
      1, 0.0f, -1.7f, -2.0f, -0.5f,
      1, 1.0f, 0.3f,
      1, 0.6f, 0.8f, 0.35f, 1.0f, 0.66f, 1.0f,
      1, 0.15f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f },
    // 2 Sylvan
    { 1.6f, 0.5f, 0.01f, 0.01f, 0, 0.0f, 1.0f, 4.0f, 1, 0.25f, 0.75f,
      2, 0.25f, 0.25f, 0.25f, 0.55f,
      1, 0.15f, 0.5f, 0.15f, 0.1f, 0.25f, 0.15f, 0.15f,
      1, 0.05f, 0.08f, 0.05f,
      1, 0.5f, 0.5f, 0.5f, -0.8f, 0.5f, 0.5f,
      1, -1.0f, -2.0f, -2.0f, 0.0f, 0.25f, 0.25f,
      1, -1.0f, -0.5f, -0.25f, -0.25f,
      1, 1.0f, 0.4f,
      1, 0.6f, 1.15f, 0.8f, 1.25f, 0.6f, 1.0f,
      1, 0.25f, 0.25f, 0.25f, 0.5f, 0.35f, 0.5f },
    // 3 Colorful
    { 1.5f, 0.5f, 0.003f, 0.003f, 0, 0.0f, 1.0f, 4.0f, 1, 0.4f, 0.5f,
      2, 0.25f, 0.35f, 0.25f, 0.55f,
      1, 0.5f, 1.0f, 0.0f, 0.5f, 0.15f, 0.15f, 0.15f,
      1, 0.05f, 0.06f, 0.05f,
      1, 0.8f, 0.5f, 0.4f, -0.8f, 0.4f, 0.4f,
      1, 0.0f, -1.25f, -1.25f, -0.25f, 0.3f, 0.5f,
      1, -0.5f, -1.25f, -1.25f, -0.5f,
      1, 1.0f, 0.4f,
      1, 0.5f, 0.8f, 0.35f, 1.0f, 0.5f, 1.0f,
      1, 0.25f, 1.0f, 0.0f, 1.0f, 0.25f, 1.0f },
    // 4 Aery
    { 1.15f, 0.5f, 0.04f, 0.006f, 0, 0.0f, 0.0f, 0.5f, 1, 0.5f, 2.0f,
      1, 0.25f, 0.25f, 0.2f, 0.5f,
      1, 0.0f, 0.5f, 0.15f, 0.1f, 0.0f, 0.1f, 0.0f,
      1, 0.05f, 0.08f, 0.05f,
      1, 0.8f, 0.35f, 0.5f, -0.9f, 0.5f, 0.3f,
      1, -3.0f, 0.0f, 0.0f, 1.0f, 0.8f, 0.15f,
      1, -1.0f, -1.0f, -1.0f, 0.0f,
      1, 0.5f, 0.25f,
      1, 0.6f, 1.0f, 0.35f, 2.0f, 0.5f, 1.5f,
      1, 0.35f, 1.0f, 0.25f, 1.0f, 0.35f, 0.5f },
    // 5 Dystopic
    { 1.6f, 0.5f, 0.01f, 0.008f, 1, 0.25f, 0.0f, 1.0f, 1, 1.0f, 0.75f,
      3, 0.25f, 0.2f, 0.25f, 0.55f,
      1, 0.15f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
      1, 0.05f, 0.08f, 0.05f,
      1, 0.25f, 0.25f, 0.8f, -0.8f, 0.6f, 0.25f,
      1, -2.0f, -2.0f, -2.0f, 0.0f, 0.35f, 0.35f,
      1, 0.0f, -1.0f, -1.0f, -1.0f,
      1, 1.0f, 0.25f,
      1, 0.7f, 1.33f, 1.0f, 2.0f, 0.75f, 2.0f,
      1, 1.0f, 0.5f, 1.0f, 1.0f, 1.0f, 0.765f },
    // 6 Umbra
    { 1.8f, 0.5f, 0.001f, 0.015f, 0, 0.0f, 1.0f, 4.0f, 1, 1.0f, 1.0f,
      5, 0.25f, 0.35f, 0.25f, 0.55f,
      1, 0.0f, 0.5f, 0.0f, 0.15f, 0.25f, 0.25f, 0.0f,
      1, 0.05f, 0.06f, 0.05f,
      1, 0.4f, 0.35f, 0.66f, -0.6f, 0.45f, 0.45f,
      1, -2.0f, -4.5f, -3.0f, -4.0f, 0.35f, 0.3f,
      1, 0.0f, -2.0f, -1.0f, -0.5f,
      1, 1.0f, 0.35f,
      1, 0.66f, 1.0f, 0.5f, 2.0f, 0.85f, 2.0f,
      1, 0.0f, 1.0f, 0.25f, 1.0f, 0.66f, 0.66f },
    // 7 Base (hidden in the DCTL combo)
    { 1.66f, 0.5f, 0.003f, 0.005f, 0, 0.0f, 1.0f, 4.0f, 0, 0.0f, 0.5f,
      2, 0.25f, 0.35f, 0.25f, 0.55f,
      1, 0.5f, 0.5f, 0.15f, 0.15f, 0.8f, 0.5f, 0.0f,
      1, 0.05f, 0.06f, 0.05f,
      0, 0.0f, 0.5f, 0.5f, 0.0f, 0.5f, 0.5f,
      0, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.35f,
      1, -0.5f, -1.6f, -1.6f, -0.8f,
      0, 0.0f, 0.25f,
      0, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
      0, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f },
};

// ---- Tonescale preset table (rows 1..13; row 0 = use look) -----------------
// 11 values: tn_con, tn_sh, tn_toe, tn_off, tn_hcon_enable, tn_hcon,
// tn_hcon_pv, tn_hcon_st, tn_lcon_enable, tn_lcon, tn_lcon_w
static const int ODT11_TONESCALE_FIELDS = 11;
static const float kOdt11TonescaleTable[ODT11_TONESCALE_COUNT][ODT11_TONESCALE_FIELDS] = {
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },                                   // 0 unused
    { 1.4f, 0.5f, 0.003f, 0.005f, 0, 0.0f, 1.0f, 4.0f, 0, 0.0f, 0.5f },    // 1 Low Contrast
    { 1.66f, 0.5f, 0.003f, 0.005f, 0, 0.0f, 1.0f, 4.0f, 0, 0.0f, 0.5f },   // 2 Medium Contrast
    { 1.4f, 0.5f, 0.003f, 0.005f, 0, 0.0f, 1.0f, 4.0f, 1, 1.0f, 0.5f },    // 3 High Contrast
    { 1.05f, 0.5f, 0.1f, 0.01f, 0, 0.0f, 1.0f, 4.0f, 1, 1.5f, 0.2f },      // 4 Arriba
    { 1.6f, 0.5f, 0.01f, 0.01f, 0, 0.0f, 1.0f, 4.0f, 1, 0.25f, 0.75f },    // 5 Sylvan
    { 1.5f, 0.5f, 0.003f, 0.003f, 0, 0.0f, 1.0f, 4.0f, 1, 0.4f, 0.5f },    // 6 Colorful
    { 1.15f, 0.5f, 0.04f, 0.006f, 0, 0.0f, 0.0f, 0.5f, 1, 0.5f, 2.0f },    // 7 Aery
    { 1.6f, 0.5f, 0.01f, 0.008f, 1, 0.25f, 0.0f, 1.0f, 1, 1.0f, 0.75f },   // 8 Dystopic
    { 1.8f, 0.5f, 0.001f, 0.015f, 0, 0.0f, 1.0f, 4.0f, 1, 1.0f, 1.0f },    // 9 Umbra
    { 1.0f, 0.35f, 0.02f, 0.0f, 1, 0.55f, 0.0f, 2.0f, 1, 1.13f, 1.0f },    // 10 ACES-1.x
    { 1.15f, 0.5f, 0.04f, 0.0f, 0, 1.0f, 1.0f, 1.0f, 0, 1.0f, 0.6f },      // 11 ACES-2.0 (hcon/lcon values dead, both disabled)
    { 1.5f, 0.5f, 0.003f, 0.01f, 1, 0.25f, 0.0f, 4.0f, 1, 1.0f, 1.0f },    // 12 Marvelous Tonescape
    { 1.2f, 0.5f, 0.02f, 0.0f, 0, 0.0f, 1.0f, 1.0f, 0, 0.0f, 0.6f },       // 13 DaGrinchi ToneGroan
};

// ---- Display encoding table: tn_su, display_gamut, eotf --------------------
static const int kOdt11EncodingTable[ODT11_ENCODING_COUNT][3] = {
    { 1, 0, 2 }, // Rec.1886
    { 2, 0, 1 }, // sRGB Display
    { 2, 1, 1 }, // Display P3
    { 0, 3, 3 }, // P3-D60
    { 0, 4, 3 }, // P3-DCI
    { 0, 5, 3 }, // DCI XYZ
    { 0, 2, 4 }, // Rec.2100 PQ
    { 0, 2, 5 }, // Rec.2100 HLG
    { 0, 1, 4 }, // Dolby PQ
};

static inline int odt11_flag(float v) { return v != 0.0f ? 1 : 0; }

static inline void odt11_apply_tonescale_row(OpenDRTv11Params* p, const float* t) {
    p->tn_con = t[0];
    p->tn_sh = t[1];
    p->tn_toe = t[2];
    p->tn_off = t[3];
    p->tn_hcon_enable = odt11_flag(t[4]);
    p->tn_hcon = t[5];
    p->tn_hcon_pv = t[6];
    p->tn_hcon_st = t[7];
    p->tn_lcon_enable = odt11_flag(t[8]);
    p->tn_lcon = t[9];
    p->tn_lcon_w = t[10];
}

static inline void odt11_apply_look_row(OpenDRTv11Params* p, const float* L) {
    odt11_apply_tonescale_row(p, L);
    p->cwp = (int)L[11];
    p->cwp_lm = L[12];
    p->rs_sa = L[13];
    p->rs_rw = L[14];
    p->rs_bw = L[15];
    p->pt_enable = odt11_flag(L[16]);
    p->pt_lml = L[17];
    p->pt_lml_r = L[18];
    p->pt_lml_g = L[19];
    p->pt_lml_b = L[20];
    p->pt_lmh = L[21];
    p->pt_lmh_r = L[22];
    p->pt_lmh_b = L[23];
    p->ptl_enable = odt11_flag(L[24]);
    p->ptl_c = L[25];
    p->ptl_m = L[26];
    p->ptl_y = L[27];
    p->ptm_enable = odt11_flag(L[28]);
    p->ptm_low = L[29];
    p->ptm_low_rng = L[30];
    p->ptm_low_st = L[31];
    p->ptm_high = L[32];
    p->ptm_high_rng = L[33];
    p->ptm_high_st = L[34];
    p->brl_enable = odt11_flag(L[35]);
    p->brl = L[36];
    p->brl_r = L[37];
    p->brl_g = L[38];
    p->brl_b = L[39];
    p->brl_rng = L[40];
    p->brl_st = L[41];
    p->brlp_enable = odt11_flag(L[42]);
    p->brlp = L[43];
    p->brlp_r = L[44];
    p->brlp_g = L[45];
    p->brlp_b = L[46];
    p->hc_enable = odt11_flag(L[47]);
    p->hc_r = L[48];
    p->hc_r_rng = L[49];
    p->hs_rgb_enable = odt11_flag(L[50]);
    p->hs_r = L[51];
    p->hs_r_rng = L[52];
    p->hs_g = L[53];
    p->hs_g_rng = L[54];
    p->hs_b = L[55];
    p->hs_b_rng = L[56];
    p->hs_cmy_enable = odt11_flag(L[57]);
    p->hs_c = L[58];
    p->hs_c_rng = L[59];
    p->hs_m = L[60];
    p->hs_m_rng = L[61];
    p->hs_y = L[62];
    p->hs_y_rng = L[63];
}

// Same as the DCTL toe_quadratic, host copy for the precompute.
static inline float odt11_host_toe_quadratic(float x, float toe, int inv) {
    if (toe == 0.0f) return x;
    if (inv == 0) {
        const float x2 = (x <= 0.0f) ? x : std::pow(x, 2.0f);
        return x2/(x + toe);
    }
    return (x + std::sqrt(x*(4.0f*toe + x)))/2.0f;
}

// DCTL tonescale constant block, same formulas and order, float math.
static inline void opendrt_v11_precompute(OpenDRTv11Params* p) {
    const float tn_con = p->tn_con;
    p->ts_x1 = std::pow(2.0f, 6.0f*p->tn_sh + 4.0f);
    p->ts_y1 = p->tn_Lp/100.0f;
    p->ts_x0 = 0.18f + p->tn_off;
    p->ts_y0 = p->tn_Lg/100.0f*(1.0f + p->tn_gb*std::log2(p->ts_y1));
    p->ts_s0 = odt11_host_toe_quadratic(p->ts_y0, p->tn_toe, 1);
    p->ts_p = tn_con/(1.0f + (float)p->tn_su*0.05f);
    p->ts_s10 = p->ts_x0*(std::pow(p->ts_s0, -1.0f/tn_con) - 1.0f);
    p->ts_m1 = p->ts_y1/std::pow(p->ts_x1/(p->ts_x1 + p->ts_s10), tn_con);
    p->ts_m2 = odt11_host_toe_quadratic(p->ts_m1, p->tn_toe, 1);
    p->ts_s = p->ts_x0*(std::pow(p->ts_s0/p->ts_m2, -1.0f/tn_con) - 1.0f);
    // DCTL: eotf==4 ? 0.01 : eotf==5 ? 0.1 : 100.0f/tn_Lp (double literals, stored as float)
    p->ts_dsc = p->eotf == 4 ? 0.01f : p->eotf == 5 ? 0.1f : 100.0f/p->tn_Lp;
    p->pt_cmp_Lf = p->pt_hdr*std::fmin(1.0f, (p->tn_Lp - 100.0f)/900.0f);
    p->s_Lp100 = p->ts_x0*(std::pow((p->tn_Lg/100.0f), -1.0f/tn_con) - 1.0f);
    p->ts_s1 = p->ts_s*p->pt_cmp_Lf + p->s_Lp100*(1.0f - p->pt_cmp_Lf);
}

// Preset mode, as the DCTL with INIT_PARAMS undefined. Reads the top-level
// fields (in_gamut, in_oetf, tn_Lp, tn_gb, pt_hdr, tn_Lg, crv_enable, ui_cwp,
// ui_cwp_lm) from p as already set, writes the selectors and every preset
// driven field. Does NOT run the precompute.
static inline void opendrt_v11_fill_presets(OpenDRTv11Params* p, int look, int tonescale, int encoding) {
    p->look_preset = look;
    p->tonescale_preset = tonescale;
    p->display_encoding_preset = encoding;

    const int li = (look >= 0 && look < ODT11_LOOK_COUNT) ? look : 0;
    odt11_apply_look_row(p, kOdt11LookTable[li]);

    if (tonescale >= 1 && tonescale < ODT11_TONESCALE_COUNT)
        odt11_apply_tonescale_row(p, kOdt11TonescaleTable[tonescale]);

    p->clamp = 1;

    if (p->ui_cwp != 0) p->cwp_lm = p->ui_cwp_lm;
    if (p->ui_cwp >= 1 && p->ui_cwp <= 6) p->cwp = p->ui_cwp - 1;

    const int ei = (encoding >= 0 && encoding < ODT11_ENCODING_COUNT) ? encoding : 0;
    p->tn_su = kOdt11EncodingTable[ei][0];
    p->display_gamut = kOdt11EncodingTable[ei][1];
    p->eotf = kOdt11EncodingTable[ei][2];
}

// DCTL UI defaults for the top-level controls, then preset mode with the
// default selectors (Standard, use look, Rec.1886), then the precompute.
static inline void opendrt_v11_defaults(OpenDRTv11Params* p) {
    OpenDRTv11Params z = {};
    *p = z;
    p->in_gamut = 15;   // DaVinci WG (DCTL default 14, bundle numbering 15)
    p->in_oetf = 1;     // DaVinci Intermediate
    p->tn_Lp = 100.0f;
    p->tn_gb = 0.13f;
    p->pt_hdr = 0.5f;
    p->tn_Lg = 10.0f;
    p->crv_enable = 0;
    p->ui_cwp = 0;
    p->ui_cwp_lm = 0.25f;
    opendrt_v11_fill_presets(p, 0, 0, 0);
    opendrt_v11_precompute(p);
}

#endif // !__CUDACC__ && !__METAL_VERSION__
