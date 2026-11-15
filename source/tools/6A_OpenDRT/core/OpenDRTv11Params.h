#pragma once
// OpenDRTv11Params.h
// Parameter block for the second OpenDRT model ("model 1"): a port of the pixel
// math of upstream OpenDRT v1.1.0 (GPL-3.0), commit af68332 (HEAD used for the
// port; tag v1.1.0 = 8804747). Not wired into the plugin yet.
//
// Layout rules:
//   - Plain C struct of 4-byte int and float fields only (no float3), so it packs
//     identically on host C++, CUDA and MSL and can later be appended to the
//     bundle OpenDRTParams as a sub-struct.
//   - Groups are contiguous. New fields go at the END of the struct only.
//   - The size is pinned by the static_assert at the bottom.
//
// Field names follow the DCTL variable names. Every DCTL UI parameter (the
// visible preset UI and the commented stickshift UI) has a field here.
//
// Index conventions:
//   in_gamut uses the BUNDLE numbering (16 entries), not the DCTL numbering:
//     0 XYZ, 1 AP0, 2 AP1, 3 P3-D65, 4 Rec.2020, 5 Rec.709, 6 AWG3, 7 AWG4,
//     8 RWG, 9 S-Gamut3, 10 S-Gamut3.Cine, 11 V-Gamut,
//     12 Blackmagic Wide Gamut (bundle only, not in the DCTL),
//     13 E-Gamut (DCTL 12), 14 E-Gamut2 (DCTL 13), 15 DaVinci WG (DCTL 14).
//   in_oetf matches both bundle and DCTL (10 entries):
//     0 Linear, 1 DaVinci Intermediate, 2 T-Log, 3 ACEScct, 4 LogC3, 5 LogC4,
//     6 Log3G10, 7 V-Log, 8 S-Log3, 9 F-Log2.
//   display_gamut: 0 Rec.709, 1 P3-D65, 2 Rec.2020 (P3 limited), 3 P3-D60,
//     4 P3-DCI, 5 XYZ.
//   eotf: 0 Linear, 1 2.2 power, 2 2.4 power, 3 2.6 power, 4 PQ, 5 HLG.
//   cwp: 0 D93, 1 D75, 2 D65, 3 D60, 4 D55, 5 D50.
//   tn_su: 0 Dark, 1 Dim, 2 Bright.

struct OpenDRTv11Params {
    // ---- Group A: input (2 fields) ----------------------------------------
    int   in_gamut;          // bundle numbering, default 15 (DaVinci WG)
    int   in_oetf;           // default 1 (DaVinci Intermediate)

    // ---- Group B: top-level UI, not preset driven (5 fields) --------------
    float tn_Lp;             // display peak luminance, 100..1000, default 100
    float tn_gb;             // HDR grey boost, 0..1, default 0.13
    float pt_hdr;            // HDR purity, 0..1, default 0.5
    float tn_Lg;             // grey luminance, 3..25, default 10
    int   crv_enable;        // tonescale overlay, default 0

    // ---- Group C: preset selectors (5 fields) ------------------------------
    // Consumed by opendrt_v11_fill_presets (host). The kernel never reads them.
    int   look_preset;       // 0..6 visible, 7 = Base (hidden in the DCTL)
    int   tonescale_preset;  // 0 = use look, 1..13
    int   ui_cwp;            // DCTL "_cwp" combo: 0 = use look, 1..6 = D93..D50
    float ui_cwp_lm;         // DCTL "_cwp_lm", 0..1, default 0.25
    int   display_encoding_preset; // 0..8

    // ---- Group D: tonescale (11 fields) ------------------------------------
    float tn_con;
    float tn_sh;
    float tn_toe;
    float tn_off;
    int   tn_hcon_enable;
    float tn_hcon;
    float tn_hcon_pv;
    float tn_hcon_st;
    int   tn_lcon_enable;
    float tn_lcon;
    float tn_lcon_w;

    // ---- Group E: creative white (2 fields) --------------------------------
    int   cwp;
    float cwp_lm;

    // ---- Group F: render space (3 fields) ----------------------------------
    float rs_sa;
    float rs_rw;
    float rs_bw;

    // ---- Group G: purity (19 fields) ---------------------------------------
    int   pt_enable;         // carried for fidelity; the DCTL never reads it
    float pt_lml;
    float pt_lml_r;
    float pt_lml_g;
    float pt_lml_b;
    float pt_lmh;
    float pt_lmh_r;
    float pt_lmh_b;
    int   ptl_enable;
    float ptl_c;
    float ptl_m;
    float ptl_y;
    int   ptm_enable;
    float ptm_low;
    float ptm_low_rng;
    float ptm_low_st;
    float ptm_high;
    float ptm_high_rng;
    float ptm_high_st;

    // ---- Group H: brilliance (12 fields) -----------------------------------
    int   brl_enable;
    float brl;
    float brl_r;
    float brl_g;
    float brl_b;
    float brl_rng;
    float brl_st;
    int   brlp_enable;
    float brlp;
    float brlp_r;
    float brlp_g;
    float brlp_b;

    // ---- Group I: hue contrast and hue shift (17 fields) -------------------
    int   hc_enable;
    float hc_r;
    float hc_r_rng;
    int   hs_rgb_enable;
    float hs_r;
    float hs_r_rng;
    float hs_g;
    float hs_g_rng;
    float hs_b;
    float hs_b_rng;
    int   hs_cmy_enable;
    float hs_c;
    float hs_c_rng;
    float hs_m;
    float hs_m_rng;
    float hs_y;
    float hs_y_rng;

    // ---- Group J: display (4 fields) ---------------------------------------
    int   tn_su;
    int   display_gamut;
    int   eotf;
    int   clamp;

    // ---- Group K: host precompute, written by opendrt_v11_precompute -------
    // (14 fields; the DCTL computes these per pixel in the same order)
    float ts_x1;
    float ts_y1;
    float ts_x0;
    float ts_y0;
    float ts_s0;
    float ts_p;
    float ts_s10;
    float ts_m1;
    float ts_m2;
    float ts_s;
    float ts_dsc;
    float pt_cmp_Lf;
    float s_Lp100;
    float ts_s1;
};

// 2 + 5 + 5 + 11 + 2 + 3 + 19 + 12 + 17 + 4 + 14 = 94 four-byte fields.
static_assert(sizeof(OpenDRTv11Params) == 94 * 4, "OpenDRTv11Params layout changed");
