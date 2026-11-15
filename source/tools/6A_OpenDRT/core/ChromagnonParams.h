#pragma once
// ChromagnonParams.h
// Plain parameter block for the Chromagnon view transform core
// (ChromagnonAlgorithm.h). Ported from upstream Chromagnon v0.0.7 (GPL-3.0),
// open-display-transform commit 29df1e643e4c078cbdaac8e5402141c7750b6a58.
//
// Layout rules (same as OpenDRTParams.h): int and float only, both 4 bytes, no
// float3 members, 3x3 matrices stored as 9 row-major floats (m[r*3+c]).
//
// UI fields mirror every upstream DEFINE_UI_PARAMS, plus the 6A input transfer
// index (in_oetf; upstream is linear-in only, so 0 reproduces upstream).
// Resolved fields are written ONLY by chromagnon_precompute on the host.
//
// Index conventions are the 6A ones listed in TesseractParams.h:
//   in_gamut 0..15 (default 15 = DWG, upstream default 13 = DWG),
//   in_oetf 0..10, display_gamut 0..2, eotf 0..5.
// direction: 0 = forward (upstream invert unchecked), 1 = inverse.

#include "../../../shared/LookDevPlatform.h"

typedef struct ChromagnonParams {
    // ---- UI: selection ------------------------------------------------------
    int   in_gamut;        // default 15 (DWG)
    int   in_oetf;         // default 0 (Linear; 6A addition)
    int   display_gamut;   // default 0 (Rec709)
    int   eotf;            // default 2 (2.4 power)
    int   clamp;           // checkbox, default 0
    int   direction;       // upstream "invert" checkbox, default 0

    // ---- UI: tonescale ------------------------------------------------------
    float Lp;              // 100   [100, 1000]
    float Lg;              // 10    [4, 20]
    float Lgb;             // Lg boost 0.12 [0, 0.5]
    float off;             // offset 0.005 [0, 0.02]
    float con;             // contrast 1.8 [1, 3]
    float conh;            // high contrast 0 [0, 0.5]
    float w0;              // high clip 128 [16, 256]

    // ---- UI: neutral axis ("split toning") ---------------------------------
    float wtm;             // white temp 0 [-0.6, 0.6]
    float wtn;             // white tint 0 [-1, 1]
    float gtm;             // grey temp  0 [-0.6, 0.6]
    float gtn;             // grey tint  0 [-1, 1]

    // ---- UI: purity compress ------------------------------------------------
    float pc_p;            // slope     0.5 [0, 1]
    float pc_x0;           // amount    0.1 [0, 0.5]
    float pc_t0;           // threshold 0.1 [0, 1]

    // ---- UI: render matrix --------------------------------------------------
    float rs_rnd;          // scale R  0.01  [0, 0.3]
    float gs_rnd;          // scale G  0.03  [0, 0.3]
    float bs_rnd;          // scale B  0.025 [0, 0.3]
    float rr_rnd;          // rotate R 0.01  [-0.1, 0.1]
    float gr_rnd;          // rotate G 0.0   [-0.1, 0.1]
    float br_rnd;          // rotate B 0.015 [-0.1, 0.1]

    // ---- UI: look matrix ----------------------------------------------------
    float rs_lk;           // R dist 0.03 [0, 0.3]
    float gs_lk;           // G dist 0.12 [0, 0.3]
    float bs_lk;           // B dist 0.02 [0, 0.3]

    // ---- Resolved by chromagnon_precompute (host only) ---------------------
    float in_to_xyz[9];
    float xyz_to_display[9];
    float display_to_xyz[9];  // inv_f33(xyz_to_display), inverse path
    float xyz_to_in[9];       // inv_f33(in_to_xyz), inverse path
    float rnd_mtx[9];
    float rnd_mtx_inv[9];
    float look_mtx[9];
    float look_mtx_inv[9];
    float ds;
    float clamp_max;
    float eotf_p;
    float pc_m;
    float c_s0;               // contrast() s0, t0, o0, m0
    float c_t0;
    float c_o0;
    float c_m0;
    float ts_m_r;             // tonescale() m per channel
    float ts_m_g;
    float ts_m_b;
    float ts_s_r;             // tonescale() s per channel
    float ts_s_g;
    float ts_s_b;
    int   _pad0;
} ChromagnonParams;

// 6 ints + 23 UI floats + 72 matrix floats + 14 resolved floats + 1 pad = 116
static_assert(sizeof(ChromagnonParams) == 116 * 4, "ChromagnonParams layout changed; update the GPU twins");
