#pragma once
// TesseractParams.h
// Plain parameter block for the Tesseract view transform core
// (TesseractAlgorithm.h). Ported from upstream Tesseract v0.0.3 (GPL-3.0),
// open-display-transform commit 29df1e643e4c078cbdaac8e5402141c7750b6a58.
//
// Layout rules (same as OpenDRTParams.h): int and float only, both 4 bytes, no
// float3 members, 3x3 matrices stored as 9 row-major floats (m[r*3+c]) so a
// Metal or CUDA twin of this struct is byte identical to the host one.
//
// Two groups of fields:
//   1. UI fields: one per upstream DEFINE_UI_PARAMS, plus the 6A input
//      transfer index. A host fills these (tesseract_defaults fills upstream
//      defaults).
//   2. Resolved fields: matrices and tonescale constants written ONLY by
//      tesseract_precompute on the host. The kernel reads them; it never
//      reads the UI index fields for gamut selection.
//
// Index conventions (6A OpenDRT menus, see OpenDRT.cpp):
//   in_gamut      : 0 XYZ, 1 AP0, 2 AP1, 3 P3D65, 4 Rec2020, 5 Rec709, 6 AWG3,
//                   7 AWG4, 8 RWG, 9 SGamut3, 10 SGamut3Cine, 11 V-Gamut,
//                   12 BMDWG, 13 E-Gamut, 14 E-Gamut2, 15 DWG
//                   (upstream 0..10 map 1:1; upstream 11 BMDWG -> 12,
//                    12 E-Gamut -> 13, 13 DWG -> 15; 11 and 14 are 6A extras)
//   in_oetf       : 0 Linear, 1 DI, 2 T-Log, 3 ACEScct, 4 LogC3, 5 LogC4,
//                   6 Log3G10, 7 V-Log, 8 S-Log3, 9 F-Log2, 10 F-Log (v1)
//                   (upstream 0 Lin,1 DI,2 TLog,3 LogC3,4 LogC4,5 VLog,6 SLog3,
//                    7 FLog map to 0,1,2,4,5,7,8,10; 3,6,9 are 6A extras)
//   display_gamut : 0 Rec709, 1 P3D65, 2 Rec2020
//   eotf          : 0 Linear, 1 2.2 power, 2 2.4 power, 3 2.6 power, 4 PQ, 5 HLG

#include "../../../shared/LookDevPlatform.h"

typedef struct TesseractParams {
    // ---- UI: output / input selection -------------------------------------
    int   in_gamut;        // default 15 (DWG; upstream default 13 = DWG)
    int   in_oetf;         // default 0 (Linear)
    int   display_gamut;   // default 0 (Rec709)
    int   eotf;            // default 2 (2.4 power)
    int   clamp;           // checkbox, default 1

    // ---- UI: tonescale ------------------------------------------------------
    float Lp;              // 100   [100, 1000]
    float Lg;              // 10    [4, 20]
    float off;             // 0.005 [0, 0.02]
    float con;             // 1.6   [1, 2]

    // ---- UI: colour ---------------------------------------------------------
    float pb;              // purity 0.6 [0, 1]
    float hl_r;            // hue low r  0.3  [-1, 1]
    float hl_g;            // hue low g  0.2  [-1, 1]
    float hl_b;            // hue low b  0.2  [-1, 1]
    float hh_r;            // hue high r 0.5  [-1, 1]
    float hh_g;            // hue high g 0.0  [-1, 1]
    float hh_b;            // hue high b -0.5 [-1, 1]

    // ---- Resolved by tesseract_precompute (host only) ----------------------
    float in_to_xyz[9];       // row-major input gamut -> XYZ
    float xyz_to_display[9];  // row-major XYZ -> display gamut
    float ds;                 // display scale
    float ts_x0;              // tonescale constants (see tonescale() upstream)
    float ts_s0;
    float ts_o0;
    float ts_m0;
    float ts_m;
    float ts_s;
    float eotf_p;             // 2 + eotf*0.2
    int   _pad0;
} TesseractParams;

// 5 ints + 11 UI floats + 18 matrix floats + 8 resolved floats + 1 pad = 43
static_assert(sizeof(TesseractParams) == 43 * 4, "TesseractParams layout changed; update the GPU twins");
