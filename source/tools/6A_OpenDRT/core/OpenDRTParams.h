#pragma once
// OpenDRTParams.h — "6A. OpenDRT" wire-contract param struct.
//
// Adapted from the canonical OpenDRT OFX source:
//   /Volumes/Server Sync Files/OFX Plugin Git/Open DRT/OpenDRTParams.h
//
// APPEND-ONLY wire contract. The field ORDER and TYPES below mirror the
// canonical struct byte-for-byte so the same struct can later back a CUDA
// constant-mem upload (cudaMemcpyToSymbol) and a Metal buffer (the MSL twin
// declares the same field order). int and float are both 4 bytes; the
// int-then-float grouping packs identically on host C++, CUDA, and MSL.
//
// NOTE ON THE GOLDEN REFERENCE (MetalKernel.mm): the golden was re-anchored from
// CudaKernel.cu to MetalKernel.mm after an audit found the CUDA path is a stale
// transcription of Metal (it omits Hue Anchor Compression, Advanced Hue Contrast,
// and the host display-encoding remap). The per-pixel kernel READS: the
// precomputed tonescale fields (ts_x0, ts_s, ts_s1, ts_m2, ts_dsc), the raw UI
// fields, the Hue Anchor Compression fields (gated on hueCompressionEnable), and
// the Advanced Hue Contrast fields (gated on advHueContrast). The kernel computes
// its gamut matrices inline via switch() on inGamut/displayGamut/cwp and reads
// params.eotf / params.inOetf directly (the FINAL gamut/eotf, after the host
// display-encoding remap in opendrt_precompute). The matrix arrays and the
// oetfType/eotfType/oetfParams/eotfParams slots remain carried for byte layout
// (the device path does not read them — it switches on inOetf/eotf directly).
//
// sizeof(OpenDRTParams) stays ~1.6 KB (the HueCompressionParams sub-struct and
// the matrix/transfer-function arrays dominate). Verified by the harness.

#include "../../../shared/LookDevPlatform.h"

// ── Hue Anchor Compression sub-struct (carried verbatim; unused by kernel) ─────
struct OpenDRT_HueCompressionParams {
    float3 anchorBaseVectors[6];
    float  anchorRotations[6];
    float  anchorStrengths[6];
    float  anchorFalloffAngles[6];

    int    useSingleAnchorSettings;   // bool -> int for cross-backend layout
    int    gangRGBAnchors;
    int    gangCMYAnchors;
    int    globalRotationEnabled;
    float  globalRotation;
    float  globalStrength;
};

struct OpenDRTParams {
    // ── Input/Output settings (UI) ──
    int inGamut;
    int inOetf;

    // ── Tonescale (UI) ──
    float tnLp;           // Display Peak Luminance
    float tnGb;           // HDR Grey Boost
    float ptHdr;          // HDR Purity

    int   clamp;          // bool -> int
    float tnLg;           // Grey Luminance
    float tnCon;          // Contrast
    float tnSh;           // Shoulder Clip
    float tnToe;          // Toe
    float tnOff;          // Offset

    // ── High Contrast (UI) ──
    float tnHcon;
    float tnHconPv;
    float tnHconSt;

    // ── Low Contrast (UI) ──
    float tnLcon;
    float tnLconW;
    float tnLconPc;

    // ── Creative White (UI) ──
    int   cwp;
    float cwpRng;

    // ── Render Space (UI) ──
    float rsSa;
    float rsRw;
    float rsBw;

    // ── Purity Compress (UI) ──
    float ptR, ptG, ptB;
    float ptRngLow;
    float ptRngHigh;

    // ── Mid Purity (UI) ──
    float ptmLow;
    float ptmLowSt;
    float ptmHigh;
    float ptmHighSt;

    // ── Brilliance (UI) ──
    float brlR, brlG, brlB;
    float brlC, brlM, brlY;
    float brlRng;

    // ── Hueshift RGB (UI) ──
    float hsR, hsG, hsB;
    float hsRgbRng;

    // ── Hueshift CMY (UI) ──
    float hsC, hsM, hsY;

    // ── Hue Contrast (UI) ──
    float hcR;

    // ── Advanced Hue Contrast (UI; READ by kernel, gated on advHueContrast) ──
    float advHcR;
    float advHcG;
    float advHcB;
    float advHcC;
    float advHcM;
    float advHcY;
    float advHcPower;

    // ── Filmic Mode + advanced controls (UI) ──
    int   filmicMode;
    float filmicDynamicRange;
    int   filmicProjectorSim;     // carried, NOT read by kernel
    float filmicSourceStops;
    float filmicTargetStops;
    float filmicStrength;
    int   advHueContrast;         // READ by kernel (gates Advanced Hue Contrast)
    int   tonescaleMap;
    int   diagnosticsMode;
    int   rgbChipsMode;
    int   betaFeaturesEnable;

    // ── Display (UI) ──
    int   displayGamut;
    int   eotf;

    // ── Matrix data (carried for layout; NOT read by kernel) ──
    float inputMatrix[9];
    float outputMatrix[9];
    float cwpMatrix[9];
    float xyzToP3Matrix[9];
    float p3ToRec709Matrix[9];

    // ── Transfer function slots (carried; NOT read by kernel) ──
    float oetfParams[8];
    float eotfParams[8];
    int   oetfType;
    int   eotfType;

    // ── PRECOMPUTED tonescale constants (host fills; kernel READS these) ──
    // The kernel consumes only ts_x0, ts_m2, ts_s, ts_dsc, ts_s1. The rest are
    // intermediates kept for layout parity with the canonical struct.
    float ts_x1;
    float ts_y1;
    float ts_x0;          // 0.18 + tnOff                         (READ by kernel)
    float ts_y0;
    float ts_s0;
    float ts_s10;
    float ts_m1;
    float ts_m2;          // compress_toe_quadratic(ts_m1,...)    (READ by kernel)
    float ts_s;           //                                      (READ by kernel)
    float ts_dsc;         // display scale factor                 (READ by kernel)
    float pt_cmp_Lf;
    float s_Lp100;
    float ts_s1;          //                                      (READ by kernel)
    // Additional tonescale slots (carried for layout; unused)
    float ts_g;
    float ts_c;
    float ts_n;
    float ts_k;
    float ts_ip;
    float ts_cp;
    float ts_w2;
    float ts_t;
    float ts_nd;

    // ── Module enable flags (carried for layout) ──
    int tnHconEnable;
    int tnLconEnable;
    int ptlEnable;
    int ptmEnable;
    int brlEnable;
    int hsRgbEnable;
    int hsCmyEnable;

    // ── Hue Anchor Compression (READ by kernel, gated on hueCompressionEnable) ──
    int   hueCompressionEnable;
    float hueAnchorRotations[6];
    float hueAnchorStrengths[6];
    float hueAnchorFalloffAngles[6];
    int   hueUseSingleAnchorSettings;
    int   hueGangRGBAnchors;
    int   hueGangCMYAnchors;
    int   hueGlobalRotationEnabled;
    float hueGlobalRotation;
    float hueGlobalStrength;

    // ── UI enable flags (READ by kernel as the *UIEnable gates) ──
    int tnHconUIEnable;
    int tnLconUIEnable;
    int ptlUIEnable;
    int ptmUIEnable;
    int brlUIEnable;
    int hsRgbUIEnable;
    int hsCmyUIEnable;
    int hcUIEnable;

    // ── Preset enable flags (READ by kernel as the *PresetEnable gates) ──
    int tnHconPresetEnable;
    int tnLconPresetEnable;
    int ptlPresetEnable;
    int ptmPresetEnable;
    int brlPresetEnable;
    int hsRgbPresetEnable;
    int hsCmyPresetEnable;
    int hcPresetEnable;

    // ── Hue Anchor Compression sub-struct (carried; NOT read by kernel) ──
    OpenDRT_HueCompressionParams hueCompression;
};
