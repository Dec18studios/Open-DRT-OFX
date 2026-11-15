#pragma once
// OpenDRTv11Algorithm.h
// Per-pixel math of the second OpenDRT model ("model 1"): a port of upstream
// OpenDRT v1.1.0 (GPL-3.0), commit af68332, file
// display-transforms/opendrt/OpenDRT.dctl. Not wired into the plugin yet.
//
// Port rules followed here:
//   - DCTL operation order is kept statement by statement, including the
//     left-to-right association of every sum and product.
//   - Every helper is a new odt11_ function. No bundle odt_ helper is reused, so
//     this header depends only on LookDevPlatform.h.
//   - float3 arithmetic that the platform header does not provide (float3*float3,
//     float3+float) is written out per component in the DCTL order.
//   - Matrices are the DCTL row-major literals, verbatim. odt11_vdot computes
//     row . v exactly like the DCTL vdot.
//   - Tonescale constants are read from the params block (filled host side by
//     opendrt_v11_precompute, same formulas and order as the DCTL).
//
// Deviations from the DCTL (all listed again in the selftest report):
//   1. _exp10f(x) is written as _powf(10.0f, x) (odt11_exp10f), because the
//      platform header has no _exp10f alias. Possible last-ulp differences.
//   2. HLG inverse: sqrt(3x) is written sqrt(fmax(0, 3x)). Only changes inputs
//      where the DCTL returns NaN (negative linear light with clamp off).
//   3. in_gamut uses the bundle 16-entry numbering (BMDWG at 12); the DCTL has
//      15 entries with E-Gamut at 12.
//   4. The DCTL PI is a double literal stored in a float and hue_offset uses
//      the double 2.0*PI; here both are float. 2*PI is exact in float, so the
//      values are identical.
//   5. Pixel x, y are taken as given. The DCTL convention is y = 0 at the top
//      row for the overlay; the host must pass that convention.

#include "../../../shared/LookDevPlatform.h"
#include "OpenDRTv11Params.h"

#define ODT11_SQRT3 1.73205080756887729353f
#define ODT11_PI    3.14159265358979323846f

// ---- Matrices (DCTL row-major, verbatim literals) --------------------------
// Each macro expands to three float3 rows: row0, row1, row2.
#define ODT11_M_AP0_TO_XYZ make_float3(0.938630948750273197f, -0.00574192055037397141f, 0.017566898851772296f), make_float3(0.338093594922021567f, 0.72721390281143572f, -0.0653074977334571899f), make_float3(0.000723121511341165988f, 0.000818441849244731985f, 1.08751618739929268f)
#define ODT11_M_AP1_TO_XYZ make_float3(0.652418717671912951f, 0.127179925537538263f, 0.170857283842220459f), make_float3(0.268064059194271287f, 0.672464478992617742f, 0.0594714618131108388f), make_float3(-0.0054699285104975676f, 0.00518279997697511721f, 1.08934487929340107f)
#define ODT11_M_REC2020_TO_XYZ make_float3(0.636958048301290991f, 0.144616903586208406f, 0.168880975164172054f), make_float3(0.26270021201126692f, 0.677998071518871148f, 0.0593017164698619384f), make_float3(4.9999999999999999e-17f, 0.0280726930490874452f, 1.06098505771079066f)
#define ODT11_M_AWG3_TO_XYZ make_float3(0.638007619284f, 0.214703856337f, 0.097744451431f), make_float3(0.291953779f, 0.823841041511f, -0.11579482051f), make_float3(0.002798279032f, -0.067034235689f, 1.15329370742f)
#define ODT11_M_AWG4_TO_XYZ make_float3(0.704858320407231953f, 0.129760295170463003f, 0.115837311473976537f), make_float3(0.254524176404026969f, 0.781477732712002049f, -0.0360019091160290391f), make_float3(0.0f, 0.0f, 1.08905775075987843f)
#define ODT11_M_REDWG_TO_XYZ make_float3(0.735275245905858799f, 0.0686094106139610721f, 0.14657127053185201f), make_float3(0.286694099499934962f, 0.842979134016975662f, -0.129673233516910319f), make_float3(-0.0796808568783676785f, -0.347343216994429771f, 1.51608182463267593f)
#define ODT11_M_SGAMUT3_TO_XYZ make_float3(0.706482713192318812f, 0.12880104979055762f, 0.115172164068795255f), make_float3(0.270979670813492168f, 0.786606411220905466f, -0.0575860820343976273f), make_float3(-0.00967784538619615754f, 0.00460003749251991934f, 1.09413555865355483f)
#define ODT11_M_SGAMUT3CINE_TO_XYZ make_float3(0.599083920758327171f, 0.248925516115423628f, 0.102446490177920776f), make_float3(0.215075820115587457f, 0.88506850174372842f, -0.100144321859315821f), make_float3(-0.0320658495445057951f, -0.0276583906794915374f, 1.1487819909838759f)
#define ODT11_M_VGAMUT_TO_XYZ make_float3(0.679644469878f, 0.15221141244f, 0.118600044733f), make_float3(0.26068555009f, 0.77489446333f, -0.03558001342f), make_float3(-0.009310198218f, -0.004612467044f, 1.10298041602f)
#define ODT11_M_EGAMUT_TO_XYZ make_float3(0.705396850087770755f, 0.164041328309919021f, 0.0810177486539819941f), make_float3(0.280130724091105898f, 0.820206641549595106f, -0.100337365640700782f), make_float3(-0.103781511569163279f, -0.0729072570266306313f, 1.26574651935567273f)
#define ODT11_M_EGAMUT2_TO_XYZ make_float3(0.736477700183697404f, 0.130739651086660136f, 0.0832385757813140781f), make_float3(0.275069984405959256f, 0.828017790215514138f, -0.103087774621473588f), make_float3(-0.124225154247852534f, -0.0871597673911067433f, 1.30044267239883782f)
#define ODT11_M_DAVINCIWG_TO_XYZ make_float3(0.700622392093671609f, 0.148774815123196763f, 0.101058719834803246f), make_float3(0.274118510906649016f, 0.873631895940436665f, -0.147750406847085763f), make_float3(-0.0989629128832311411f, -0.137895325075543307f, 1.32591598871865268f)
// Bundle-only entry (not in the DCTL): Blackmagic Wide Gamut, row-major form of
// the bundle column-major literals.
#define ODT11_M_BMDWG_TO_XYZ make_float3(0.606538414955f, 0.220412746072f, 0.123504832387f), make_float3(0.267992943525f, 0.832748472691f, -0.100741356611f), make_float3(-0.029442556202f, -0.086612440646f, 1.205112814903f)

#define ODT11_M_P3D65_TO_XYZ make_float3(0.486570948648216151f, 0.265667693169093f, 0.198217285234362467f), make_float3(0.228974564069748754f, 0.691738521836506193f, 0.079286914093744984f), make_float3(-4.00000000000000029e-17f, 0.0451133818589026167f, 1.04394436890097575f)
#define ODT11_M_XYZ_TO_P3D65 make_float3(2.49349691194142542f, -0.93138361791912383f, -0.402710784450716841f), make_float3(-0.829488969561574696f, 1.76266406031834655f, 0.0236246858419435941f), make_float3(0.0358458302437844531f, -0.0761723892680418041f, 0.956884524007687309f)
#define ODT11_M_REC709_TO_XYZ make_float3(0.412390799265959229f, 0.357584339383878125f, 0.180480788401834347f), make_float3(0.212639005871510217f, 0.71516867876775625f, 0.0721923153607337414f), make_float3(0.0193308187155918181f, 0.119194779794626018f, 0.950532152249661033f)
#define ODT11_M_XYZ_TO_REC709 make_float3(3.24096994190452348f, -1.53738317757009435f, -0.498610760293003552f), make_float3(-0.969243636280879506f, 1.87596750150771996f, 0.0415550574071755843f), make_float3(0.0556300796969936354f, -0.20397695888897649f, 1.05697151424287816f)
#define ODT11_M_P3_TO_REC2020 make_float3(0.753833034361722221f, 0.198597369052616435f, 0.0475695965856618441f), make_float3(0.0457438489653582137f, 0.9417772198116936f, 0.0124789312229481135f), make_float3(-0.0012103403545183941f, 0.0176017173010899926f, 0.983608623053428777f)

#define ODT11_M_CAT_DCI_TO_D93 make_float3(0.965685009956359863f, 0.00183745240792632103f, 0.0912967324256896973f), make_float3(0.000514572137035429044f, 0.965166747570037842f, 0.036014653742313385f), make_float3(0.00154250487685203596f, 0.00702651776373386383f, 1.47287476062774658f)
#define ODT11_M_CAT_DCI_TO_D75 make_float3(0.990120768547058105f, 0.0151389474049210548f, 0.0511047691106796265f), make_float3(0.0102197211235761642f, 0.971718132495880127f, 0.0200536623597145081f), make_float3(0.000743072712793946049f, 0.0042176349088549614f, 1.27959656715393066f)
#define ODT11_M_CAT_DCI_TO_D65 make_float3(1.00951600074768066f, 0.0269675441086292267f, 0.0213620811700820923f), make_float3(0.0187991037964820862f, 0.975330352783203125f, 0.0082273334264755249f), make_float3(0.000134543282911180989f, 0.00217903498560190201f, 1.138663649559021f)
#define ODT11_M_CAT_DCI_TO_D60 make_float3(1.02159523963928223f, 0.034848678857088089f, 0.00371252000331878705f), make_float3(0.0244968775659799576f, 0.976937234401702881f, 0.00120301544666290305f), make_float3(-0.00023391586728393999f, 0.000986687839031219049f, 1.05594265460968018f)
#define ODT11_M_CAT_DCI_TO_D55 make_float3(1.03594577312469482f, 0.0450937561690807343f, -0.0157573819160461426f), make_float3(0.0318740680813789368f, 0.977744519710540771f, -0.00655744969844818115f), make_float3(-0.000653609400615095984f, -0.000297372229397297014f, 0.966327786445617676f)
#define ODT11_M_CAT_DCI_TO_D50 make_float3(1.05306875705718994f, 0.0581297315657138824f, -0.0376100838184356689f), make_float3(0.0412359423935413361f, 0.977693676948547363f, -0.0152792222797870636f), make_float3(-0.00113777676597237609f, -0.00170759297907352404f, 0.867368340492248535f)

#define ODT11_M_CAT_D65_TO_D93 make_float3(0.95703423023223877f, -0.0247171502560377121f, 0.0624028593301773071f), make_float3(-0.0179296955466270447f, 0.990019857883453369f, 0.0248119533061981201f), make_float3(0.00127589143812656403f, 0.00427919067442417058f, 1.29345715045928955f)
#define ODT11_M_CAT_D65_TO_D75 make_float3(0.981001079082489014f, -0.0116619253531098366f, 0.0265614092350006104f), make_float3(-0.00843488052487373352f, 0.996506094932556152f, 0.0105696544051170349f), make_float3(0.000552809564396739006f, 0.00179840810596942902f, 1.12374722957611084f)
#define ODT11_M_CAT_D65_TO_D60 make_float3(1.01182246208190918f, 0.00778879318386316299f, -0.0157783031463623047f), make_float3(0.00561682833358645439f, 1.00150644779205322f, -0.00628517568111419678f), make_float3(-0.000335735734552145004f, -0.0010509500280022619f, 0.927366673946380615f)
#define ODT11_M_CAT_D65_TO_D55 make_float3(1.02585089206695557f, 0.0179439820349216461f, -0.0332137793302536011f), make_float3(0.0129133854061365128f, 1.00214779376983643f, -0.0132421031594276428f), make_float3(-0.000719940289855003032f, -0.00218106806278228803f, 0.84868013858795166f)
#define ODT11_M_CAT_D65_TO_D50 make_float3(1.04257404804229736f, 0.03089117631316185f, -0.052812620997428894f), make_float3(0.0221935361623764038f, 1.00185668468475342f, -0.0210737623274326324f), make_float3(-0.00116488314233720303f, -0.00342052709311246915f, 0.761789083480834961f)
#define ODT11_M_CAT_D65_TO_DCI make_float3(0.991085588932037354f, -0.0273622870445251465f, -0.0183956623077392578f), make_float3(-0.0191021915525197983f, 1.02583777904510498f, -0.00705372542142868042f), make_float3(-8.05503223091359977e-05f, -0.00195988826453685804f, 0.878238439559936523f)

#define ODT11_M_CAT_D60_TO_D93 make_float3(0.946056902408599854f, -0.0319503024220466614f, 0.0831701457500457764f), make_float3(-0.0231979694217443466f, 0.988745808601379395f, 0.0330617502331733704f), make_float3(0.00169203430414199807f, 0.00572328735142946243f, 1.39483106136322021f)
#define ODT11_M_CAT_D60_TO_D75 make_float3(0.969659984111785889f, -0.019138311967253685f, 0.0450099557638168335f), make_float3(-0.0138545772060751915f, 0.995133817195892334f, 0.0179062262177467346f), make_float3(0.000931452261283994046f, 0.00306008197367191315f, 1.21179807186126709f)
#define ODT11_M_CAT_D60_TO_D65 make_float3(0.988363921642303467f, -0.00766910053789615631f, 0.0167641639709472656f), make_float3(-0.0055409618653357029f, 0.998546123504638672f, 0.00667332112789154139f), make_float3(0.000351537019014359008f, 0.00112883746623992898f, 1.07833576202392578f)
#define ODT11_M_CAT_D60_TO_D55 make_float3(1.01380288600921631f, 0.0100131509825587273f, -0.018498346209526062f), make_float3(0.00720565160736441612f, 1.00057685375213623f, -0.00737529993057250977f), make_float3(-0.000401133671402930997f, -0.00121434964239597299f, 0.915135681629180908f)
#define ODT11_M_CAT_D60_TO_D50 make_float3(1.03025269508361816f, 0.0227910466492176056f, -0.0392656922340393066f), make_float3(0.0163766480982303619f, 1.00020599365234375f, -0.0156668238341808319f), make_float3(-0.000864576781168579947f, -0.00254668481647968292f, 0.821422040462493896f)

// ---- Math helpers (DCTL "Math helper functions") ---------------------------

// DCTL vdot: row . v for each of the three rows.
LDT_FUNC float3 odt11_vdot(float3 r0, float3 r1, float3 r2, float3 v) {
    return make_float3(r0.x*v.x + r0.y*v.y + r0.z*v.z,
                       r1.x*v.x + r1.y*v.y + r1.z*v.z,
                       r2.x*v.x + r2.y*v.y + r2.z*v.z);
}

LDT_FUNC float odt11_sdivf(float a, float b) {
    if (b == 0.0f) return 0.0f;
    else return a/b;
}

LDT_FUNC float3 odt11_sdivf3f(float3 a, float b) {
    return make_float3(odt11_sdivf(a.x, b), odt11_sdivf(a.y, b), odt11_sdivf(a.z, b));
}

LDT_FUNC float odt11_spowf(float a, float b) {
    if (a <= 0.0f) return a;
    else return _powf(a, b);
}

LDT_FUNC float3 odt11_spowf3(float3 a, float b) {
    return make_float3(odt11_spowf(a.x, b), odt11_spowf(a.y, b), odt11_spowf(a.z, b));
}

// DCTL hypotf2 takes a float2; the two components are passed separately here
// (float2 is not uniformly available on every backend).
LDT_FUNC float odt11_hypotf2(float x, float y) { return _sqrtf(_fmaxf(0.0f, x*x + y*y)); }

LDT_FUNC float odt11_hypotf3(float3 v) { return _sqrtf(_fmaxf(0.0f, v.x*v.x + v.y*v.y + v.z*v.z)); }

LDT_FUNC float3 odt11_clampminf3(float3 a, float mn) {
    return make_float3(_fmaxf(a.x, mn), _fmaxf(a.y, mn), _fmaxf(a.z, mn));
}

LDT_FUNC float odt11_clampf(float a, float mn, float mx) { return _fminf(_fmaxf(a, mn), mx); }

LDT_FUNC float3 odt11_clampf3(float3 a, float mn, float mx) {
    return make_float3(odt11_clampf(a.x, mn, mx), odt11_clampf(a.y, mn, mx), odt11_clampf(a.z, mn, mx));
}

// Deviation 1: DCTL _exp10f(x).
LDT_FUNC float odt11_exp10f(float x) { return _powf(10.0f, x); }

// ---- OETF linearization (DCTL "OETF Linearization Transfer Functions") -----
LDT_FUNC float odt11_oetf_davinci_intermediate(float x) {
    return x <= 0.02740668f ? x/10.44426855f : _exp2f(x/0.07329248f - 7.0f) - 0.0075f;
}
LDT_FUNC float odt11_oetf_filmlight_tlog(float x) {
    return x < 0.075f ? (x-0.075f)/16.184376489665897f : _expf((x - 0.5520126568606655f)/0.09232902596577353f) - 0.0057048244042473785f;
}
LDT_FUNC float odt11_oetf_acescct(float x) {
    return x <= 0.155251141552511f ? (x - 0.0729055341958355f)/10.5402377416545f : _exp2f(x*17.52f - 9.72f);
}
LDT_FUNC float odt11_oetf_arri_logc3(float x) {
    return x < 5.367655f*0.010591f + 0.092809f ? (x - 0.092809f)/5.367655f : (odt11_exp10f((x - 0.385537f)/0.247190f) - 0.052272f)/5.555556f;
}
LDT_FUNC float odt11_oetf_arri_logc4(float x) {
    return x < -0.7774983977293537f ? x*0.3033266726886969f - 0.7774983977293537f : (_exp2f(14.0f*(x - 0.09286412512218964f)/0.9071358748778103f + 6.0f) - 64.0f)/2231.8263090676883f;
}
LDT_FUNC float odt11_oetf_red_log3g10(float x) {
    return x < 0.0f ? (x/15.1927f) - 0.01f : (odt11_exp10f(x/0.224282f) - 1.0f)/155.975327f - 0.01f;
}
LDT_FUNC float odt11_oetf_panasonic_vlog(float x) {
    return x < 0.181f ? (x - 0.125f)/5.6f : odt11_exp10f((x - 0.598206f)/0.241514f) - 0.00873f;
}
LDT_FUNC float odt11_oetf_sony_slog3(float x) {
    return x < 171.2102946929f/1023.0f ? (x*1023.0f - 95.0f)*0.01125f/(171.2102946929f - 95.0f) : (odt11_exp10f(((x*1023.0f - 420.0f)/261.5f))*(0.18f + 0.01f) - 0.01f);
}
LDT_FUNC float odt11_oetf_fujifilm_flog2(float x) {
    return x < 0.100686685370811f ? (x - 0.092864f)/8.799461f : (odt11_exp10f(((x - 0.384316f)/0.245281f))/5.555556f - 0.064829f/5.555556f);
}

LDT_FUNC float odt11_oetf(float x, int tf) {
    if (tf == 1) return odt11_oetf_davinci_intermediate(x);
    else if (tf == 2) return odt11_oetf_filmlight_tlog(x);
    else if (tf == 3) return odt11_oetf_acescct(x);
    else if (tf == 4) return odt11_oetf_arri_logc3(x);
    else if (tf == 5) return odt11_oetf_arri_logc4(x);
    else if (tf == 6) return odt11_oetf_red_log3g10(x);
    else if (tf == 7) return odt11_oetf_panasonic_vlog(x);
    else if (tf == 8) return odt11_oetf_sony_slog3(x);
    else if (tf == 9) return odt11_oetf_fujifilm_flog2(x);
    return x;
}

LDT_FUNC float3 odt11_linearize(float3 rgb, int tf) {
    if (tf == 0) return rgb;
    return make_float3(odt11_oetf(rgb.x, tf), odt11_oetf(rgb.y, tf), odt11_oetf(rgb.z, tf));
}

// ---- Inverse display EOTFs --------------------------------------------------
// HLG inverse for 1000 nits (DCTL eotf_hlg, inverse == 1 branch only).
// Deviation 2: sqrt argument guarded with fmax(0, .).
LDT_FUNC float odt11_hlg_oetf1(float x) {
    return x <= 1.0f/12.0f ? _sqrtf(_fmaxf(0.0f, 3.0f*x)) : 0.17883277f*_logf(12.0f*x - 0.28466892f) + 0.55991073f;
}
LDT_FUNC float3 odt11_eotf_hlg_inv(float3 rgb) {
    float Yd = 0.2627f*rgb.x + 0.6780f*rgb.y + 0.0593f*rgb.z;
    rgb = rgb*odt11_spowf(Yd, (1.0f - 1.2f)/1.2f);
    return make_float3(odt11_hlg_oetf1(rgb.x), odt11_hlg_oetf1(rgb.y), odt11_hlg_oetf1(rgb.z));
}

// ST 2084 PQ inverse, input normalized so 1.0 = 10000 nits (DCTL eotf_pq,
// inverse == 1 branch only).
LDT_FUNC float odt11_pq_inv1(float x) {
    const float c1 = 107.0f/128.0f;
    const float c2 = 2413.0f/128.0f;
    const float c3 = 2392.0f/128.0f;
    return (c1 + c2*x)/(1.0f + c3*x);
}
LDT_FUNC float3 odt11_eotf_pq_inv(float3 rgb) {
    const float m1 = 2610.0f/16384.0f;
    const float m2 = 2523.0f/32.0f;
    rgb = odt11_spowf3(rgb, m1);
    rgb = odt11_spowf3(make_float3(odt11_pq_inv1(rgb.x), odt11_pq_inv1(rgb.y), odt11_pq_inv1(rgb.z)), m2);
    return rgb;
}

// ---- OpenDRT functions (DCTL "Functions for OpenDRT") ----------------------
LDT_FUNC float odt11_compress_hyperbolic_power(float x, float s, float p) {
    return odt11_spowf(x/(x + s), p);
}

LDT_FUNC float odt11_compress_toe_quadratic(float x, float toe, int inv) {
    if (toe == 0.0f) return x;
    if (inv == 0) {
        return odt11_spowf(x, 2.0f)/(x + toe);
    } else {
        return (x + _sqrtf(x*(4.0f*toe + x)))/2.0f;
    }
}

LDT_FUNC float odt11_compress_toe_cubic(float x, float m, float w, int inv) {
    if (m == 1.0f) return x;
    float x2 = x*x;
    if (inv == 0) {
        return x*(x2 + m*w)/(x2 + w);
    } else {
        float p0 = x2 - 3.0f*m*w;
        float p1 = 2.0f*x2 + 27.0f*w - 9.0f*m*w;
        // DCTL writes the integer literal 4; 4.0f gives the same value.
        float p2 = _powf(_sqrtf(x2*p1*p1 - 4.0f*p0*p0*p0)/2.0f + x*p1/2.0f, 1.0f/3.0f);
        return p0/(3.0f*p2) + p2/3.0f + x/3.0f;
    }
}

LDT_FUNC float odt11_contrast_high(float x, float p, float pv, float pv_lx, int inv) {
    const float x0 = 0.18f*_powf(2.0f, pv);
    if (x < x0 || p == 1.0f) return x;

    const float o = x0 - x0/p;
    const float s0 = _powf(x0, 1.0f - p)/p;
    const float x1 = x0*_powf(2.0f, pv_lx);
    const float k1 = p*s0*_powf(x1, p)/x1;
    const float y1 = s0*_powf(x1, p) + o;
    if (inv == 1)
        return x > y1 ? (x - y1)/k1 + x1 : _powf((x - o)/s0, 1.0f/p);
    else
        return x > x1 ? k1*(x - x1) + y1 : s0*_powf(x, p) + o;
}

LDT_FUNC float odt11_softplus(float x, float s) {
    if (x > 10.0f*s || s < 1e-4f) return x;
    return s*_logf(_fmaxf(0.0f, 1.0f + _expf(x/s)));
}

LDT_FUNC float odt11_gauss_window(float x, float w) {
    return _expf(-x*x/w);
}

// Deviation 4: float 2*PI (exact) in place of the DCTL double 2.0*PI.
LDT_FUNC float odt11_hue_offset(float h, float o) {
    return _fmod(h - o + ODT11_PI, 2.0f*ODT11_PI) - ODT11_PI;
}

// ---- Input gamut to XYZ (bundle numbering, deviation 3) --------------------
LDT_FUNC float3 odt11_in_to_xyz(float3 rgb, int g) {
    if (g == 1) return odt11_vdot(ODT11_M_AP0_TO_XYZ, rgb);
    else if (g == 2) return odt11_vdot(ODT11_M_AP1_TO_XYZ, rgb);
    else if (g == 3) return odt11_vdot(ODT11_M_P3D65_TO_XYZ, rgb);
    else if (g == 4) return odt11_vdot(ODT11_M_REC2020_TO_XYZ, rgb);
    else if (g == 5) return odt11_vdot(ODT11_M_REC709_TO_XYZ, rgb);
    else if (g == 6) return odt11_vdot(ODT11_M_AWG3_TO_XYZ, rgb);
    else if (g == 7) return odt11_vdot(ODT11_M_AWG4_TO_XYZ, rgb);
    else if (g == 8) return odt11_vdot(ODT11_M_REDWG_TO_XYZ, rgb);
    else if (g == 9) return odt11_vdot(ODT11_M_SGAMUT3_TO_XYZ, rgb);
    else if (g == 10) return odt11_vdot(ODT11_M_SGAMUT3CINE_TO_XYZ, rgb);
    else if (g == 11) return odt11_vdot(ODT11_M_VGAMUT_TO_XYZ, rgb);
    else if (g == 12) return odt11_vdot(ODT11_M_BMDWG_TO_XYZ, rgb);
    else if (g == 13) return odt11_vdot(ODT11_M_EGAMUT_TO_XYZ, rgb);
    else if (g == 14) return odt11_vdot(ODT11_M_EGAMUT2_TO_XYZ, rgb);
    else if (g == 15) return odt11_vdot(ODT11_M_DAVINCIWG_TO_XYZ, rgb);
    return rgb; // 0: XYZ identity
}

// ---- Display gamut and creative whitepoint (DCTL display_gamut_whitepoint) --
LDT_FUNC float3 odt11_display_gamut_whitepoint(float3 rgb, float tsn, float cwp_lm, int display_gamut, int cwp) {
    // P3D65 to XYZ D65
    rgb = odt11_vdot(ODT11_M_P3D65_TO_XYZ, rgb);

    float3 cwp_neutral = rgb;

    float cwp_f = _powf(tsn, 2.0f*cwp_lm);

    if (display_gamut < 3) { // D65 aligned P3 or Rec.709
        if (cwp == 0) rgb = odt11_vdot(ODT11_M_CAT_D65_TO_D93, rgb);
        else if (cwp == 1) rgb = odt11_vdot(ODT11_M_CAT_D65_TO_D75, rgb);
        else if (cwp == 3) rgb = odt11_vdot(ODT11_M_CAT_D65_TO_D60, rgb);
        else if (cwp == 4) rgb = odt11_vdot(ODT11_M_CAT_D65_TO_D55, rgb);
        else if (cwp == 5) rgb = odt11_vdot(ODT11_M_CAT_D65_TO_D50, rgb);
    }
    else if (display_gamut == 3) { // P3-D60
        if (cwp == 0) rgb = odt11_vdot(ODT11_M_CAT_D60_TO_D93, rgb);
        else if (cwp == 1) rgb = odt11_vdot(ODT11_M_CAT_D60_TO_D75, rgb);
        else if (cwp == 2) rgb = odt11_vdot(ODT11_M_CAT_D60_TO_D65, rgb);
        else if (cwp == 4) rgb = odt11_vdot(ODT11_M_CAT_D60_TO_D55, rgb);
        else if (cwp == 5) rgb = odt11_vdot(ODT11_M_CAT_D60_TO_D50, rgb);
    }
    else { // DCI P3 or DCI XYZ. Upstream applies the DCI CATs to D65 XYZ; kept as is.
        cwp_neutral = odt11_vdot(ODT11_M_CAT_DCI_TO_D65, rgb);
        if (cwp == 0) rgb = odt11_vdot(ODT11_M_CAT_DCI_TO_D93, rgb);
        else if (cwp == 1) rgb = odt11_vdot(ODT11_M_CAT_DCI_TO_D75, rgb);
        else if (cwp == 2) rgb = cwp_neutral;
        else if (cwp == 3) rgb = odt11_vdot(ODT11_M_CAT_DCI_TO_D60, rgb);
        else if (cwp == 4) rgb = odt11_vdot(ODT11_M_CAT_DCI_TO_D55, rgb);
        else if (cwp == 5) rgb = odt11_vdot(ODT11_M_CAT_DCI_TO_D50, rgb);
    }

    // rgb*cwp_f + cwp_neutral*(1 - cwp_f)
    {
        const float omf = 1.0f - cwp_f;
        rgb = make_float3(rgb.x*cwp_f + cwp_neutral.x*omf,
                          rgb.y*cwp_f + cwp_neutral.y*omf,
                          rgb.z*cwp_f + cwp_neutral.z*omf);
    }

    if (display_gamut == 0) {
        rgb = odt11_vdot(ODT11_M_XYZ_TO_REC709, rgb);
    }
    else if (display_gamut == 5) {
        rgb = odt11_vdot(ODT11_M_CAT_D65_TO_DCI, rgb);
    }
    else { // P3-D65, Rec.2020 (P3 limited), P3-D60 and P3-DCI all end in P3D65 RGB
        rgb = odt11_vdot(ODT11_M_XYZ_TO_P3D65, rgb);
    }

    float cwp_norm = 1.0f;
    if (display_gamut == 0) {
        if (cwp == 0) cwp_norm = 0.744192699063f;
        else if (cwp == 1) cwp_norm = 0.873470832146f;
        else if (cwp == 3) cwp_norm = 0.955936992163f;
        else if (cwp == 4) cwp_norm = 0.905671332781f;
        else if (cwp == 5) cwp_norm = 0.850004385027f;
    }
    else if (display_gamut == 1 || display_gamut == 2) {
        if (cwp == 0) cwp_norm = 0.762687057298f;
        else if (cwp == 1) cwp_norm = 0.884054083328f;
        else if (cwp == 3) cwp_norm = 0.964320186739f;
        else if (cwp == 4) cwp_norm = 0.923076518860f;
        else if (cwp == 5) cwp_norm = 0.876572837784f;
    }
    else if (display_gamut == 3) {
        if (cwp == 0) cwp_norm = 0.704956321013f;
        else if (cwp == 1) cwp_norm = 0.816715709816f;
        else if (cwp == 2) cwp_norm = 0.923382193663f;
        else if (cwp == 4) cwp_norm = 0.956138500287f;
        else if (cwp == 5) cwp_norm = 0.906801453023f;
    }
    else if (display_gamut == 4) {
        if (cwp == 0) cwp_norm = 0.665336141225f;
        else if (cwp == 1) cwp_norm = 0.770397131382f;
        else if (cwp == 2) cwp_norm = 0.870572343302f;
        else if (cwp == 3) cwp_norm = 0.891354547503f;
        else if (cwp == 4) cwp_norm = 0.855327825187f;
        else if (cwp == 5) cwp_norm = 0.814566436117f;
    }
    else if (display_gamut == 5) {
        if (cwp == 0) cwp_norm = 0.707142784007f;
        else if (cwp == 1) cwp_norm = 0.815561082617f;
        else if (cwp >= 2) cwp_norm = 0.916555279740f;
    }

    // rgb *= cwp_norm*cwp_f + 1.0f - cwp_f
    {
        const float nf = cwp_norm*cwp_f + 1.0f - cwp_f;
        rgb = rgb*nf;
    }
    return rgb;
}

// ---- Per-pixel entry point (DCTL transform) --------------------------------
// in: RGB in the selected input gamut and transfer function.
// x, y: pixel position (DCTL convention, y = 0 at the top), w, h: frame size.
// Only the tonescale overlay (crv_enable) uses x, y, w, h.
// p must have been filled and then passed through opendrt_v11_precompute.
LDT_FUNC float3 opendrt_v11_processPixel(float3 in, int x, int y, int w, int h, const OpenDRTv11Params* p)
{
    float3 rgb = in;

    const float pos_x = (float)x;
    const float pos_y = (float)y;
    const float res_x = (float)w;
    const float res_y = (float)h;

    float crv_tsn = 0.0f;
    if (p->crv_enable == 1) crv_tsn = odt11_oetf_filmlight_tlog(pos_x/res_x);

    rgb = odt11_linearize(rgb, p->in_oetf);

    // Tonescale constants (precomputed host side in the DCTL order)
    const float ts_x0 = p->ts_x0;
    const float ts_p = p->ts_p;
    const float ts_m2 = p->ts_m2;
    const float ts_s = p->ts_s;
    const float ts_dsc = p->ts_dsc;
    const float s_Lp100 = p->s_Lp100;
    const float ts_s1 = p->ts_s1;

    // Input gamut to P3-D65
    rgb = odt11_in_to_xyz(rgb, p->in_gamut);
    rgb = odt11_vdot(ODT11_M_XYZ_TO_P3D65, rgb);

    // Rendering space
    const float rs_sa = p->rs_sa;
    const float3 rs_w = make_float3(p->rs_rw, 1.0f - p->rs_rw - p->rs_bw, p->rs_bw);
    float sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;
    {
        const float oms = 1.0f - rs_sa;
        rgb = make_float3(sat_L*rs_sa + rgb.x*oms,
                          sat_L*rs_sa + rgb.y*oms,
                          sat_L*rs_sa + rgb.z*oms);
    }

    // Offset
    rgb = make_float3(rgb.x + p->tn_off, rgb.y + p->tn_off, rgb.z + p->tn_off);
    if (p->crv_enable == 1) crv_tsn += p->tn_off;

    // Tonescale norm
    float tsn = odt11_hypotf3(rgb)/ODT11_SQRT3;

    // RGB ratios
    rgb = odt11_sdivf3f(rgb, tsn);

    // Opponent space (DCTL opponent(), float2 split into two floats)
    const float opp_x = rgb.x - rgb.z;
    const float opp_y = rgb.y - (rgb.x + rgb.z)/2.0f;
    float ach_d = odt11_hypotf2(opp_x, opp_y)/2.0f;

    ach_d = (1.25f)*odt11_compress_toe_quadratic(ach_d, 0.25f, 0);

    // Hue angle, rotated so that red = 0.0
    const float hue = _fmod(_atan2f(opp_x, opp_y) + ODT11_PI + 1.10714931f, 2.0f*ODT11_PI);

    const float3 ha_rgb = make_float3(
        odt11_gauss_window(odt11_hue_offset(hue, 0.1f), 0.66f),
        odt11_gauss_window(odt11_hue_offset(hue, 4.3f), 0.66f),
        odt11_gauss_window(odt11_hue_offset(hue, 2.3f), 0.66f));

    const float3 ha_rgb_hs = make_float3(
        odt11_gauss_window(odt11_hue_offset(hue, -0.4f), 0.66f),
        ha_rgb.y,
        odt11_gauss_window(odt11_hue_offset(hue, 2.5f), 0.66f));

    const float3 ha_cmy = make_float3(
        odt11_gauss_window(odt11_hue_offset(hue, 3.3f), 0.5f),
        odt11_gauss_window(odt11_hue_offset(hue, 1.3f), 0.5f),
        odt11_gauss_window(odt11_hue_offset(hue, -1.15f), 0.5f));

    // Brilliance
    if (p->brl_enable) {
        float brl_tsf = _powf(tsn/(tsn + 1.0f), 1.0f - p->brl_rng);
        float brl_exf = (p->brl + p->brl_r*ha_rgb.x + p->brl_g*ha_rgb.y + p->brl_b*ha_rgb.z)*_powf(ach_d, 1.0f/p->brl_st);
        float brl_ex = _powf(2.0f, brl_exf*(brl_exf < 0.0f ? brl_tsf : 1.0f - brl_tsf));
        tsn *= brl_ex;
    }

    // Contrast low
    if (p->tn_lcon_enable) {
        float lcon_m = _powf(2.0f, -p->tn_lcon);
        float lcon_w = p->tn_lcon_w/4.0f;
        lcon_w *= lcon_w;
        const float lcon_cnst_sc = odt11_compress_toe_cubic(ts_x0, lcon_m, lcon_w, 1)/ts_x0;
        tsn *= lcon_cnst_sc;
        tsn = odt11_compress_toe_cubic(tsn, lcon_m, lcon_w, 0);
        if (p->crv_enable == 1) crv_tsn = odt11_compress_toe_cubic(crv_tsn*lcon_cnst_sc, lcon_m, lcon_w, 0);
    }

    // Contrast high
    if (p->tn_hcon_enable) {
        float hcon_p = _powf(2.0f, p->tn_hcon);
        tsn = odt11_contrast_high(tsn, hcon_p, p->tn_hcon_pv, p->tn_hcon_st, 0);
        if (p->crv_enable == 1) crv_tsn = odt11_contrast_high(crv_tsn, hcon_p, p->tn_hcon_pv, p->tn_hcon_st, 0);
    }

    // Hyperbolic compression
    const float tsn_pt = odt11_compress_hyperbolic_power(tsn, ts_s1, ts_p);
    const float tsn_const = odt11_compress_hyperbolic_power(tsn, s_Lp100, ts_p);
    tsn = odt11_compress_hyperbolic_power(tsn, ts_s, ts_p);

    float crv_tsn_const = 0.0f;
    if (p->crv_enable == 1) {
        crv_tsn_const = odt11_compress_hyperbolic_power(crv_tsn, s_Lp100, ts_p);
        crv_tsn = odt11_compress_hyperbolic_power(crv_tsn, ts_s, ts_p);
    }

    // Hue contrast R
    if (p->hc_enable) {
        float hc_ts = 1.0f - tsn_const;
        float hc_c = hc_ts*(1.0f - ach_d) + ach_d*(1.0f - hc_ts);
        hc_c *= ach_d*ha_rgb.x;
        hc_ts = _powf(hc_ts, 1.0f/p->hc_r_rng);
        float hc_f = p->hc_r*(hc_c - 2.0f*hc_c*hc_ts) + 1.0f;
        rgb = make_float3(rgb.x, rgb.y*hc_f, rgb.z*hc_f);
    }

    // Hue shift RGB
    if (p->hs_rgb_enable) {
        const float3 hs_rgb = make_float3(
            ha_rgb_hs.x*ach_d*_powf(tsn_pt, 1.0f/p->hs_r_rng),
            ha_rgb_hs.y*ach_d*_powf(tsn_pt, 1.0f/p->hs_g_rng),
            ha_rgb_hs.z*ach_d*_powf(tsn_pt, 1.0f/p->hs_b_rng));
        float3 hsf = make_float3(hs_rgb.x*p->hs_r, hs_rgb.y*-p->hs_g, hs_rgb.z*-p->hs_b);
        hsf = make_float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
        rgb = rgb + hsf;
    }

    // Hue shift CMY
    if (p->hs_cmy_enable) {
        const float tsn_pt_compl = 1.0f - tsn_pt;
        const float3 hs_cmy = make_float3(
            ha_cmy.x*ach_d*_powf(tsn_pt_compl, 1.0f/p->hs_c_rng),
            ha_cmy.y*ach_d*_powf(tsn_pt_compl, 1.0f/p->hs_m_rng),
            ha_cmy.z*ach_d*_powf(tsn_pt_compl, 1.0f/p->hs_y_rng));
        float3 hsf = make_float3(hs_cmy.x*-p->hs_c, hs_cmy.y*p->hs_m, hs_cmy.z*p->hs_y);
        hsf = make_float3(hsf.z - hsf.y, hsf.x - hsf.z, hsf.y - hsf.x);
        rgb = rgb + hsf;
    }

    // Purity limit low (not gated: the DCTL never reads pt_enable)
    const float pt_lml_p = 1.0f + 4.0f*(1.0f - tsn_pt)*(p->pt_lml + p->pt_lml_r*ha_rgb_hs.x + p->pt_lml_g*ha_rgb_hs.y + p->pt_lml_b*ha_rgb_hs.z);
    float ptf = 1.0f - _powf(tsn_pt, pt_lml_p);

    // Purity limit high
    const float pt_lmh_p = (1.0f - ach_d*(p->pt_lmh_r*ha_rgb_hs.x + p->pt_lmh_b*ha_rgb_hs.z))*(1.0f - p->pt_lmh*ach_d);
    ptf = _powf(ptf, pt_lmh_p);

    // Mid-range purity
    if (p->ptm_enable) {
        float ptm_low_f;
        if (p->ptm_low_st == 0.0f || p->ptm_low_rng == 0.0f) ptm_low_f = 1.0f;
        else ptm_low_f = 1.0f + p->ptm_low*_expf(-2.0f*ach_d*ach_d/p->ptm_low_st)*_powf(1.0f - tsn_const, 1.0f/p->ptm_low_rng);

        float ptm_high_f;
        if (p->ptm_high_st == 0.0f || p->ptm_high_rng == 0.0f) ptm_high_f = 1.0f;
        else ptm_high_f = 1.0f + p->ptm_high*_expf(-2.0f*ach_d*ach_d/p->ptm_high_st)*_powf(tsn_pt, 1.0f/(4.0f*p->ptm_high_rng));

        ptf *= ptm_low_f*ptm_high_f;
    }

    // Lerp to peak achromatic by ptf in rgb ratios: rgb*ptf + 1.0f - ptf
    rgb = make_float3(rgb.x*ptf + 1.0f - ptf, rgb.y*ptf + 1.0f - ptf, rgb.z*ptf + 1.0f - ptf);

    // Inverse rendering space
    sat_L = rgb.x*rs_w.x + rgb.y*rs_w.y + rgb.z*rs_w.z;
    {
        const float dn = rs_sa - 1.0f;
        rgb = make_float3((sat_L*rs_sa - rgb.x)/dn,
                          (sat_L*rs_sa - rgb.y)/dn,
                          (sat_L*rs_sa - rgb.z)/dn);
    }

    // Display gamut and creative white
    rgb = odt11_display_gamut_whitepoint(rgb, tsn_const, p->cwp_lm, p->display_gamut, p->cwp);

    // Post brilliance
    if (p->brlp_enable) {
        const float bo_x = rgb.x - rgb.z;
        const float bo_y = rgb.y - (rgb.x + rgb.z)/2.0f;
        float brlp_ach_d = odt11_hypotf2(bo_x, bo_y)/4.0f;
        brlp_ach_d = 1.1f*(brlp_ach_d*brlp_ach_d/(brlp_ach_d + 0.1f));
        const float3 brlp_ha_rgb = ach_d*ha_rgb;
        const float brlp_m = p->brlp + p->brlp_r*brlp_ha_rgb.x + p->brlp_g*brlp_ha_rgb.y + p->brlp_b*brlp_ha_rgb.z;
        const float brlp_ex = _powf(2.0f, brlp_m*brlp_ach_d*tsn);
        rgb = rgb*brlp_ex;
    }

    // Purity compress low
    if (p->ptl_enable) rgb = make_float3(odt11_softplus(rgb.x, p->ptl_c), odt11_softplus(rgb.y, p->ptl_m), odt11_softplus(rgb.z, p->ptl_y));

    // Final tonescale adjustments
    tsn *= ts_m2;
    tsn = odt11_compress_toe_quadratic(tsn, p->tn_toe, 0);
    tsn *= ts_dsc;

    if (p->crv_enable == 1) {
        crv_tsn *= ts_m2;
        crv_tsn = odt11_compress_toe_quadratic(crv_tsn, p->tn_toe, 0);
        crv_tsn *= ts_dsc;
        if (p->eotf == 4) crv_tsn *= 10.0f;
    }

    float3 crv_rgb = make_float3(crv_tsn, crv_tsn, crv_tsn);
    if (p->crv_enable == 1) crv_rgb = odt11_display_gamut_whitepoint(crv_rgb, crv_tsn_const, p->cwp_lm, p->display_gamut, p->cwp);

    // Return from RGB ratios
    rgb = rgb*tsn;

    // Rec.2020 (P3 limited)
    if (p->display_gamut == 2) {
        rgb = odt11_clampminf3(rgb, 0.0f);
        rgb = odt11_vdot(ODT11_M_P3_TO_REC2020, rgb);
    }

    if (p->clamp) rgb = odt11_clampf3(rgb, 0.0f, 1.0f);

    // Inverse display EOTF
    const int eotf = p->eotf;
    const float eotf_p = 2.0f + eotf*0.2f;
    if ((eotf > 0) && (eotf < 4)) rgb = odt11_spowf3(rgb, 1.0f/eotf_p);
    else if (eotf == 4) rgb = odt11_eotf_pq_inv(rgb);
    else if (eotf == 5) rgb = odt11_eotf_hlg_inv(rgb);

    if (p->crv_enable == 1) {
        if ((eotf > 0) && (eotf < 4)) crv_rgb = odt11_spowf3(crv_rgb, 1.0f/eotf_p);
        else if (eotf == 4) crv_rgb = odt11_eotf_pq_inv(crv_rgb);
        else if (eotf == 5) crv_rgb = odt11_eotf_hlg_inv(crv_rgb);
    }

    // Tonescale overlay
    if (p->crv_enable == 1) {
        float3 crv_rgb_dst = make_float3(res_y - pos_y - crv_rgb.x*res_y,
                                         res_y - pos_y - crv_rgb.y*res_y,
                                         res_y - pos_y - crv_rgb.z*res_y);
        const float crv_w0 = 0.35f;
        crv_rgb_dst.x = _expf(-crv_rgb_dst.x*crv_rgb_dst.x*crv_w0);
        crv_rgb_dst.y = _expf(-crv_rgb_dst.y*crv_rgb_dst.y*crv_w0);
        crv_rgb_dst.z = _expf(-crv_rgb_dst.z*crv_rgb_dst.z*crv_w0);
        const float crv_lm = 1.0f; // DCTL: eotf < 4 ? 1.0f : 1.0f
        crv_rgb_dst = odt11_clampf3(crv_rgb_dst, 0.0f, 1.0f);
        rgb = make_float3(rgb.x*(1.0f - crv_rgb_dst.x) + crv_lm*crv_rgb_dst.x,
                          rgb.y*(1.0f - crv_rgb_dst.y) + crv_lm*crv_rgb_dst.y,
                          rgb.z*(1.0f - crv_rgb_dst.z) + crv_lm*crv_rgb_dst.z);
    }

    return rgb;
}
