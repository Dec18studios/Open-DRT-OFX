// LdtFactorySettings.cpp - see LdtFactorySettings.h.
//
// Layer A is plain C++17 + std::filesystem. Layer B (below the
// LDT_FACTORY_NO_OFX guard) is the OFX glue. File IO never goes through
// LdtControl.cpp (stubbed on Windows).

#include "LdtFactorySettings.h"
#include "ldt_json.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <mutex>
#include <sstream>
#include <system_error>

#ifdef _WIN32
#include <process.h>
#else
#include <pwd.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace ldt {
namespace factory {

// ── small helpers ─────────────────────────────────────────────────────────────
namespace {

std::string lower(const std::string& s)
{
    std::string out(s);
    for (char& c : out) c = (char)std::tolower((unsigned char)c);
    return out;
}

[[maybe_unused]] bool startsWith(const std::string& s, const std::string& prefix)
{
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

std::string formatDouble(double v)
{
    if (!std::isfinite(v)) v = 0.0;
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

bool readTextFile(const fs::path& p, std::string& out)
{
    std::ifstream in(p, std::ios::in | std::ios::binary);
    if (!in) return false;
    std::ostringstream ss;
    ss << in.rdbuf();
    out = ss.str();
    return !in.bad();
}

std::string fileStem(const std::string& name)
{
    std::string stem;
    for (char c : name) {
        unsigned char u = (unsigned char)c;
        if (std::isalnum(u))           stem += (char)std::tolower(u);
        else if (c == '-' || c == '_') stem += c;
        else if (c == ' ' || c == '.') { if (!stem.empty() && stem.back() != '-') stem += '-'; }
    }
    while (!stem.empty() && stem.back() == '-') stem.pop_back();
    if (stem.size() > 48) stem.resize(48);
    if (stem.empty() || stem == "default") stem = "starting-point";
    return stem;
}

long currentPid()
{
#ifdef _WIN32
    return (long)_getpid();
#else
    return (long)getpid();
#endif
}

struct Cache {
    std::mutex mutex;
    std::map<std::string, std::vector<Slot>> byDir;
};
Cache& cache() { static Cache c; return c; }

} // namespace

// ── directories ───────────────────────────────────────────────────────────────
fs::path baseDirectory()
{
#if defined(_WIN32)
    const wchar_t* appData = _wgetenv(L"APPDATA");
    fs::path root = (appData && *appData) ? fs::path(appData) : fs::temp_directory_path();
    return root / "LookDevTools" / "starting-points";
#else
    std::string home;
    if (const char* h = std::getenv("HOME")) home = h;
    if (home.empty()) {
        if (const passwd* pw = getpwuid(getuid())) if (pw->pw_dir) home = pw->pw_dir;
    }
    if (home.empty()) home = fs::temp_directory_path().string();
#if defined(__APPLE__)
    return fs::path(home) / "Library" / "Application Support" / "LookDevTools" / "starting-points";
#else
    fs::path cfg;
    if (const char* x = std::getenv("XDG_CONFIG_HOME")) if (*x) cfg = x;
    if (cfg.empty()) cfg = fs::path(home) / ".config";
    return cfg / "LookDevTools" / "starting-points";
#endif
#endif
}

fs::path storeDirectory(const std::string& storeKey)
{
    // Store keys are fixed ASCII identifiers chosen in code; sanitize anyway so
    // a key can never climb out of the base directory.
    std::string key;
    for (char c : storeKey) {
        unsigned char u = (unsigned char)c;
        if (std::isalnum(u) || c == '-' || c == '_') key += c;
    }
    if (key.empty()) key = "default-store";
    return baseDirectory() / fs::u8path(key);
}

// ── names ─────────────────────────────────────────────────────────────────────
std::string sanitizeName(const std::string& raw)
{
    std::string out;
    bool pendingSpace = false;
    for (char c : raw) {
        unsigned char u = (unsigned char)c;
        if (u < 0x20 || u == 0x7f) { pendingSpace = true; continue; }
        if (c == ' ' || c == '\t') { pendingSpace = true; continue; }
        if (std::string("/\\:*?\"<>|").find(c) != std::string::npos) continue;
        if (pendingSpace && !out.empty()) out += ' ';
        pendingSpace = false;
        out += c;
    }
    if (out.size() > 64) {
        out.resize(64);
        // Never leave half a UTF-8 sequence behind.
        while (!out.empty() && (((unsigned char)out.back()) & 0xC0) == 0x80) out.pop_back();
        if (!out.empty() && ((unsigned char)out.back()) >= 0xC0) out.pop_back();
        while (!out.empty() && out.back() == ' ') out.pop_back();
    }
    return out;
}

bool sameName(const std::string& a, const std::string& b) { return lower(a) == lower(b); }

bool isReservedName(const std::string& name) { return sameName(name, kFactoryLabel); }

// ── serialize / parse ─────────────────────────────────────────────────────────
std::string serialize(const Slot& s)
{
    using ldtjson::escape;
    std::ostringstream o;
    o << "{\n";
    o << "  \"format\": \"" << kFormatName << "\",\n";
    o << "  \"kind\": \"startingPoint\",\n";
    o << "  \"schema\": " << kSchemaMajor << ",\n";
    o << "  \"pluginId\": \"" << escape(s.pluginId) << "\",\n";
    o << "  \"pluginVersion\": \"" << escape(s.pluginVersion) << "\",\n";
    o << "  \"name\": \"" << escape(s.name) << "\",\n";
    o << "  \"savedAt\": \"" << escape(s.savedAt) << "\",\n";

    o << "  \"values\": {";
    bool first = true;
    for (const auto& kv : s.values) {
        o << (first ? "\n" : ",\n") << "    \"" << escape(kv.first) << "\": ";
        first = false;
        if (kv.second.size() == 1) {
            o << formatDouble(kv.second[0]);
        } else {
            o << "[";
            for (size_t i = 0; i < kv.second.size(); ++i)
                o << (i ? ", " : "") << formatDouble(kv.second[i]);
            o << "]";
        }
    }
    o << (first ? "},\n" : "\n  },\n");

    o << "  \"choices\": {";
    first = true;
    for (const auto& kv : s.choices) {
        o << (first ? "\n" : ",\n") << "    \"" << escape(kv.first) << "\": { \"label\": \""
          << escape(kv.second.label) << "\", \"index\": " << kv.second.index << " }";
        first = false;
    }
    o << (first ? "},\n" : "\n  },\n");

    o << "  \"strings\": {";
    first = true;
    for (const auto& kv : s.strings) {
        o << (first ? "\n" : ",\n") << "    \"" << escape(kv.first) << "\": \""
          << escape(kv.second) << "\"";
        first = false;
    }
    o << (first ? "}\n" : "\n  }\n");
    o << "}\n";
    return o.str();
}

bool parse(const std::string& text, Slot& out, std::string& err)
{
    ldtjson::Value root;
    if (!ldtjson::parse(text, root, err)) return false;
    if (!root.isObject()) { err = "not an object"; return false; }

    const ldtjson::Value* fmt = root.find("format");
    if (!fmt || !fmt->isString() || fmt->str != kFormatName) { err = "wrong format"; return false; }
    const ldtjson::Value* kind = root.find("kind");
    if (!kind || !kind->isString() || kind->str != "startingPoint") { err = "wrong kind"; return false; }
    const ldtjson::Value* schema = root.find("schema");
    double schemaNum = 0.0;
    if (!schema || !schema->asDouble(schemaNum) || (int)std::floor(schemaNum) != kSchemaMajor) {
        err = "unsupported schema";
        return false;
    }
    const ldtjson::Value* name = root.find("name");
    if (!name || !name->isString() || sanitizeName(name->str).empty()) { err = "missing name"; return false; }

    Slot s;
    s.name = sanitizeName(name->str);
    if (const ldtjson::Value* v = root.find("pluginId"))      if (v->isString()) s.pluginId = v->str;
    if (const ldtjson::Value* v = root.find("pluginVersion")) if (v->isString()) s.pluginVersion = v->str;
    if (const ldtjson::Value* v = root.find("savedAt"))       if (v->isString()) s.savedAt = v->str;

    if (const ldtjson::Value* vals = root.find("values")) {
        if (vals->isObject()) {
            for (const auto& kv : vals->obj) {
                std::vector<double> nums;
                if (kv.second.isNumber()) nums.push_back(kv.second.number);
                else if (kv.second.isBool()) nums.push_back(kv.second.boolean ? 1.0 : 0.0);
                else if (kv.second.isArray()) {
                    bool ok = true;
                    for (const auto& e : kv.second.arr) {
                        if (!e.isNumber()) { ok = false; break; }
                        nums.push_back(e.number);
                    }
                    if (!ok) continue;
                }
                if (!nums.empty()) s.values[kv.first] = nums;
            }
        }
    }
    if (const ldtjson::Value* ch = root.find("choices")) {
        if (ch->isObject()) {
            for (const auto& kv : ch->obj) {
                ChoiceValue cv;
                if (kv.second.isString()) {
                    cv.label = kv.second.str;
                } else if (kv.second.isNumber()) {
                    cv.index = (int)std::llround(kv.second.number);
                } else if (kv.second.isObject()) {
                    if (const ldtjson::Value* l = kv.second.find("label")) if (l->isString()) cv.label = l->str;
                    long long ix = -1;
                    if (const ldtjson::Value* i = kv.second.find("index")) if (i->asInt(ix)) cv.index = (int)ix;
                } else {
                    continue;
                }
                s.choices[kv.first] = cv;
            }
        }
    }
    if (const ldtjson::Value* st = root.find("strings")) {
        if (st->isObject()) {
            for (const auto& kv : st->obj)
                if (kv.second.isString()) s.strings[kv.first] = kv.second.str;
        }
    }
    out = s;
    return true;
}

// ── list / find ───────────────────────────────────────────────────────────────
static std::vector<Slot> scanDirectory(const fs::path& dir)
{
    std::vector<Slot> slots;
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) return slots;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const fs::path p = it->path();
        std::error_code fec;
        if (!it->is_regular_file(fec)) continue;
        if (p.extension() != ".json") continue;
        const std::string fname = p.filename().u8string();
        if (fname.empty() || fname[0] == '.' || fname == "default.json") continue;
        std::string text, err;
        if (!readTextFile(p, text)) continue;
        Slot s;
        if (!parse(text, s, err)) continue;
        s.path = p;
        bool dup = false;
        for (const Slot& o : slots) if (sameName(o.name, s.name)) { dup = true; break; }
        if (!dup) slots.push_back(s);
    }
    std::sort(slots.begin(), slots.end(), [](const Slot& a, const Slot& b) {
        const std::string la = lower(a.name), lb = lower(b.name);
        return la != lb ? la < lb : a.name < b.name;
    });
    return slots;
}

std::vector<Slot> listSlots(const fs::path& dir, bool refresh)
{
    Cache& c = cache();
    const std::string key = dir.u8string();
    {
        std::lock_guard<std::mutex> lock(c.mutex);
        if (!refresh) {
            auto it = c.byDir.find(key);
            if (it != c.byDir.end()) return it->second;
        }
    }
    std::vector<Slot> slots = scanDirectory(dir);
    std::lock_guard<std::mutex> lock(c.mutex);
    c.byDir[key] = slots;
    return slots;
}

void invalidate(const fs::path& dir)
{
    Cache& c = cache();
    std::lock_guard<std::mutex> lock(c.mutex);
    c.byDir.erase(dir.u8string());
}

bool findSlot(const fs::path& dir, const std::string& name, Slot& out, bool refresh)
{
    const std::string clean = sanitizeName(name);
    if (clean.empty()) return false;
    for (const Slot& s : listSlots(dir, refresh)) {
        if (sameName(s.name, clean)) { out = s; return true; }
    }
    return false;
}

fs::path uniquePath(const fs::path& dir, const std::string& name)
{
    const std::string stem = fileStem(name);
    std::error_code ec;
    fs::path p = dir / fs::u8path(stem + ".json");
    for (int n = 2; fs::exists(p, ec) && n < 10000; ++n)
        p = dir / fs::u8path(stem + "-" + std::to_string(n) + ".json");
    return p;
}

// ── atomic write ──────────────────────────────────────────────────────────────
bool writeAtomic(const fs::path& path, const std::string& text, std::string& err)
{
    static std::atomic<unsigned long> s_counter(0);
    std::error_code ec;
    const fs::path dir = path.parent_path();
    if (!dir.empty()) {
        fs::create_directories(dir, ec);
        if (ec) { err = "could not create " + dir.u8string() + ": " + ec.message(); return false; }
    }

    // Unique temp name in the same directory, so the rename never crosses a
    // volume and two writers never share a temp file.
    const unsigned long long ticks =
        (unsigned long long)std::chrono::steady_clock::now().time_since_epoch().count();
    std::string tmpName = "." + path.filename().u8string() + "." + std::to_string(currentPid()) + "." +
                          std::to_string(++s_counter) + "." + std::to_string(ticks % 1000000007ULL) + ".tmp";
    const fs::path tmp = dir / fs::u8path(tmpName);

    {
        std::ofstream out(tmp, std::ios::out | std::ios::binary | std::ios::trunc);
        if (!out) { err = "could not open " + tmp.u8string(); return false; }
        out.write(text.data(), (std::streamsize)text.size());
        out.flush();
        if (!out) {
            out.close();
            fs::remove(tmp, ec);
            err = "could not write " + tmp.u8string();
            return false;
        }
    }

    // std::filesystem::rename replaces an existing target (rename(2) on POSIX,
    // MoveFileExW with MOVEFILE_REPLACE_EXISTING on MSVC).
    fs::rename(tmp, path, ec);
    if (ec) {
        std::error_code ec2;
        fs::remove(tmp, ec2); // only ever our own temp file
        err = "could not replace " + path.u8string() + ": " + ec.message();
        return false;
    }
    return true;
}

bool writeSlot(const Slot& slot, std::string& err)
{
    if (slot.path.empty()) { err = "no path"; return false; }
    if (!writeAtomic(slot.path, serialize(slot), err)) return false;
    invalidate(slot.path.parent_path());
    return true;
}

// ── default pointer ───────────────────────────────────────────────────────────
std::string readDefaultName(const fs::path& dir)
{
    std::string text, err;
    if (!readTextFile(dir / "default.json", text)) return std::string();
    ldtjson::Value root;
    if (!ldtjson::parse(text, root, err) || !root.isObject()) return std::string();
    const ldtjson::Value* fmt = root.find("format");
    const ldtjson::Value* kind = root.find("kind");
    const ldtjson::Value* schema = root.find("schema");
    double sn = 0.0;
    if (!fmt || !fmt->isString() || fmt->str != kFormatName) return std::string();
    if (!kind || !kind->isString() || kind->str != "defaultPointer") return std::string();
    if (!schema || !schema->asDouble(sn) || (int)std::floor(sn) != kSchemaMajor) return std::string();
    const ldtjson::Value* name = root.find("name");
    if (!name || !name->isString()) return std::string();
    const std::string clean = sanitizeName(name->str);
    return isReservedName(clean) ? std::string() : clean;
}

bool writeDefaultName(const fs::path& dir, const std::string& name, std::string& err)
{
    const std::string clean = isReservedName(name) ? std::string() : sanitizeName(name);
    std::ostringstream o;
    o << "{\n  \"format\": \"" << kFormatName << "\",\n  \"kind\": \"defaultPointer\",\n"
      << "  \"schema\": " << kSchemaMajor << ",\n  \"name\": \"" << ldtjson::escape(clean) << "\",\n"
      << "  \"savedAt\": \"" << nowIso8601() << "\"\n}\n";
    return writeAtomic(dir / "default.json", o.str(), err);
}

// ── backup rename ─────────────────────────────────────────────────────────────
bool renameToBackup(const fs::path& path, fs::path& backupOut, std::string& err)
{
    std::error_code ec;
    if (!fs::exists(path, ec)) { err = "not found: " + path.u8string(); return false; }
    fs::path bak = path;
    bak += ".bak";
    for (int n = 2; fs::exists(bak, ec) && n < 10000; ++n) {
        bak = path;
        bak += ".bak." + std::to_string(n);
    }
    fs::rename(path, bak, ec);
    if (ec) { err = "could not rename " + path.u8string() + ": " + ec.message(); return false; }
    invalidate(path.parent_path());
    backupOut = bak;
    return true;
}

std::string nowIso8601()
{
    std::time_t t = std::time(nullptr);
    std::tm tmv{};
#ifdef _WIN32
    gmtime_s(&tmv, &t);
#else
    gmtime_r(&t, &tmv);
#endif
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tmv);
    return buf;
}

// ── value resolution ──────────────────────────────────────────────────────────
int resolveChoiceIndex(const ChoiceValue* stored, const std::vector<std::string>& options, int defaultIndex)
{
    if (!stored) return defaultIndex;
    if (!stored->label.empty()) {
        for (size_t i = 0; i < options.size(); ++i)
            if (options[i] == stored->label) return (int)i;
    }
    if (stored->index >= 0 && stored->index < (int)options.size()) return stored->index;
    return defaultIndex;
}

std::vector<double> resolveNumeric(const std::vector<double>* stored, const std::vector<double>& def)
{
    std::vector<double> out(def);
    if (!stored) return out;
    for (size_t i = 0; i < out.size() && i < stored->size(); ++i)
        if (std::isfinite((*stored)[i])) out[i] = (*stored)[i];
    return out;
}

} // namespace factory
} // namespace ldt

// ══════════════════════════════════════════════════════════════════════════════
// Layer B: OFX
// ══════════════════════════════════════════════════════════════════════════════
#ifndef LDT_FACTORY_NO_OFX

#include "ofxsParam.h"

namespace ldt {
namespace factory {

namespace {

const char* const kUuidParam        = "ldt_instance_uuid";
const char* const kPrefix           = "factory_";
const char* const kGroup            = "factory_settings_group";
const char* const kCurrentDefault   = "factory_current_default";
const char* const kChoice           = "factory_starting_point";
const char* const kSelected         = "factory_starting_point_selected";
const char* const kName             = "factory_starting_point_name";
const char* const kSaveNew          = "factory_save_new";
const char* const kUpdateSelected   = "factory_update_selected";
const char* const kMakeDefault      = "factory_make_default";
const char* const kApplySelected    = "factory_apply_selected";
const char* const kResetToFactory   = "factory_reset_to_factory";
const char* const kDeleteSelected   = "factory_delete_selected";
const char* const kDefaultSuffix    = " (Default)";

// Reads the descriptor's own map of defined params (protected in the support
// lib). Pointer-to-member formed through a derived class, applied to the base.
struct DescPeek : OFX::ParamSetDescriptor {
    static const std::map<std::string, OFX::ParamDescriptor*>& defined(const OFX::ParamSetDescriptor& d)
    {
        return d.*(&DescPeek::_definedParams);
    }
};

struct RegEntry {
    std::string name;
    OFX::ParamTypeEnum type;
};

struct RegInfo {
    Options options;
    std::vector<RegEntry> params;
};

std::mutex& regMutex() { static std::mutex m; return m; }
std::map<std::string, RegInfo>& registry() { static std::map<std::string, RegInfo> r; return r; }

bool capturableType(OFX::ParamTypeEnum t)
{
    switch (t) {
        case OFX::eStringParam: case OFX::eIntParam: case OFX::eInt2DParam: case OFX::eInt3DParam:
        case OFX::eDoubleParam: case OFX::eDouble2DParam: case OFX::eDouble3DParam:
        case OFX::eRGBParam: case OFX::eRGBAParam: case OFX::eBooleanParam: case OFX::eChoiceParam:
            return true;
        default:
            return false; // dummy, custom, parametric, strchoice, group, page, push button
    }
}

bool alwaysExcluded(const std::string& id)
{
    return startsWith(id, kPrefix) || id == kInitFlagParam || id == kUuidParam;
}

// One value in a uniform shape: numbers for numeric/bool/choice-index params,
// str for strings.
struct Val {
    std::vector<double> num;
    std::string str;
};

bool numClose(double v, double d) { return std::fabs(v - d) <= 1e-6 + 1e-5 * std::fabs(d); }

bool valEqual(const RegEntry& e, const Val& a, const Val& b)
{
    if (e.type == OFX::eStringParam) return a.str == b.str;
    if (a.num.size() != b.num.size()) return false;
    for (size_t i = 0; i < a.num.size(); ++i) {
        if (e.type == OFX::eDoubleParam || e.type == OFX::eDouble2DParam || e.type == OFX::eDouble3DParam ||
            e.type == OFX::eRGBParam || e.type == OFX::eRGBAParam) {
            if (!numClose(a.num[i], b.num[i])) return false;
        } else if (std::llround(a.num[i]) != std::llround(b.num[i])) {
            return false;
        }
    }
    return true;
}

// Typed access by registry type. fetch*Param caches, so repeated calls are cheap.
bool readParam(OFX::ImageEffect& fx, const RegEntry& e, double t, bool wantDefault, Val& out)
{
    try {
        out = Val();
        switch (e.type) {
            case OFX::eIntParam: {
                OFX::IntParam* p = fx.fetchIntParam(e.name); int v = 0;
                if (wantDefault) p->getDefault(v); else p->getValueAtTime(t, v);
                out.num = { (double)v }; return true;
            }
            case OFX::eInt2DParam: {
                OFX::Int2DParam* p = fx.fetchInt2DParam(e.name); int x = 0, y = 0;
                if (wantDefault) p->getDefault(x, y); else p->getValueAtTime(t, x, y);
                out.num = { (double)x, (double)y }; return true;
            }
            case OFX::eInt3DParam: {
                OFX::Int3DParam* p = fx.fetchInt3DParam(e.name); int x = 0, y = 0, z = 0;
                if (wantDefault) p->getDefault(x, y, z); else p->getValueAtTime(t, x, y, z);
                out.num = { (double)x, (double)y, (double)z }; return true;
            }
            case OFX::eDoubleParam: {
                OFX::DoubleParam* p = fx.fetchDoubleParam(e.name); double v = 0;
                if (wantDefault) p->getDefault(v); else p->getValueAtTime(t, v);
                out.num = { v }; return true;
            }
            case OFX::eDouble2DParam: {
                OFX::Double2DParam* p = fx.fetchDouble2DParam(e.name); double x = 0, y = 0;
                if (wantDefault) p->getDefault(x, y); else p->getValueAtTime(t, x, y);
                out.num = { x, y }; return true;
            }
            case OFX::eDouble3DParam: {
                OFX::Double3DParam* p = fx.fetchDouble3DParam(e.name); double x = 0, y = 0, z = 0;
                if (wantDefault) p->getDefault(x, y, z); else p->getValueAtTime(t, x, y, z);
                out.num = { x, y, z }; return true;
            }
            case OFX::eRGBParam: {
                OFX::RGBParam* p = fx.fetchRGBParam(e.name); double r = 0, g = 0, b = 0;
                if (wantDefault) p->getDefault(r, g, b); else p->getValueAtTime(t, r, g, b);
                out.num = { r, g, b }; return true;
            }
            case OFX::eRGBAParam: {
                OFX::RGBAParam* p = fx.fetchRGBAParam(e.name); double r = 0, g = 0, b = 0, a = 0;
                if (wantDefault) p->getDefault(r, g, b, a); else p->getValueAtTime(t, r, g, b, a);
                out.num = { r, g, b, a }; return true;
            }
            case OFX::eBooleanParam: {
                OFX::BooleanParam* p = fx.fetchBooleanParam(e.name); bool v = false;
                if (wantDefault) p->getDefault(v); else p->getValueAtTime(t, v);
                out.num = { v ? 1.0 : 0.0 }; return true;
            }
            case OFX::eChoiceParam: {
                OFX::ChoiceParam* p = fx.fetchChoiceParam(e.name); int v = 0;
                if (wantDefault) p->getDefault(v); else p->getValueAtTime(t, v);
                out.num = { (double)v }; return true;
            }
            case OFX::eStringParam: {
                OFX::StringParam* p = fx.fetchStringParam(e.name); std::string v;
                if (wantDefault) p->getDefault(v); else p->getValueAtTime(t, v);
                out.str = v; return true;
            }
            default:
                return false;
        }
    } catch (...) {
        return false;
    }
}

int ri(double v) { return (int)std::llround(v); }

bool writeParam(OFX::ImageEffect& fx, const RegEntry& e, const Val& v)
{
    try {
        const std::vector<double>& n = v.num;
        switch (e.type) {
            case OFX::eIntParam:      fx.fetchIntParam(e.name)->setValue(ri(n[0])); return true;
            case OFX::eInt2DParam:    fx.fetchInt2DParam(e.name)->setValue(ri(n[0]), ri(n[1])); return true;
            case OFX::eInt3DParam:    fx.fetchInt3DParam(e.name)->setValue(ri(n[0]), ri(n[1]), ri(n[2])); return true;
            case OFX::eDoubleParam:   fx.fetchDoubleParam(e.name)->setValue(n[0]); return true;
            case OFX::eDouble2DParam: fx.fetchDouble2DParam(e.name)->setValue(n[0], n[1]); return true;
            case OFX::eDouble3DParam: fx.fetchDouble3DParam(e.name)->setValue(n[0], n[1], n[2]); return true;
            case OFX::eRGBParam:      fx.fetchRGBParam(e.name)->setValue(n[0], n[1], n[2]); return true;
            case OFX::eRGBAParam:     fx.fetchRGBAParam(e.name)->setValue(n[0], n[1], n[2], n[3]); return true;
            case OFX::eBooleanParam:  fx.fetchBooleanParam(e.name)->setValue(n[0] != 0.0); return true;
            case OFX::eChoiceParam:   fx.fetchChoiceParam(e.name)->setValue(ri(n[0])); return true;
            case OFX::eStringParam:   fx.fetchStringParam(e.name)->setValue(v.str); return true;
            default: return false;
        }
    } catch (...) {
        return false;
    }
}

OFX::ValueParam* valueParam(OFX::ImageEffect& fx, const RegEntry& e)
{
    try {
        switch (e.type) {
            case OFX::eIntParam:      return fx.fetchIntParam(e.name);
            case OFX::eInt2DParam:    return fx.fetchInt2DParam(e.name);
            case OFX::eInt3DParam:    return fx.fetchInt3DParam(e.name);
            case OFX::eDoubleParam:   return fx.fetchDoubleParam(e.name);
            case OFX::eDouble2DParam: return fx.fetchDouble2DParam(e.name);
            case OFX::eDouble3DParam: return fx.fetchDouble3DParam(e.name);
            case OFX::eRGBParam:      return fx.fetchRGBParam(e.name);
            case OFX::eRGBAParam:     return fx.fetchRGBAParam(e.name);
            case OFX::eBooleanParam:  return fx.fetchBooleanParam(e.name);
            case OFX::eChoiceParam:   return fx.fetchChoiceParam(e.name);
            case OFX::eStringParam:   return fx.fetchStringParam(e.name);
            default: return nullptr;
        }
    } catch (...) {
        return nullptr;
    }
}

} // namespace

// ── describe ──────────────────────────────────────────────────────────────────
void describe(OFX::ImageEffectDescriptor& desc, OFX::PageParamDescriptor* page, const Options& options)
{
    using namespace OFX;

    // Hidden persistent "this node already received its starting point" flag.
    BooleanParamDescriptor* flag = desc.defineBooleanParam(kInitFlagParam);
    flag->setLabels("Factory Initialized", "Factory Initialized", "Factory Initialized");
    flag->setHint("Internal Look Dev Tools flag. Not user-editable.");
    flag->setDefault(false);
    flag->setIsSecret(true);
    flag->setIsPersistant(true);
    flag->setAnimates(false);
    flag->setEvaluateOnChange(false);
    if (page) page->addChild(*flag);

    GroupParamDescriptor* group = desc.defineGroupParam(kGroup);
    group->setLabels("Factory Settings", "Factory Settings", "Factory Settings");
    group->setHint("Save your own starting points and choose which one new nodes open with. "
                   "View modes and diagnostics are never stored.");
    group->setOpen(false);
    if (page) page->addChild(*group);

    // Every factory_* control is UI only: non-persistent, never re-renders.
    auto uiString = [&](const char* id, const char* label, const char* hint, const char* def) {
        StringParamDescriptor* p = desc.defineStringParam(id);
        p->setLabels(label, label, label);
        p->setHint(hint);
        p->setStringType(eStringTypeSingleLine);
        p->setDefault(def);
        p->setIsPersistant(false);
        p->setAnimates(false);
        p->setEvaluateOnChange(false);
        p->setParent(*group);
        if (page) page->addChild(*p);
        return p;
    };
    auto button = [&](const char* id, const char* label, const char* hint) {
        PushButtonParamDescriptor* p = desc.definePushButtonParam(id);
        p->setLabels(label, label, label);
        p->setHint(hint);
        // Push buttons are never persistent and never trigger a render.
        p->setParent(*group);
        if (page) page->addChild(*p);
        return p;
    };

    StringParamDescriptor* current = uiString(kCurrentDefault, "Default for New Nodes",
        "The starting point new nodes open with on this machine.", kFactoryLabel);
    current->setEnabled(false);

    ChoiceParamDescriptor* choice = desc.defineChoiceParam(kChoice);
    choice->setLabels("Starting Point", "Starting Point", "Starting Point");
    choice->setHint("Pick a saved starting point to apply, update, delete or make the default. "
                    "Factory is the built-in setup.");
    choice->appendOption(kFactoryLabel);
    choice->setDefault(0);
    choice->setIsPersistant(false);
    choice->setAnimates(false);
    choice->setEvaluateOnChange(false);
    choice->setParent(*group);
    if (page) page->addChild(*choice);

    StringParamDescriptor* selected = uiString(kSelected, "Selected Starting Point",
        "Internal: the selected starting point, tracked by name.", "");
    selected->setIsSecret(true);

    uiString(kName, "Name", "Name used by Save Current as New Starting Point.", "My Starting Point");

    button(kSaveNew, "Save Current as New Starting Point",
           "Saves this node's current settings as a starting point under the name above. "
           "An existing starting point with the same name is replaced.");
    PushButtonParamDescriptor* update = button(kUpdateSelected, "Update Selected",
           "Overwrites the selected starting point with this node's current settings.");
    update->setEnabled(false);
    button(kMakeDefault, "Make Selected the Default for New Nodes",
           "New nodes open with the selected starting point. Select Factory to go back to the "
           "built-in setup. Nodes already in a project are never changed.");
    button(kApplySelected, "Apply Selected to This Node",
           "Sets this node to the selected starting point. Controls it does not store go back to Factory.");
    button(kResetToFactory, "Reset This Node to Factory",
           "Sets every stored control on this node back to the built-in setup.");
    PushButtonParamDescriptor* del = button(kDeleteSelected, "Delete Selected",
           "Deletes the selected starting point. The file is kept as a .bak backup next to the others.");
    del->setEnabled(false);

    // ── registry ─────────────────────────────────────────────────────────────
    RegInfo info;
    info.options = options;
    const std::map<std::string, ParamDescriptor*>& defined = DescPeek::defined(desc);
    for (const auto& kv : defined) {
        const std::string& id = kv.first;
        ParamDescriptor* d = kv.second;
        if (!d) continue;
        if (!capturableType(d->getType())) continue;
        if (alwaysExcluded(id)) continue;
        bool excluded = false;
        for (const std::string& x : options.excludeIds) if (id == x) { excluded = true; break; }
        for (const std::string& x : options.excludePrefixes) if (!excluded && startsWith(id, x)) excluded = true;
        if (excluded) continue;
        PropertySet& props = d->getPropertySet();
        if (props.propGetInt(kOfxParamPropPersistant, 0, false) == 0) continue;
        if (d->getType() == eStringParam &&
            props.propGetString(kOfxParamPropStringMode, 0, false) == kOfxParamStringIsLabel)
            continue;
        RegEntry e;
        e.name = id;
        e.type = d->getType();
        info.params.push_back(e);
    }
    std::lock_guard<std::mutex> lock(regMutex());
    registry()[options.pluginId] = info; // describeInContext may run per context: overwrite
}

std::vector<std::string> registeredParams(const std::string& pluginId)
{
    std::vector<std::string> out;
    std::lock_guard<std::mutex> lock(regMutex());
    auto it = registry().find(pluginId);
    if (it != registry().end())
        for (const RegEntry& e : it->second.params) out.push_back(e.name);
    return out;
}

// ── Controller ────────────────────────────────────────────────────────────────
struct Controller::Impl {
    OFX::ImageEffect& fx;
    RegInfo reg;
    bool haveReg = false;
    AfterApplyFn afterApply;
    fs::path dir;
    bool busy = false;
    std::vector<std::string> menuNames; // index -> slot name ("" = Factory)

    OFX::StringParam*     currentDefault = nullptr;
    OFX::ChoiceParam*     choice = nullptr;
    OFX::StringParam*     selected = nullptr;
    OFX::StringParam*     name = nullptr;
    OFX::PushButtonParam* update = nullptr;
    OFX::PushButtonParam* del = nullptr;

    Impl(OFX::ImageEffect& e) : fx(e) {}

    struct Busy {
        bool& b; bool prev;
        explicit Busy(bool& f) : b(f), prev(f) { b = true; }
        ~Busy() { b = prev; }
    };

    void message(OFX::Message::MessageTypeEnum type, const std::string& msg)
    {
        try { fx.sendMessage(type, "", msg); } catch (...) {}
    }

    std::string selectedName()
    {
        std::string s;
        try { if (selected) selected->getValue(s); } catch (...) {}
        return s;
    }

    void setStringIfDifferent(OFX::StringParam* p, const std::string& v)
    {
        if (!p) return;
        try {
            std::string cur;
            p->getValue(cur);
            if (cur != v) p->setValue(v);
        } catch (...) {}
    }

    // Rebuild the menu, the default readout and the enabled state. Touches only
    // the non-persistent factory_* params.
    void refreshControls(bool refreshDisk)
    {
        if (!choice) return;
        Busy guard(busy);
        std::vector<Slot> slots = listSlots(dir, refreshDisk);
        std::string def = readDefaultName(dir);
        bool defFound = def.empty();
        for (const Slot& s : slots) if (!def.empty() && sameName(s.name, def)) { def = s.name; defFound = true; }
        if (!defFound) def.clear(); // pointer to a missing slot reads as Factory

        std::string sel = selectedName();
        menuNames.clear();
        menuNames.push_back(std::string());
        int selIndex = 0;
        try {
            choice->resetOptions();
            choice->appendOption(kFactoryLabel);
            for (const Slot& s : slots) {
                std::string label = s.name;
                if (!def.empty() && sameName(s.name, def)) label += kDefaultSuffix;
                choice->appendOption(label);
                menuNames.push_back(s.name);
                if (!sel.empty() && sameName(s.name, sel)) selIndex = (int)menuNames.size() - 1;
            }
            if (selIndex == 0) sel.clear();
            int cur = 0;
            choice->getValue(cur);
            if (cur != selIndex) choice->setValue(selIndex);
        } catch (...) {}
        setStringIfDifferent(selected, sel);
        setStringIfDifferent(currentDefault, def.empty() ? std::string(kFactoryLabel) : def);
        try {
            if (update) update->setEnabled(!sel.empty());
            if (del) del->setEnabled(!sel.empty());
        } catch (...) {}
    }

    Slot capture(double t)
    {
        Slot s;
        s.pluginId = reg.options.pluginId;
        s.pluginVersion = reg.options.pluginVersion;
        s.savedAt = nowIso8601();
        for (const RegEntry& e : reg.params) {
            Val v;
            if (!readParam(fx, e, t, false, v)) continue;
            if (e.type == OFX::eStringParam) {
                s.strings[e.name] = v.str;
            } else if (e.type == OFX::eChoiceParam) {
                ChoiceValue cv;
                cv.index = ri(v.num[0]);
                try { fx.fetchChoiceParam(e.name)->getOption(cv.index, cv.label); } catch (...) {}
                s.choices[e.name] = cv;
            } else {
                s.values[e.name] = v.num;
            }
        }
        return s;
    }

    // Target value for one param: from the slot when it has a usable entry,
    // otherwise the param default (missing keys fill from defaults).
    bool target(const RegEntry& e, const Slot* slot, const Val& def, Val& out)
    {
        out = def;
        if (!slot) return true;
        if (e.type == OFX::eStringParam) {
            auto it = slot->strings.find(e.name);
            if (it != slot->strings.end()) out.str = it->second;
            return true;
        }
        if (e.type == OFX::eChoiceParam) {
            auto it = slot->choices.find(e.name);
            if (it == slot->choices.end()) return true;
            try {
                OFX::ChoiceParam* p = fx.fetchChoiceParam(e.name);
                std::vector<std::string> opts;
                const int n = p->getNOptions();
                for (int i = 0; i < n; ++i) {
                    std::string opt;
                    p->getOption(i, opt);
                    opts.push_back(opt);
                }
                out.num = { (double)resolveChoiceIndex(&it->second, opts, ri(def.num[0])) };
            } catch (...) {}
            return true;
        }
        auto it = slot->values.find(e.name);
        out.num = resolveNumeric(it == slot->values.end() ? nullptr : &it->second, def.num);
        return true;
    }

    // The one apply path, used by Apply Selected, Reset to Factory and the
    // new-node default. slot == nullptr means Factory. Keyframed params are
    // left alone and counted.
    int applyValues(const Slot* slot, double t)
    {
        Busy guard(busy);
        int skippedKeyed = 0;
        for (const RegEntry& e : reg.params) {
            Val cur, def, want;
            if (!readParam(fx, e, t, false, cur)) continue;
            if (!readParam(fx, e, t, true, def)) continue;
            target(e, slot, def, want);
            if (valEqual(e, cur, want)) continue;
            OFX::ValueParam* vp = valueParam(fx, e);
            try {
                if (vp && vp->getNumKeys() > 0) { ++skippedKeyed; continue; }
            } catch (...) {}
            writeParam(fx, e, want);
        }
        return skippedKeyed;
    }

    bool allAtDefault()
    {
        for (const RegEntry& e : reg.params) {
            Val cur, def;
            if (!readParam(fx, e, 0.0, false, cur)) return false;
            if (!readParam(fx, e, 0.0, true, def)) return false;
            if (!valEqual(e, cur, def)) return false;
        }
        return true;
    }

    void runAfterApply(double t)
    {
        if (!afterApply) return;
        try { afterApply(t); } catch (...) {}
    }

    void userApply(const Slot* slot, double t, const char* undoLabel)
    {
        int skipped = 0;
        fx.beginEditBlock(undoLabel); // one undo step
        try {
            skipped = applyValues(slot, t);
            {
                Busy guard(busy);
                runAfterApply(t);
            }
        } catch (...) {}
        fx.endEditBlock();
        if (skipped > 0) {
            message(OFX::Message::eMessageWarning,
                    "Left " + std::to_string(skipped) + " keyframed control(s) unchanged. "
                    "Remove their keyframes to apply a starting point to them.");
        }
    }

    void saveFailed(const std::string& err)
    {
        (void)err;
        message(OFX::Message::eMessageError,
                "Could not save the starting point. Check that this folder is writable:\n" + dir.u8string());
    }

    void notFound(const std::string& n)
    {
        message(OFX::Message::eMessageError,
                "Starting point \"" + n + "\" was not found. It may have been deleted from another node.");
        setStringIfDifferent(selected, std::string());
        refreshControls(true);
    }

    void onSaveNew(double t)
    {
        std::string raw;
        try { name->getValue(raw); } catch (...) {}
        const std::string clean = sanitizeName(raw);
        if (clean.empty()) { message(OFX::Message::eMessageError, "Type a name for the starting point first."); return; }
        if (isReservedName(clean)) { message(OFX::Message::eMessageError, "\"Factory\" is reserved. Pick another name."); return; }
        Slot existing;
        Slot s = capture(t);
        s.name = clean;
        s.path = findSlot(dir, clean, existing, true) ? existing.path : uniquePath(dir, clean);
        std::string err;
        if (!writeSlot(s, err)) { saveFailed(err); return; }
        setStringIfDifferent(selected, clean);
        refreshControls(true);
    }

    void onUpdateSelected(double t)
    {
        const std::string sel = selectedName();
        if (sel.empty()) {
            message(OFX::Message::eMessageError, "Factory cannot be changed. Use Save Current as New Starting Point instead.");
            return;
        }
        Slot existing;
        if (!findSlot(dir, sel, existing, true)) { notFound(sel); return; }
        Slot s = capture(t);
        s.name = existing.name;
        s.path = existing.path;
        std::string err;
        if (!writeSlot(s, err)) { saveFailed(err); return; }
        refreshControls(true);
    }

    void onMakeDefault()
    {
        const std::string sel = selectedName();
        std::string chosen;
        if (!sel.empty()) {
            Slot existing;
            if (!findSlot(dir, sel, existing, true)) { notFound(sel); return; }
            chosen = existing.name;
        }
        std::string err;
        if (!writeDefaultName(dir, chosen, err)) { saveFailed(err); return; }
        refreshControls(true);
        const std::string tool = reg.options.toolName.empty() ? std::string("Look Dev Tools") : reg.options.toolName;
        message(OFX::Message::eMessageMessage,
                "New " + tool + " nodes will open with \"" + (chosen.empty() ? std::string(kFactoryLabel) : chosen) +
                "\". Nodes already in your projects are not changed.");
    }

    void onApplySelected(double t)
    {
        const std::string sel = selectedName();
        if (sel.empty()) { userApply(nullptr, t, "Reset to Factory"); return; }
        Slot s;
        if (!findSlot(dir, sel, s, true)) { notFound(sel); return; }
        userApply(&s, t, "Apply Starting Point");
    }

    void onDeleteSelected()
    {
        const std::string sel = selectedName();
        if (sel.empty()) { message(OFX::Message::eMessageError, "Factory cannot be deleted."); return; }
        Slot s;
        if (!findSlot(dir, sel, s, true)) { notFound(sel); return; }
        fs::path bak;
        std::string err;
        if (!renameToBackup(s.path, bak, err)) {
            message(OFX::Message::eMessageError, "Could not delete the starting point. Check that this folder is writable:\n" + dir.u8string());
            return;
        }
        if (sameName(readDefaultName(dir), s.name)) {
            std::string err2;
            writeDefaultName(dir, std::string(), err2);
        }
        setStringIfDifferent(selected, std::string());
        refreshControls(true);
    }

    void onChoiceChanged()
    {
        int ix = 0;
        try { choice->getValue(ix); } catch (...) { return; }
        std::string sel = (ix >= 0 && ix < (int)menuNames.size()) ? menuNames[ix] : std::string();
        setStringIfDifferent(selected, sel);
        if (!sel.empty()) setStringIfDifferent(name, sel);
        refreshControls(false);
    }
};

Controller::Controller(OFX::ImageEffect& effect, const std::string& pluginId, AfterApplyFn afterApply)
    : _impl(new Impl(effect))
{
    Impl& m = *_impl;
    m.afterApply = afterApply;
    {
        std::lock_guard<std::mutex> lock(regMutex());
        auto it = registry().find(pluginId);
        if (it != registry().end()) { m.reg = it->second; m.haveReg = true; }
    }
    if (!m.haveReg) return; // describe never ran for this id: stay inert
    m.dir = storeDirectory(m.reg.options.storeKey);
    try {
        m.currentDefault = effect.fetchStringParam(kCurrentDefault);
        m.choice         = effect.fetchChoiceParam(kChoice);
        m.selected       = effect.fetchStringParam(kSelected);
        m.name           = effect.fetchStringParam(kName);
        m.update         = effect.fetchPushButtonParam(kUpdateSelected);
        m.del            = effect.fetchPushButtonParam(kDeleteSelected);
    } catch (...) {
        m.haveReg = false;
        m.choice = nullptr;
    }
}

Controller::~Controller() = default;

void Controller::onConstructed()
{
    Impl& m = *_impl;
    if (!m.haveReg) return;
    m.refreshControls(false); // non-persistent UI only

    // New-node test. All three must hold:
    //   1. the persistent init flag is still false,
    //   2. the uuid is still empty (bind() has not minted one yet, so this
    //      instance has never been saved with the control channel, and it is
    //      not a paste or clone, which carry the source uuid),
    //   3. every captured param sits at its default.
    OFX::BooleanParam* flag = nullptr;
    try {
        flag = m.fx.fetchBooleanParam(kInitFlagParam);
        bool done = false;
        flag->getValue(done);
        if (done) return;
        if (!m.fx.paramExists(kUuidParam)) return;
        std::string uuid;
        m.fx.fetchStringParam(kUuidParam)->getValue(uuid);
        if (!uuid.empty()) return;
    } catch (...) {
        return;
    }
    if (!m.allAtDefault()) return;

    // Brand-new node: apply the default (no edit block in the ctor) and mark it.
    Impl::Busy guard(m.busy);
    const std::string def = readDefaultName(m.dir);
    if (!def.empty()) {
        Slot s;
        if (findSlot(m.dir, def, s, false)) {
            m.applyValues(&s, 0.0);
            m.runAfterApply(0.0);
        }
    }
    try { flag->setValue(true); } catch (...) {}
}

bool Controller::handleChangedParam(const OFX::InstanceChangedArgs& args, const std::string& name)
{
    Impl& m = *_impl;
    if (!m.haveReg) return false;
    if (m.busy) return true; // our own setValue echoes, and tool params mid-apply
    if (!startsWith(name, kPrefix)) return false;
    if (args.reason != OFX::eChangeUserEdit) return true;

    const double t = args.time;
    if      (name == kChoice)          m.onChoiceChanged();
    else if (name == kSaveNew)         m.onSaveNew(t);
    else if (name == kUpdateSelected)  m.onUpdateSelected(t);
    else if (name == kMakeDefault)     m.onMakeDefault();
    else if (name == kApplySelected)   m.onApplySelected(t);
    else if (name == kResetToFactory)  m.userApply(nullptr, t, "Reset to Factory");
    else if (name == kDeleteSelected)  m.onDeleteSelected();
    return true;
}

} // namespace factory
} // namespace ldt

#endif // LDT_FACTORY_NO_OFX
