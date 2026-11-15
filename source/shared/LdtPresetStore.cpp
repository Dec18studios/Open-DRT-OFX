// LdtPresetStore.cpp - host-free user preset store for 6A OpenDRT.
// See LdtPresetStore.h and docs/design/OPENDRT-PRESET-MANAGER-DESIGN.md.
//
// Reuses the Factory Settings Layer A helpers (atomic write, .bak retire,
// timestamps, choice resolve, LookDevTools root) through the public header.
// The few helpers that are file-static in LdtFactorySettings.cpp (lower,
// readTextFile) have small local equivalents here.

#include "LdtPresetStore.h"
#include "LdtFactorySettings.h"
#include "ldt_json.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <system_error>

namespace fs = std::filesystem;

namespace ldt {
namespace presets {

// -- local helpers ------------------------------------------------------------
namespace {

std::string lowerAscii(const std::string& s)
{
    std::string out(s);
    for (char& c : out) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    return out;
}

bool endsWith(const std::string& s, const std::string& suffix)
{
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool endsWithNoCase(const std::string& s, const std::string& suffix)
{
    return endsWith(lowerAscii(s), lowerAscii(suffix));
}

void trimSpaces(std::string& s)
{
    while (!s.empty() && s.back() == ' ') s.pop_back();
    size_t i = 0;
    while (i < s.size() && s[i] == ' ') ++i;
    s.erase(0, i);
}

// Cut to at most maxBytes without leaving half a UTF-8 sequence.
void cutUtf8(std::string& s, size_t maxBytes)
{
    if (s.size() <= maxBytes) return;
    size_t n = maxBytes;
    // Step back over continuation bytes, then drop the lead byte if its
    // sequence would not fit.
    while (n > 0 && (((unsigned char)s[n]) & 0xC0) == 0x80) --n;
    s.resize(n);
}

// Read at most cap+1 bytes so a file that grows between stat and read is still
// caught. Returns false when the file cannot be opened.
bool readCapped(const fs::path& p, size_t cap, std::string& out, bool& tooBig)
{
    tooBig = false;
    std::ifstream in(p, std::ios::in | std::ios::binary);
    if (!in) return false;
    out.clear();
    std::vector<char> buf(64 * 1024);
    while (in) {
        in.read(buf.data(), (std::streamsize)buf.size());
        const std::streamsize got = in.gcount();
        if (got <= 0) break;
        out.append(buf.data(), (size_t)got);
        if (out.size() > cap) { tooBig = true; return true; }
    }
    return !in.bad();
}

// Shortest of %.15g / %.16g / %.17g that reads back to the same double, with
// the decimal point forced to '.' whatever the C locale says.
std::string formatNumber(double v)
{
    char buf[64];
    for (int prec = 15; prec <= 17; ++prec) {
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        if (prec == 17 || std::strtod(buf, nullptr) == v) break;
    }
    std::string s(buf);
    for (char& c : s) if (c == ',') c = '.';
    return s;
}

std::string quote(const std::string& s) { return "\"" + ldtjson::escape(s) + "\""; }

bool isPresetFileName(const fs::path& p)
{
    const std::string fn = p.filename().u8string();
    if (fn.empty() || fn[0] == '.') return false;   // temp files and hidden files
    return endsWith(lowerAscii(fn), ".json");
}

// First free <path>.bak / <path>.bak.N, matching renameToBackup's naming.
bool copyToBackup(const fs::path& path, fs::path& bakOut, std::string& err)
{
    std::error_code ec;
    fs::path bak = path;
    bak += ".bak";
    for (int n = 2; fs::exists(bak, ec) && n < 10000; ++n) {
        bak = path;
        bak += ".bak." + std::to_string(n);
    }
    if (!fs::copy_file(path, bak, fs::copy_options::none, ec) || ec) {
        err = "could not back up " + path.filename().u8string() + ": " + ec.message();
        return false;
    }
    bakOut = bak;
    return true;
}

// Stamp a fresh backup with the current time (a rename keeps the old mtime)
// and trim its siblings down to kMaxBackupsPerPreset. Best effort: a backup
// that cannot be stamped or pruned never fails the save.
void stampAndPrune(const fs::path& file, const fs::path& bak)
{
    std::error_code ec;
    if (!bak.empty()) fs::last_write_time(bak, fs::file_time_type::clock::now(), ec);
    pruneBackups(file, kMaxBackupsPerPreset, bak);
}

std::vector<std::string> modelSuffixKeys(const std::vector<std::string>& known)
{
    // Strip model suffixes for every key this build knows plus the planned
    // ones, so a name can never fake a model label.
    static const std::vector<std::string> planned { "default", "v101", "v11", "tesseract", "chromagnon" };
    std::vector<std::string> merged = planned;
    for (const auto& k : known)
        if (std::find(merged.begin(), merged.end(), k) == merged.end()) merged.push_back(k);
    return merged;
}

} // namespace

// -- backups ------------------------------------------------------------------
int pruneBackups(const fs::path& file, int keep, const fs::path& justMade)
{
    if (keep < 0) keep = 0;
    std::error_code ec;
    const fs::path dir = file.parent_path();
    const std::string base = file.filename().u8string() + ".bak";
    struct Bak { fs::path path; fs::file_time_type mtime; long n; bool fresh; };
    std::vector<Bak> baks;
    fs::directory_iterator it(dir, ec), end;
    for (; !ec && it != end; it.increment(ec)) {
        const std::string fn = it->path().filename().u8string();
        long n = -1;
        if (fn == base) n = 1;
        else if (fn.size() > base.size() + 1 && fn.compare(0, base.size() + 1, base + ".") == 0) {
            const std::string tail = fn.substr(base.size() + 1);
            if (tail.size() > 9 || tail.find_first_not_of("0123456789") != std::string::npos) continue;
            n = std::strtol(tail.c_str(), nullptr, 10);
        }
        if (n < 1) continue;
        std::error_code e2;
        if (!it->is_regular_file(e2)) continue;
        Bak b;
        b.path = it->path();
        b.mtime = fs::last_write_time(b.path, e2);
        b.n = n;
        b.fresh = !justMade.empty() && b.path.filename() == justMade.filename();
        baks.push_back(b);
    }
    if ((int)baks.size() <= keep) return 0;
    std::sort(baks.begin(), baks.end(), [](const Bak& a, const Bak& b) {
        if (a.fresh != b.fresh) return a.fresh;
        if (a.mtime != b.mtime) return a.mtime > b.mtime;
        return a.n > b.n;
    });
    int removed = 0;
    for (std::size_t i = (std::size_t)keep; i < baks.size(); ++i) {
        std::error_code e3;
        if (fs::remove(baks[i].path, e3)) ++removed;
    }
    return removed;
}

// -- families -----------------------------------------------------------------
const char* familyKey(Family f)
{
    switch (f) {
        case Family::Full:      return "full";
        case Family::Look:      return "look";
        case Family::Tonescale: return "tonescale";
    }
    return "full";
}

bool familyFromKey(const std::string& key, Family& out)
{
    if (key == "full")      { out = Family::Full; return true; }
    if (key == "look")      { out = Family::Look; return true; }
    if (key == "tonescale") { out = Family::Tonescale; return true; }
    return false;
}

const char* familyMenuPrefix(Family f)
{
    switch (f) {
        case Family::Full:      return "";
        case Family::Look:      return "Look: ";
        case Family::Tonescale: return "Tone: ";
    }
    return "";
}

// -- names and ids ------------------------------------------------------------
std::string sanitizePresetName(const std::string& raw, const std::vector<std::string>& knownModels)
{
    std::string out;
    bool pendingSpace = false;
    for (char c : raw) {
        const unsigned char u = (unsigned char)c;
        if (u < 0x20 || u == 0x7f || c == ' ') { pendingSpace = true; continue; }
        if (std::string("/\\:*?\"<>|").find(c) != std::string::npos) { pendingSpace = true; continue; }
        if (pendingSpace && !out.empty()) out += ' ';
        pendingSpace = false;
        out += c;
    }
    // Leading dots would make hidden files on POSIX and "." / ".." names.
    while (!out.empty() && (out[0] == '.' || out[0] == ' ')) out.erase(0, 1);

    // Strip suffixes the menu adds, repeatedly, so "x (Custom) (missing)" and
    // "x (v11)" cannot pose as a decorated label.
    const std::vector<std::string> models = modelSuffixKeys(knownModels);
    bool stripped = true;
    while (stripped) {
        stripped = false;
        trimSpaces(out);
        if (endsWithNoCase(out, kCustomSuffix)) {
            out.resize(out.size() - std::string(kCustomSuffix).size()); stripped = true; continue;
        }
        if (endsWithNoCase(out, kMissingSuffix)) {
            out.resize(out.size() - std::string(kMissingSuffix).size()); stripped = true; continue;
        }
        for (const auto& m : models) {
            const std::string suf = " (" + m + ")";
            if (endsWithNoCase(out, suf)) { out.resize(out.size() - suf.size()); stripped = true; break; }
        }
    }
    cutUtf8(out, kMaxNameBytes);
    trimSpaces(out);
    if (out.empty()) out = kDefaultName;
    return out;
}

std::string normalizePresetNameKey(const std::string& name)
{
    std::string out;
    bool pendingSpace = false;
    for (char c : name) {
        const unsigned char u = (unsigned char)c;
        if (c == ' ' || c == '\t' || u < 0x20) { pendingSpace = true; continue; }
        if (pendingSpace && !out.empty()) out += ' ';
        pendingSpace = false;
        out += c;
    }
    return lowerAscii(out);
}

std::string sanitizeFileComponent(const std::string& raw)
{
    std::string out;
    for (char c : raw) {
        const unsigned char u = (unsigned char)c;
        if (u < 0x80 && std::isalnum(u)) out += (char)std::tolower(u);
        else if (!out.empty() && out.back() != '_') out += '_';
        if (out.size() >= 64) break;
    }
    while (!out.empty() && out.back() == '_') out.pop_back();
    if (out.size() > 64) out.resize(64);
    if (out.empty()) out = "preset";
    return out;
}

std::string makePresetId(Family f)
{
    static std::atomic<unsigned long> s_counter(0);
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const long long micros = (long long)std::chrono::duration_cast<std::chrono::microseconds>(now).count();
    const long long secs = micros / 1000000LL;
    const long long frac = micros % 1000000LL;
    return std::string(familyKey(f)) + "_" + std::to_string(secs) + "_" + std::to_string(frac) + "_" +
           std::to_string(++s_counter);
}

bool isReservedPresetName(const std::string& sanitizedName)
{
    const std::string k = normalizePresetNameKey(sanitizedName);
    return k == "none" || ldt::factory::isReservedName(sanitizedName);
}

std::string decorateLabel(const std::string& label, bool custom, bool missing)
{
    std::string out = label;
    if (missing) out += kMissingSuffix;
    else if (custom) out += kCustomSuffix;
    return out;
}

// -- config -------------------------------------------------------------------
StoreConfig openDrtConfig()
{
    StoreConfig c;
    c.foreignFormats["TechnicolorDRTPreset"] = "This is a Technicolor DRT preset, not an OpenDRT one.";
    c.foreignFormats["PhotoChemistPreset"]   = "This is a PhotoChemist preset, not an OpenDRT one.";
    c.foreignFormats[ldt::factory::kFormatName] =
        "This is a Factory Settings file. Load it from Factory Settings instead.";
    return c;
}

// -- serialize ----------------------------------------------------------------
std::string serialize(const Preset& p)
{
    // Keys in sorted order so files diff cleanly.
    std::ostringstream o;
    o << "{\n";
    o << "  \"createdAtUtc\": " << quote(p.createdAtUtc) << ",\n";
    o << "  \"family\": " << quote(familyKey(p.family)) << ",\n";
    o << "  \"format\": " << quote(kPresetFormat) << ",\n";
    o << "  \"id\": " << quote(p.id) << ",\n";
    o << "  \"model\": " << quote(p.model.empty() ? std::string(kDefaultModel) : p.model) << ",\n";
    o << "  \"name\": " << quote(p.name) << ",\n";
    o << "  \"plugin\": " << quote(p.plugin.empty() ? std::string(kSuitePluginId) : p.plugin) << ",\n";
    o << "  \"updatedAtUtc\": " << quote(p.updatedAtUtc) << ",\n";
    o << "  \"values\": {";
    bool first = true;
    for (const auto& kv : p.values) {
        const Value& v = kv.second;
        std::string text;
        switch (v.kind) {
            case Value::Kind::Number:
                if (!std::isfinite(v.number)) continue;   // never written
                text = formatNumber(v.number);
                break;
            case Value::Kind::Bool:
                text = v.boolean ? "true" : "false";
                break;
            case Value::Kind::Choice:
                text = "{ \"index\": " + std::to_string(v.index < 0 ? -1 : v.index) +
                       ", \"label\": " + quote(v.label) + " }";
                break;
        }
        o << (first ? "\n" : ",\n") << "    " << quote(kv.first) << ": " << text;
        first = false;
    }
    o << (first ? "},\n" : "\n  },\n");
    o << "  \"version\": " << kPresetVersion << "\n";
    o << "}\n";
    return o.str();
}

// -- parse --------------------------------------------------------------------
bool parse(const std::string& textIn, const StoreConfig& cfg, bool requireKnownModel,
           Preset& out, std::string& reason, int* skippedOut)
{
    if (skippedOut) *skippedOut = 0;
    std::string text = textIn;
    if (text.size() >= 3 && (unsigned char)text[0] == 0xEF && (unsigned char)text[1] == 0xBB &&
        (unsigned char)text[2] == 0xBF)
        text.erase(0, 3);
    if (text.size() > cfg.maxFileBytes) { reason = "This file is too big to be a preset."; return false; }

    ldtjson::Value root;
    std::string jerr;
    if (!ldtjson::parse(text, root, jerr) || !root.isObject()) {
        reason = "This file isn't a preset file.";
        return false;
    }

    const ldtjson::Value* fmt = root.find("format");
    if (!fmt || !fmt->isString()) { reason = "This file isn't a preset file."; return false; }
    if (fmt->str != kPresetFormat) {
        auto it = cfg.foreignFormats.find(fmt->str);
        reason = (it != cfg.foreignFormats.end()) ? it->second : "This file isn't an OpenDRT preset.";
        return false;
    }

    const ldtjson::Value* ver = root.find("version");
    double vd = 0.0;
    if (!ver || !ver->asDouble(vd) || !std::isfinite(vd) || vd != std::floor(vd) || vd < 1.0) {
        reason = "This preset file is damaged (no version).";
        return false;
    }
    if (vd > (double)kPresetVersion) {
        reason = "This preset was saved by a newer OpenDRT. Update LookDevTools to import it.";
        return false;
    }

    Preset p;
    if (const ldtjson::Value* pl = root.find("plugin")) {
        if (!pl->isString()) { reason = "This preset file is damaged."; return false; }
        if (std::find(cfg.acceptedPluginIds.begin(), cfg.acceptedPluginIds.end(), pl->str) ==
            cfg.acceptedPluginIds.end()) {
            reason = "This preset was made for a different plugin (" + pl->str + ").";
            return false;
        }
        p.plugin = pl->str;
    } else {
        p.plugin = cfg.pluginId;   // older or hand-written files: assume ours
    }

    const ldtjson::Value* fam = root.find("family");
    const std::string famKey = (fam && fam->isString()) ? fam->str : std::string("none");
    if (!familyFromKey(famKey, p.family)) {
        reason = "This preset is a kind this version doesn't know (" + famKey + ").";
        return false;
    }

    p.model = kDefaultModel;
    if (const ldtjson::Value* m = root.find("model")) {
        if (!m->isString()) { reason = "This preset file is damaged."; return false; }
        if (!m->str.empty()) p.model = m->str;
    }
    if (requireKnownModel &&
        std::find(cfg.knownModels.begin(), cfg.knownModels.end(), p.model) == cfg.knownModels.end()) {
        reason = "This preset needs a model this version doesn't have (" + p.model +
                 "). Update LookDevTools to use it.";
        return false;
    }

    const ldtjson::Value* nm = root.find("name");
    if (!nm || !nm->isString()) { reason = "This preset file is damaged (no name)."; return false; }
    p.sourceName = nm->str;
    p.name = sanitizePresetName(nm->str, cfg.knownModels);

    if (const ldtjson::Value* id = root.find("id")) {
        if (id->isString()) {
            // Ids land in file names: keep [a-z0-9_] only. An odd id just
            // becomes a different (still stable) id.
            std::string clean;
            for (char c : id->str) {
                const unsigned char u = (unsigned char)c;
                if (u < 0x80 && (std::isalnum(u) || c == '_')) clean += (char)std::tolower(u);
            }
            if (clean.size() > 96) clean.resize(96);
            p.id = clean;
        }
    }
    if (const ldtjson::Value* t = root.find("createdAtUtc")) if (t->isString()) p.createdAtUtc = t->str;
    if (const ldtjson::Value* t = root.find("updatedAtUtc")) if (t->isString()) p.updatedAtUtc = t->str;
    if (p.createdAtUtc.size() > 64) p.createdAtUtc.resize(64);
    if (p.updatedAtUtc.size() > 64) p.updatedAtUtc.resize(64);

    const ldtjson::Value* vals = root.find("values");
    if (!vals || !vals->isObject()) { reason = "This preset has no settings this version can use."; return false; }
    int skipped = 0;
    for (const auto& kv : vals->obj) {
        if (kv.first.empty() || kv.first.size() > 128) { ++skipped; continue; }
        const ldtjson::Value& v = kv.second;
        if (v.isNumber()) {
            if (!std::isfinite(v.number)) { reason = "This preset file is damaged."; return false; }
            p.values[kv.first] = Value::num(v.number);
        } else if (v.isBool()) {
            p.values[kv.first] = Value::flag(v.boolean);
        } else if (v.isString()) {
            p.values[kv.first] = Value::choice(v.str, -1);
        } else if (v.isObject()) {
            Value c = Value::choice(std::string(), -1);
            bool any = false;
            if (const ldtjson::Value* l = v.find("label")) if (l->isString()) { c.label = l->str; any = true; }
            if (const ldtjson::Value* i = v.find("index")) {
                if (i->isNumber()) {
                    if (!std::isfinite(i->number)) { reason = "This preset file is damaged."; return false; }
                    if (i->number >= 0.0 && i->number < 100000.0) { c.index = (int)std::llround(i->number); any = true; }
                }
            }
            if (any) p.values[kv.first] = c; else ++skipped;
        } else {
            ++skipped;   // null, array
        }
    }
    if (skippedOut) *skippedOut = skipped;
    if (p.values.empty()) { reason = "This preset has no settings this version can use."; return false; }

    out = p;
    return true;
}

bool readPresetFile(const fs::path& file, const StoreConfig& cfg, bool requireKnownModel,
                    Preset& out, std::string& reason, int* skippedOut)
{
    std::error_code ec;
    if (!fs::is_regular_file(file, ec)) { reason = "Couldn't find that file."; return false; }
    const auto sz = fs::file_size(file, ec);
    if (!ec && sz > cfg.maxFileBytes + 3) { reason = "This file is too big to be a preset."; return false; }
    std::string text;
    bool tooBig = false;
    if (!readCapped(file, cfg.maxFileBytes + 3, text, tooBig)) { reason = "Couldn't read that file."; return false; }
    if (tooBig) { reason = "This file is too big to be a preset."; return false; }
    if (!parse(text, cfg, requireKnownModel, out, reason, skippedOut)) return false;
    out.path = file;
    return true;
}

// -- directory ----------------------------------------------------------------
namespace {
std::mutex& dirMutex() { static std::mutex m; return m; }
fs::path& dirOverride() { static fs::path p; return p; }
}

fs::path presetDirectory()
{
    {
        std::lock_guard<std::mutex> lock(dirMutex());
        if (!dirOverride().empty()) return dirOverride();
    }
    // baseDirectory() is <root>/LookDevTools/starting-points; share its root.
    return ldt::factory::baseDirectory().parent_path() / "OpenDRT" / "presets";
}

void setPresetDirectoryOverride(const fs::path& dir)
{
    std::lock_guard<std::mutex> lock(dirMutex());
    dirOverride() = dir;
}

Store& sharedOpenDrtStore()
{
    static std::mutex m;
    static std::unique_ptr<Store> s;
    std::lock_guard<std::mutex> lock(m);
    if (!s) s.reset(new Store(presetDirectory(), openDrtConfig()));
    return *s;
}

// -- store --------------------------------------------------------------------
namespace {
bool presetLess(const Preset& a, const Preset& b)
{
    if (a.family != b.family) return (int)a.family < (int)b.family;
    const std::string ka = normalizePresetNameKey(a.name), kb = normalizePresetNameKey(b.name);
    if (ka != kb) return ka < kb;
    if (a.id != b.id) return a.id < b.id;
    return a.path < b.path;
}

OpResult fail(Status s, const std::string& reason)
{
    OpResult r;
    r.status = s;
    r.reason = reason;
    return r;
}

OpResult okResult(const Preset& p)
{
    OpResult r;
    r.status = Status::Ok;
    r.preset = p;
    return r;
}

const char* kGoneMessage = "That preset is gone. It may have been deleted somewhere else.";
const char* kReadOnlyMessage = "That preset belongs to a model this version doesn't have, so it's left as is.";
}

Store::Store(fs::path dir, StoreConfig cfg) : _dir(std::move(dir)), _cfg(std::move(cfg)) {}

bool Store::isKnownModel(const std::string& m) const
{
    return std::find(_cfg.knownModels.begin(), _cfg.knownModels.end(), m) != _cfg.knownModels.end();
}

void Store::scanLocked() const
{
    _cache.clear();
    _scanned = true;
    std::error_code ec;
    if (!fs::is_directory(_dir, ec)) return;
    std::vector<fs::path> files;
    for (fs::directory_iterator it(_dir, ec), end; !ec && it != end; it.increment(ec)) {
        std::error_code ec2;
        if (!it->is_regular_file(ec2)) continue;
        if (!isPresetFileName(it->path())) continue;
        files.push_back(it->path());
    }
    std::sort(files.begin(), files.end());
    for (const auto& f : files) {
        Preset p;
        std::string reason;
        if (!readPresetFile(f, _cfg, false, p, reason)) continue;   // left on disk, untouched
        if (p.id.empty()) p.id = "file_" + sanitizeFileComponent(f.stem().u8string());
        bool dup = false;
        for (const auto& q : _cache) if (q.id == p.id) { dup = true; break; }
        if (dup) continue;   // first file (by path) wins; the other stays on disk
        _cache.push_back(p);
    }
    std::sort(_cache.begin(), _cache.end(), presetLess);
}

void Store::ensureScannedLocked() const
{
    if (!_scanned) scanLocked();
}

void Store::reload()
{
    std::lock_guard<std::mutex> lock(_mutex);
    scanLocked();
}

std::vector<Preset> Store::list(const std::string& model) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    ensureScannedLocked();
    std::vector<Preset> out;
    for (const auto& p : _cache) if (p.model == model) out.push_back(p);
    return out;
}

std::vector<Preset> Store::list(const std::string& model, Family f) const
{
    std::lock_guard<std::mutex> lock(_mutex);
    ensureScannedLocked();
    std::vector<Preset> out;
    for (const auto& p : _cache) if (p.model == model && p.family == f) out.push_back(p);
    return out;
}

const Preset* Store::findCachedById(const std::string& id) const
{
    for (const auto& p : _cache) if (p.id == id) return &p;
    return nullptr;
}

const Preset* Store::findCachedByName(Family f, const std::string& model, const std::string& key,
                                      const std::string& exceptId) const
{
    for (const auto& p : _cache)
        if (p.family == f && p.model == model && p.id != exceptId && normalizePresetNameKey(p.name) == key)
            return &p;
    return nullptr;
}

std::string Store::uniqueName(Family f, const std::string& model, const std::string& base) const
{
    if (!findCachedByName(f, model, normalizePresetNameKey(base), std::string())) return base;
    for (int n = 2; n < 100000; ++n) {
        const std::string suffix = " " + std::to_string(n);
        std::string stem = base;
        cutUtf8(stem, kMaxNameBytes > suffix.size() ? kMaxNameBytes - suffix.size() : 0);
        trimSpaces(stem);
        const std::string cand = stem + suffix;
        if (!findCachedByName(f, model, normalizePresetNameKey(cand), std::string())) return cand;
    }
    return base + " " + makePresetId(f);
}

fs::path Store::pathFor(const Preset& p) const
{
    const std::string stem = std::string(familyKey(p.family)) + "__" + sanitizeFileComponent(p.name) + "__" +
                             sanitizeFileComponent(p.id);
    return _dir / fs::u8path(stem + ".json");
}

// Writes a record that does not exist on disk yet. Never overwrites a file.
OpResult Store::writeNewLocked(Preset p)
{
    fs::path target = pathFor(p);
    std::error_code ec;
    for (int n = 2; fs::exists(target, ec) && n < 10000; ++n) {
        const std::string stem = pathFor(p).stem().u8string() + "_" + std::to_string(n);
        target = _dir / fs::u8path(stem + ".json");
    }
    if (fs::exists(target, ec)) return fail(Status::Failed, "Couldn't find a free file name.");
    std::string err;
    p.path = target;
    if (!ldt::factory::writeAtomic(target, serialize(p), err))
        return fail(Status::Failed, "Couldn't save the preset (" + err + ").");
    _cache.push_back(p);
    std::sort(_cache.begin(), _cache.end(), presetLess);
    return okResult(p);
}

// Swaps `existing` for `incoming` (same id). The old file always ends up as a
// .bak (older backups past kMaxBackupsPerPreset are pruned). On any failure
// the old file is left intact.
OpResult Store::commitOverLocked(const Preset& existing, Preset incoming)
{
    std::error_code ec;
    const fs::path newPath = pathFor(incoming);
    std::string err;
    if (newPath == existing.path) {
        fs::path bak;
        if (!copyToBackup(existing.path, bak, err)) return fail(Status::Failed, "Couldn't save the preset (" + err + ").");
        incoming.path = newPath;
        if (!ldt::factory::writeAtomic(newPath, serialize(incoming), err))
            return fail(Status::Failed, "Couldn't save the preset (" + err + ").");
        stampAndPrune(existing.path, bak);
    } else {
        fs::path target = newPath;
        for (int n = 2; fs::exists(target, ec) && n < 10000; ++n)
            target = _dir / fs::u8path(newPath.stem().u8string() + "_" + std::to_string(n) + ".json");
        incoming.path = target;
        if (!ldt::factory::writeAtomic(target, serialize(incoming), err))
            return fail(Status::Failed, "Couldn't save the preset (" + err + ").");
        fs::path bak;
        if (fs::exists(existing.path, ec) && !ldt::factory::renameToBackup(existing.path, bak, err)) {
            // New file is in place; the old one could not be retired. Report it
            // rather than unlinking anything.
            for (auto& q : _cache) if (q.id == existing.id) q = incoming;
            std::sort(_cache.begin(), _cache.end(), presetLess);
            OpResult r = okResult(incoming);
            r.reason = "Saved, but the old file couldn't be moved aside (" + err + ").";
            return r;
        }
        if (!bak.empty()) stampAndPrune(existing.path, bak);
    }
    bool replaced = false;
    for (auto& q : _cache) if (q.id == existing.id) { q = incoming; replaced = true; }
    if (!replaced) _cache.push_back(incoming);
    std::sort(_cache.begin(), _cache.end(), presetLess);
    return okResult(incoming);
}

OpResult Store::loadById(const std::string& id)
{
    std::lock_guard<std::mutex> lock(_mutex);
    ensureScannedLocked();
    const Preset* c = findCachedById(id);
    if (!c) return fail(Status::NotFound, kGoneMessage);
    Preset fresh;
    std::string reason;
    const fs::path path = c->path;
    if (!readPresetFile(path, _cfg, false, fresh, reason)) {
        // Vanished or damaged since the scan: drop it from the cache only.
        _cache.erase(std::remove_if(_cache.begin(), _cache.end(),
                                    [&](const Preset& q) { return q.id == id; }), _cache.end());
        return fail(Status::NotFound, kGoneMessage);
    }
    if (fresh.id.empty()) fresh.id = id;
    if (fresh.id != id) {
        // Edited by hand since the scan; refresh the cache entry.
        _cache.erase(std::remove_if(_cache.begin(), _cache.end(),
                                    [&](const Preset& q) { return q.id == id; }), _cache.end());
        return fail(Status::NotFound, kGoneMessage);
    }
    return okResult(fresh);
}

OpResult Store::loadByName(Family f, const std::string& model, const std::string& name)
{
    std::string id;
    {
        std::lock_guard<std::mutex> lock(_mutex);
        ensureScannedLocked();
        const Preset* c = findCachedByName(f, model, normalizePresetNameKey(name), std::string());
        if (!c) return fail(Status::NotFound, kGoneMessage);
        id = c->id;
    }
    return loadById(id);
}

OpResult Store::saveNew(Preset p, ClashPolicy policy)
{
    std::lock_guard<std::mutex> lock(_mutex);
    ensureScannedLocked();
    if (p.model.empty()) p.model = kDefaultModel;
    if (!isKnownModel(p.model))
        return fail(Status::Rejected, "This version doesn't have the model (" + p.model + ").");
    p.name = sanitizePresetName(p.name, _cfg.knownModels);
    if (isReservedPresetName(p.name)) return fail(Status::Rejected, "That name is taken by the menu. Pick another.");
    if (p.values.empty()) return fail(Status::Rejected, "Nothing to save.");
    p.plugin = _cfg.pluginId;
    p.createdAtUtc = p.updatedAtUtc = ldt::factory::nowIso8601();
    p.id = makePresetId(p.family);
    while (findCachedById(p.id)) p.id = makePresetId(p.family);

    if (const Preset* clash = findCachedByName(p.family, p.model, normalizePresetNameKey(p.name), std::string())) {
        if (policy == ClashPolicy::Fail) {
            OpResult r = fail(Status::NameClash, "You already have a preset called \"" + clash->name + "\".");
            r.preset = *clash;
            return r;
        }
        if (policy == ClashPolicy::Replace) {
            const Preset old = *clash;
            p.id = old.id;
            p.createdAtUtc = old.createdAtUtc;
            return commitOverLocked(old, p);
        }
        p.name = uniqueName(p.family, p.model, p.name);
    }
    return writeNewLocked(p);
}

OpResult Store::update(const std::string& id, const std::map<std::string, Value>& values)
{
    std::lock_guard<std::mutex> lock(_mutex);
    ensureScannedLocked();
    const Preset* c = findCachedById(id);
    if (!c) return fail(Status::NotFound, kGoneMessage);
    if (!isKnownModel(c->model)) return fail(Status::ReadOnly, kReadOnlyMessage);
    std::error_code ec;
    if (!fs::exists(c->path, ec)) return fail(Status::NotFound, kGoneMessage);
    if (values.empty()) return fail(Status::Rejected, "Nothing to save.");
    const Preset old = *c;
    Preset p = old;
    p.values = values;
    p.updatedAtUtc = ldt::factory::nowIso8601();
    if (p.createdAtUtc.empty()) p.createdAtUtc = p.updatedAtUtc;
    return commitOverLocked(old, p);
}

OpResult Store::rename(const std::string& id, const std::string& newName)
{
    std::lock_guard<std::mutex> lock(_mutex);
    ensureScannedLocked();
    const Preset* c = findCachedById(id);
    if (!c) return fail(Status::NotFound, kGoneMessage);
    if (!isKnownModel(c->model)) return fail(Status::ReadOnly, kReadOnlyMessage);
    std::error_code ec;
    if (!fs::exists(c->path, ec)) return fail(Status::NotFound, kGoneMessage);
    const std::string clean = sanitizePresetName(newName, _cfg.knownModels);
    if (isReservedPresetName(clean)) return fail(Status::Rejected, "That name is taken by the menu. Pick another.");
    if (const Preset* clash = findCachedByName(c->family, c->model, normalizePresetNameKey(clean), id)) {
        OpResult r = fail(Status::NameClash, "You already have a preset called \"" + clash->name + "\".");
        r.preset = *clash;
        return r;
    }
    const Preset old = *c;
    if (old.name == clean) return okResult(old);
    Preset p = old;
    p.name = clean;
    p.updatedAtUtc = ldt::factory::nowIso8601();
    return commitOverLocked(old, p);
}

OpResult Store::remove(const std::string& id)
{
    std::lock_guard<std::mutex> lock(_mutex);
    ensureScannedLocked();
    const Preset* c = findCachedById(id);
    if (!c) return fail(Status::NotFound, kGoneMessage);
    if (!isKnownModel(c->model)) return fail(Status::ReadOnly, kReadOnlyMessage);
    const Preset old = *c;
    fs::path bak;
    std::string err;
    std::error_code ec;
    if (fs::exists(old.path, ec) && !ldt::factory::renameToBackup(old.path, bak, err))
        return fail(Status::Failed, "Couldn't delete the preset (" + err + ").");
    if (!bak.empty()) stampAndPrune(old.path, bak);
    _cache.erase(std::remove_if(_cache.begin(), _cache.end(),
                                [&](const Preset& q) { return q.id == id; }), _cache.end());
    return okResult(old);
}

OpResult Store::exportPreset(const Preset& p, const fs::path& dest)
{
    if (dest.empty()) return fail(Status::Failed, "No file picked.");
    Preset out = p;
    out.path.clear();
    if (out.plugin.empty()) out.plugin = kSuitePluginId;
    if (out.model.empty()) out.model = kDefaultModel;
    if (out.createdAtUtc.empty()) out.createdAtUtc = ldt::factory::nowIso8601();
    if (out.updatedAtUtc.empty()) out.updatedAtUtc = out.createdAtUtc;
    if (out.id.empty()) out.id = makePresetId(out.family);
    std::string err;
    if (!ldt::factory::writeAtomic(dest, serialize(out), err))
        return fail(Status::Failed, "Couldn't write the file (" + err + ").");
    out.path = dest;
    return okResult(out);
}

OpResult Store::exportTo(const std::string& id, const fs::path& dest)
{
    OpResult r = loadById(id);
    if (!r.ok()) return r;
    return exportPreset(r.preset, dest);
}

OpResult Store::importFrom(const fs::path& src, ClashPolicy policy)
{
    Preset in;
    std::string reason;
    int skipped = 0;
    if (!readPresetFile(src, _cfg, true, in, reason, &skipped)) return fail(Status::Rejected, reason);
    if (_cfg.importCheck && !_cfg.importCheck(in, reason))
        return fail(Status::Rejected, reason.empty() ? std::string("This preset file can't be used here.") : reason);
    in.path.clear();

    std::lock_guard<std::mutex> lock(_mutex);
    ensureScannedLocked();
    in.plugin = in.plugin.empty() ? _cfg.pluginId : in.plugin;
    if (isReservedPresetName(in.name)) in.name = kDefaultName;
    const std::string now = ldt::factory::nowIso8601();
    if (in.createdAtUtc.empty()) in.createdAtUtc = now;
    in.updatedAtUtc = in.updatedAtUtc.empty() ? now : in.updatedAtUtc;

    const std::string key = normalizePresetNameKey(in.name);
    const Preset* byId = in.id.empty() ? nullptr : findCachedById(in.id);
    const bool sameRecord = byId && byId->family == in.family && byId->model == in.model &&
                            normalizePresetNameKey(byId->name) == key;
    if (sameRecord) {
        if (policy == ClashPolicy::Fail) {
            OpResult r = fail(Status::SameId, "You already have \"" + byId->name + "\". Replace it with this file?");
            r.preset = *byId;
            r.skipped = skipped;
            return r;
        }
        if (policy == ClashPolicy::Replace) {
            const Preset old = *byId;
            in.id = old.id;
            if (!old.createdAtUtc.empty()) in.createdAtUtc = old.createdAtUtc;
            OpResult r = commitOverLocked(old, in);
            r.skipped = skipped;
            return r;
        }
        in.id.clear();   // KeepBoth: becomes a new record below
    }
    // An id that already exists (or none at all) never reuses the old record.
    if (in.id.empty() || findCachedById(in.id)) {
        do { in.id = makePresetId(in.family); } while (findCachedById(in.id));
    }

    if (const Preset* clash = findCachedByName(in.family, in.model, normalizePresetNameKey(in.name), in.id)) {
        if (policy == ClashPolicy::Fail) {
            OpResult r = fail(Status::NameClash, "You already have a preset called \"" + clash->name + "\".");
            r.preset = *clash;
            r.skipped = skipped;
            return r;
        }
        if (policy == ClashPolicy::Replace) {
            const Preset old = *clash;
            in.id = old.id;
            if (!old.createdAtUtc.empty()) in.createdAtUtc = old.createdAtUtc;
            OpResult r = commitOverLocked(old, in);
            r.skipped = skipped;
            return r;
        }
        in.name = uniqueName(in.family, in.model, in.name);
    }
    OpResult r = writeNewLocked(in);
    r.skipped = skipped;
    return r;
}

// -- apply and match ----------------------------------------------------------
bool inFamily(const ParamDef& d, Family f)
{
    switch (f) {
        case Family::Full:      return (d.families & (kInLook | kInTonescale | kFullOnly)) != 0;
        case Family::Look:      return (d.families & kInLook) != 0;
        case Family::Tonescale: return (d.families & kInTonescale) != 0;
    }
    return false;
}

namespace {
double clampTo(const ParamDef& d, double v, bool& clamped)
{
    clamped = false;
    if (d.lo < d.hi) {
        if (v < d.lo) { v = d.lo; clamped = true; }
        if (v > d.hi) { v = d.hi; clamped = true; }
    }
    return v;
}
}

std::vector<Resolved> resolveForApply(const Preset& p, Family f, const std::vector<ParamDef>& table,
                                      const SkipFn& skip, ResolveReport* report)
{
    ResolveReport rep;
    std::vector<Resolved> out;
    for (const ParamDef& d : table) {
        if (!inFamily(d, f)) continue;
        if (skip && skip(d.id)) { ++rep.locked; continue; }
        Resolved r;
        r.id = d.id;
        r.type = d.type;
        r.value = d.def;
        auto it = p.values.find(d.id);
        if (it == p.values.end()) {
            ++rep.fromDefault;
            out.push_back(r);
            continue;
        }
        const Value& v = it->second;
        bool used = false;
        switch (d.type) {
            case ParamType::Double:
            case ParamType::Int:
                if (v.kind == Value::Kind::Number && std::isfinite(v.number)) {
                    double x = v.number;
                    if (d.type == ParamType::Int) x = (double)std::llround(x);
                    bool c = false;
                    x = clampTo(d, x, c);
                    if (c) ++rep.clamped;
                    r.value = x;
                    used = true;
                } else {
                    ++rep.badType;
                }
                break;
            case ParamType::Bool:
                if (v.kind == Value::Kind::Bool) { r.value = v.boolean ? 1.0 : 0.0; used = true; }
                else if (v.kind == Value::Kind::Number && std::isfinite(v.number)) {
                    r.value = (v.number != 0.0) ? 1.0 : 0.0; used = true;
                } else {
                    ++rep.badType;
                }
                break;
            case ParamType::Choice: {
                ldt::factory::ChoiceValue cv;
                bool have = false;
                if (v.kind == Value::Kind::Choice) { cv.label = v.label; cv.index = v.index; have = true; }
                else if (v.kind == Value::Kind::Number && std::isfinite(v.number) && v.number >= 0.0 &&
                         v.number < 100000.0) {
                    cv.index = (int)std::llround(v.number); have = true;
                }
                if (!have) { ++rep.badType; break; }
                const int sentinel = -1;
                const int idx = ldt::factory::resolveChoiceIndex(&cv, d.options, sentinel);
                if (idx == sentinel) { ++rep.badChoice; break; }
                r.value = (double)idx;
                used = true;
                break;
            }
        }
        if (used) { r.fromPreset = true; ++rep.fromPreset; }
        else ++rep.fromDefault;
        out.push_back(r);
    }
    for (const auto& kv : p.values) {
        bool known = false;
        for (const ParamDef& d : table) if (d.id == kv.first) { known = inFamily(d, f); break; }
        if (!known) ++rep.skippedUnknown;
    }
    if (report) *report = rep;
    return out;
}

bool matchesPreset(const Preset& p, Family f, const std::vector<ParamDef>& table,
                   const std::map<std::string, double>& live, const SkipFn& skip, double relTol)
{
    const std::vector<Resolved> want = resolveForApply(p, f, table, skip, nullptr);
    for (const Resolved& r : want) {
        auto it = live.find(r.id);
        if (it == live.end()) return false;
        const double a = it->second, b = r.value;
        if (r.type == ParamType::Double) {
            if (!std::isfinite(a)) return false;
            if (std::fabs(a - b) > relTol * std::max(1.0, std::fabs(b))) return false;
        } else {
            if (std::llround(a) != std::llround(b)) return false;
        }
    }
    return true;
}

Preset capturePreset(const std::string& name, Family f, const std::string& model,
                     const std::vector<ParamDef>& table, const std::map<std::string, double>& live)
{
    Preset p;
    p.family = f;
    p.model = model.empty() ? std::string(kDefaultModel) : model;
    p.name = name;
    for (const ParamDef& d : table) {
        if (!inFamily(d, f)) continue;
        auto it = live.find(d.id);
        if (it == live.end() || !std::isfinite(it->second)) continue;
        const double v = it->second;
        switch (d.type) {
            case ParamType::Double: p.values[d.id] = Value::num(v); break;
            case ParamType::Int:    p.values[d.id] = Value::num((double)std::llround(v)); break;
            case ParamType::Bool:   p.values[d.id] = Value::flag(v != 0.0); break;
            case ParamType::Choice: {
                const long long i = std::llround(v);
                const std::string label = (i >= 0 && i < (long long)d.options.size()) ? d.options[(size_t)i]
                                                                                       : std::string();
                p.values[d.id] = Value::choice(label, (int)i);
                break;
            }
        }
    }
    return p;
}

// -- menu helpers -------------------------------------------------------------
std::string menuLabel(const Preset& p) { return std::string(familyMenuPrefix(p.family)) + p.name; }

std::string encodeSelectedRef(const SelectedRef& r)
{
    if (r.empty()) return std::string();
    std::string name = r.name;
    for (char& c : name) if (c == '|') c = ' ';
    return std::string(familyKey(r.family)) + "|" + r.model + "|" + r.id + "|" + name;
}

bool decodeSelectedRef(const std::string& s, SelectedRef& out)
{
    out = SelectedRef();
    if (s.empty()) return false;
    const size_t a = s.find('|');
    if (a == std::string::npos) return false;
    const size_t b = s.find('|', a + 1);
    if (b == std::string::npos) return false;
    const size_t c = s.find('|', b + 1);
    if (c == std::string::npos) return false;
    SelectedRef r;
    if (!familyFromKey(s.substr(0, a), r.family)) return false;
    r.model = s.substr(a + 1, b - a - 1);
    if (r.model.empty()) r.model = kDefaultModel;
    r.id = s.substr(b + 1, c - b - 1);
    r.name = s.substr(c + 1);
    if (r.empty()) return false;
    out = r;
    return true;
}

int resolveSelectedRef(const std::vector<Preset>& entries, const SelectedRef& r)
{
    if (r.empty()) return -1;
    const std::string model = r.model.empty() ? std::string(kDefaultModel) : r.model;
    if (!r.id.empty())
        for (size_t i = 0; i < entries.size(); ++i)
            if (entries[i].family == r.family && entries[i].model == model && entries[i].id == r.id) return (int)i;
    const std::string key = normalizePresetNameKey(r.name);
    if (!key.empty())
        for (size_t i = 0; i < entries.size(); ++i)
            if (entries[i].family == r.family && entries[i].model == model &&
                normalizePresetNameKey(entries[i].name) == key)
                return (int)i;
    return -1;
}

} // namespace presets
} // namespace ldt
