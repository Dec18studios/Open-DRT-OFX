#pragma once
// LdtPresetStore.h - host-free user preset store for 6A OpenDRT.
//
// Design: docs/design/OPENDRT-PRESET-MANAGER-DESIGN.md (sections 3.1, 3.3, 3.4,
// 3.6, 3.7, 3.8, 3.11). No OFX includes: the selftest
// (tools/control/ldt_preset_selftest.cpp) builds this with plain clang++.
//
// One JSON file per preset:
//   {
//     "format": "LookDevToolsOpenDRTPreset", "version": 1,
//     "plugin": "com.dec18studios.lookdev.openDRT",
//     "family": "look", "model": "default", "name": "Warm Night",
//     "id": "look_1763164800_123456_1",
//     "createdAtUtc": "...", "updatedAtUtc": "...",
//     "values": { "_pt_r": 0.25, "_ptl_enable": true,
//                 "look_preset": { "label": "Umbra", "index": 2 } }
//   }
// Choices are written as {label, index}; on read a bare string is a label and
// a bare number is an index (the resolve step decides by the param's type).
//
// Store directory (shared by the standalone and in-suite OpenDRT):
//   macOS:   ~/Library/Application Support/LookDevTools/OpenDRT/presets
//   Windows: %APPDATA%\LookDevTools\OpenDRT\presets
//   Linux:   ${XDG_CONFIG_HOME:-~/.config}/LookDevTools/OpenDRT/presets
// The root is the same LookDevTools root Factory Settings uses.
//
// Safety rules kept by every write:
//   - writes are atomic (unique temp in the same folder, then rename);
//   - delete renames to .bak, overwrite keeps a .bak copy of the previous
//     file; only the newest kMaxBackupsPerPreset backups of a file are kept,
//     older ones are pruned (the only files the store ever unlinks);
//   - a file whose model this build does not know is never rewritten, renamed
//     or retired, and only shows up when listing that exact model key.

#include <cstddef>
#include <filesystem>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ldt {
namespace presets {

static const char* const kPresetFormat     = "LookDevToolsOpenDRTPreset";
static const int         kPresetVersion    = 1;
static const char* const kDefaultModel     = "default";
static const char* const kDefaultName      = "My Preset";
static const char* const kCustomSuffix     = " (Custom)";
static const char* const kMissingSuffix    = " (missing)";
static const char* const kSuitePluginId    = "com.dec18studios.lookdev.openDRT";
static const char* const kStandalonePluginId = "com.dec18studios.OpenDRT";
static const std::size_t kMaxPresetFileBytes = 4u * 1024u * 1024u;
static const std::size_t kMaxNameBytes     = 96;
static const int         kMaxBackupsPerPreset = 3;

// Index order matches the planned preset_manager_family choice: append-only.
enum class Family { Full = 0, Look = 1, Tonescale = 2 };

const char* familyKey(Family f);                          // "full" / "look" / "tonescale"
bool familyFromKey(const std::string& key, Family& out);
const char* familyMenuPrefix(Family f);                   // "" / "Look: " / "Tone: "

// One stored value. The resolve step interprets it against the param type.
struct Value {
    enum class Kind { Number, Bool, Choice };
    Kind        kind = Kind::Number;
    double      number = 0.0;   // Number
    bool        boolean = false;// Bool
    std::string label;          // Choice: option label, resolved first
    int         index = -1;     // Choice: fallback when the label is not found

    static Value num(double v)   { Value x; x.kind = Kind::Number; x.number = v; return x; }
    static Value flag(bool b)    { Value x; x.kind = Kind::Bool; x.boolean = b; return x; }
    static Value choice(const std::string& l, int i) { Value x; x.kind = Kind::Choice; x.label = l; x.index = i; return x; }
};

struct Preset {
    Family      family = Family::Full;
    std::string model = kDefaultModel;
    std::string name;
    std::string id;
    std::string plugin;
    std::string createdAtUtc;
    std::string updatedAtUtc;
    std::map<std::string, Value> values;
    std::filesystem::path path;   // set when the preset lives in a store
    std::string sourceName;       // raw "name" as read, before sanitizing; never written
};

// -- names and ids ------------------------------------------------------------
// Strip control characters and path characters (/ \ : * ? " < > |), trim,
// collapse spaces, drop leading dots, strip trailing " (Custom)", " (missing)"
// and " (<model key>)" for any key in knownModels, cut at 96 bytes on a UTF-8
// boundary. Empty result becomes "My Preset".
std::string sanitizePresetName(const std::string& raw,
                               const std::vector<std::string>& knownModels = {});
// Lowercase, single spaces. Used for every name comparison.
std::string normalizePresetNameKey(const std::string& name);
// [a-z0-9_] only, at most 64 chars, "preset" when nothing is left.
std::string sanitizeFileComponent(const std::string& raw);
// "<family>_<secs>_<micros>_<counter>".
std::string makePresetId(Family f);
// "None" (the menu's empty entry) and "Factory" are reserved. Menu suffixes
// are already stripped by sanitizePresetName.
bool isReservedPresetName(const std::string& sanitizedName);

// -- format -------------------------------------------------------------------
struct StoreConfig {
    std::string pluginId = kSuitePluginId;   // written into new files
    std::vector<std::string> acceptedPluginIds { kSuitePluginId, kStandalonePluginId };
    std::vector<std::string> knownModels { kDefaultModel };
    // format string -> user message for files from other products
    std::map<std::string, std::string> foreignFormats;
    std::size_t maxFileBytes = kMaxPresetFileBytes;
    // Optional extra import validation, run after a successful parse and
    // before anything is written. False rejects the file with `reason`.
    std::function<bool(const Preset&, std::string& reason)> importCheck;
};
// The OpenDRT config (both plugin ids, model "default", foreign format messages).
StoreConfig openDrtConfig();

// Locale-independent shortest round-trip number, sorted keys, 2-space indent.
std::string serialize(const Preset& p);

// Strict parse. Returns false with a short user-facing reason on any reject.
// requireKnownModel=false accepts any model key (used when scanning the store,
// so unknown-model files are kept aside instead of dropped).
// skippedOut counts values dropped for a bad type (null, string, array).
bool parse(const std::string& text, const StoreConfig& cfg, bool requireKnownModel,
           Preset& out, std::string& reason, int* skippedOut = nullptr);

// Read a file with the size cap, strip a UTF-8 BOM, then parse. Never uses the
// file name for anything.
bool readPresetFile(const std::filesystem::path& file, const StoreConfig& cfg,
                    bool requireKnownModel, Preset& out, std::string& reason,
                    int* skippedOut = nullptr);

// -- directory ----------------------------------------------------------------
// <LookDevTools root>/OpenDRT/presets, or the override when one is set.
std::filesystem::path presetDirectory();
// Test hook. Set before the first sharedOpenDrtStore() call. Empty clears it.
void setPresetDirectoryOverride(const std::filesystem::path& dir);

// -- backups ------------------------------------------------------------------
// Keep the newest `keep` of <file>.bak / <file>.bak.N (justMade first, then by
// mtime, then by N) and remove the rest. Returns how many were removed.
int pruneBackups(const std::filesystem::path& file, int keep = kMaxBackupsPerPreset,
                 const std::filesystem::path& justMade = std::filesystem::path());

// -- store --------------------------------------------------------------------
enum class Status {
    Ok,
    NameClash,   // same family+model+normalized name, different id
    SameId,      // import: same id already in the store
    NotFound,
    Rejected,    // validation failed (reason says why)
    ReadOnly,    // record belongs to a model this build does not know
    Failed       // IO failure (reason says why)
};

enum class ClashPolicy {
    Fail,        // report NameClash / SameId and do nothing
    KeepBoth,    // suffix " 2", " 3", ... and mint a fresh id if needed
    Replace      // overwrite the clashing record, keeping its id
};

struct OpResult {
    Status      status = Status::Failed;
    std::string reason;    // short user-facing text when not Ok
    Preset      preset;    // the stored record on Ok, the clashing record on a clash
    int         skipped = 0; // import: values dropped for a bad type
    bool ok() const { return status == Status::Ok; }
};

class Store {
public:
    Store(std::filesystem::path dir, StoreConfig cfg);

    const std::filesystem::path& directory() const { return _dir; }
    const StoreConfig& config() const { return _cfg; }

    // Re-scan the folder. Cheap; call before a manager action (lazy refresh).
    void reload();

    // Presets of one model, sorted by family (Full, Look, Tonescale), then
    // normalized name, then id. A missing "model" key reads as "default".
    std::vector<Preset> list(const std::string& model) const;
    std::vector<Preset> list(const std::string& model, Family f) const;

    // Fresh read from disk. False (NotFound) when the file vanished since the
    // last scan; the cache drops it.
    OpResult loadById(const std::string& id);
    OpResult loadByName(Family f, const std::string& model, const std::string& name);

    // New record: sanitized name, fresh id, timestamps, unique file name. Never
    // overwrites a different preset unless policy is Replace.
    OpResult saveNew(Preset p, ClashPolicy policy = ClashPolicy::Fail);
    // New values for an existing record. Keeps id, name, family, model, createdAt.
    OpResult update(const std::string& id, const std::map<std::string, Value>& values);
    // Keeps the id. Writes the new file first, then retires the old one to .bak.
    OpResult rename(const std::string& id, const std::string& newName);
    // Rename to .bak (then prune that file's backups to the newest 3).
    OpResult remove(const std::string& id);

    // Write a stored record (or any preset) to an arbitrary path, atomically.
    OpResult exportTo(const std::string& id, const std::filesystem::path& dest);
    static OpResult exportPreset(const Preset& p, const std::filesystem::path& dest);

    // Validate a file and add it. Id clash with a different name: fresh id.
    // Same id and same name: SameId unless Replace/KeepBoth. Name clash:
    // NameClash unless Replace/KeepBoth.
    OpResult importFrom(const std::filesystem::path& src, ClashPolicy policy = ClashPolicy::Fail);

private:
    bool isKnownModel(const std::string& m) const;
    std::filesystem::path pathFor(const Preset& p) const;
    const Preset* findCachedById(const std::string& id) const;
    const Preset* findCachedByName(Family f, const std::string& model, const std::string& key,
                                   const std::string& exceptId) const;
    std::string uniqueName(Family f, const std::string& model, const std::string& base) const;
    void scanLocked() const;
    void ensureScannedLocked() const;
    OpResult writeNewLocked(Preset p);
    OpResult commitOverLocked(const Preset& existing, Preset incoming);

    std::filesystem::path _dir;
    StoreConfig _cfg;
    mutable std::mutex _mutex;
    mutable std::vector<Preset> _cache;
    mutable bool _scanned = false;
};

// Process-wide OpenDRT store at presetDirectory(), created on first use.
Store& sharedOpenDrtStore();

// -- apply and match (pure, no IO) --------------------------------------------
enum class ParamType { Double, Int, Bool, Choice };

// Family membership bits. Full covers every bit; kFullOnly marks the beta block.
static const unsigned kInLook      = 1u;
static const unsigned kInTonescale = 2u;
static const unsigned kFullOnly    = 4u;

struct ParamDef {
    std::string id;
    ParamType   type = ParamType::Double;
    double      def = 0.0;                // bool 0/1, choice index
    std::vector<std::string> options;     // choice labels in current order
    unsigned    families = 0;             // kInLook | kInTonescale | kFullOnly
    double      lo = 0.0, hi = 0.0;       // hard range; lo >= hi means no clamp
};

bool inFamily(const ParamDef& d, Family f);

struct Resolved {
    std::string id;
    ParamType   type = ParamType::Double;
    double      value = 0.0;   // bool 0/1, choice index, int rounded
    bool        fromPreset = false;
};

struct ResolveReport {
    int fromPreset = 0;
    int fromDefault = 0;      // missing key, or a value that could not be used
    int skippedUnknown = 0;   // preset keys not in the table or not in the family
    int badChoice = 0;        // label and index both failed
    int badType = 0;          // wrong JSON kind for the param
    int clamped = 0;
    int locked = 0;           // in family but excluded by the skip predicate
};

// Return true to leave a param untouched (a locked module).
typedef std::function<bool(const std::string& id)> SkipFn;

// Every in-family param not skipped, in table order. A key the preset does
// not store gets the table default (design 3.3 step 5). Out-of-family and
// skipped params are left out, so the caller never touches them.
std::vector<Resolved> resolveForApply(const Preset& p, Family f, const std::vector<ParamDef>& table,
                                      const SkipFn& skip = SkipFn(), ResolveReport* report = nullptr);

// Live values by id (bool 0/1, choice index). True when every param that
// resolveForApply would write equals the live value: doubles within
// |a-b| <= relTol * max(1, |b|), everything else exact. A missing live value
// is a mismatch.
bool matchesPreset(const Preset& p, Family f, const std::vector<ParamDef>& table,
                   const std::map<std::string, double>& live, const SkipFn& skip = SkipFn(),
                   double relTol = 1e-6);

// Build a preset from live values: every in-family param of the table,
// choices stored as {label, index}, non-finite numbers left out.
Preset capturePreset(const std::string& name, Family f, const std::string& model,
                     const std::vector<ParamDef>& table, const std::map<std::string, double>& live);

// -- menu helpers -------------------------------------------------------------
// "<prefix><name>" for the my_preset menu.
std::string menuLabel(const Preset& p);
// label + " (missing)" when missing, else label + " (Custom)" when custom.
std::string decorateLabel(const std::string& label, bool custom, bool missing);

// Hidden preset_manager_selected string: "family|model|id|name".
struct SelectedRef {
    Family family = Family::Full;
    std::string model, id, name;
    bool empty() const { return id.empty() && name.empty(); }
};
std::string encodeSelectedRef(const SelectedRef& r);
bool decodeSelectedRef(const std::string& s, SelectedRef& out);
// Index into entries matching by family+id, then family+normalized name. -1 when none.
int resolveSelectedRef(const std::vector<Preset>& entries, const SelectedRef& r);

} // namespace presets
} // namespace ldt
