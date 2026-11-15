#pragma once
// LdtFactorySettings.h - user starting points ("Factory Settings") for every
// Look Dev Tools node.
//
// Two layers:
//   Layer A (always compiled, no OFX includes): the on-disk store. Slots are
//   small JSON files under a per-tool directory. Host-free so the selftest can
//   exercise it without a host.
//   Layer B (OFX, skipped when LDT_FACTORY_NO_OFX is defined): the descriptor
//   side (describe the Factory Settings group, build the per-plugin param
//   registry) and an instance Controller that each tool owns.
//
// Store layout (one directory per store key):
//   macOS:   ~/Library/Application Support/LookDevTools/starting-points/<key>/
//   Windows: %APPDATA%\LookDevTools\starting-points\<key>\
//   Linux:   ${XDG_CONFIG_HOME:-~/.config}/LookDevTools/starting-points/<key>/
// Each starting point is <stem>.json. default.json holds the name of the slot
// new nodes open with (empty name = Factory). Deleting renames to .bak.

#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace ldt {
namespace factory {

// ── Layer A: host-free store ──────────────────────────────────────────────────

static const char* const kFormatName   = "LookDevToolsStartingPoint";
static const int         kSchemaMajor  = 1;
static const char* const kFactoryLabel = "Factory";

struct ChoiceValue {
    std::string label;      // option label, resolved first
    int         index = -1; // fallback only when the label is not found
};

struct Slot {
    std::string name;
    std::filesystem::path path;
    std::string savedAt;
    std::string pluginId;
    std::string pluginVersion;
    std::map<std::string, std::vector<double>> values;  // numeric / bool params
    std::map<std::string, ChoiceValue>         choices; // choice params
    std::map<std::string, std::string>         strings; // string params
};

// Root of every LookDevTools store on this machine (platform specific).
std::filesystem::path baseDirectory();
// <baseDirectory>/<storeKey>. Does not create it.
std::filesystem::path storeDirectory(const std::string& storeKey);

// Trim, collapse whitespace, drop control and path-hostile characters, cap
// length. Returns "" when nothing usable is left.
std::string sanitizeName(const std::string& raw);
// True when the name is the reserved built-in label (case-insensitive).
bool isReservedName(const std::string& name);
// Case-insensitive equality used for slot names.
bool sameName(const std::string& a, const std::string& b);

// All valid slots in dir, sorted by name. Cached per directory; refresh=true
// (or invalidate()) re-reads the folder.
std::vector<Slot> listSlots(const std::filesystem::path& dir, bool refresh = false);
void invalidate(const std::filesystem::path& dir);
bool findSlot(const std::filesystem::path& dir, const std::string& name, Slot& out,
              bool refresh = false);
// dir/<stem>.json, or dir/<stem>-N.json when taken. Does not create the file.
std::filesystem::path uniquePath(const std::filesystem::path& dir, const std::string& name);

std::string serialize(const Slot& slot);
// Rejects a wrong format, a wrong kind or a different schema major. Unknown
// keys are ignored.
bool parse(const std::string& text, Slot& out, std::string& err);

// Write text to path atomically: unique temp file in the same directory, then
// rename over the target. Creates the directory if needed.
bool writeAtomic(const std::filesystem::path& path, const std::string& text, std::string& err);
bool writeSlot(const Slot& slot, std::string& err); // uses slot.path

// Active default pointer ("" = Factory). A missing or unreadable file reads as "".
std::string readDefaultName(const std::filesystem::path& dir);
bool writeDefaultName(const std::filesystem::path& dir, const std::string& name, std::string& err);

// Rename path to path.bak (or .bak.N when taken). Never unlinks.
bool renameToBackup(const std::filesystem::path& path, std::filesystem::path& backupOut,
                    std::string& err);

std::string nowIso8601();

// Value resolution shared by every apply (host-free so the selftest covers it).
// Choice: the stored label wins; the stored index is used only when the label
// is not among the current options; otherwise the default index.
int resolveChoiceIndex(const ChoiceValue* stored, const std::vector<std::string>& options,
                       int defaultIndex);
// Numeric: each component from the stored value when present and finite,
// otherwise from the default. stored == nullptr returns the default.
std::vector<double> resolveNumeric(const std::vector<double>* stored, const std::vector<double>& def);

} // namespace factory
} // namespace ldt

// ── Layer B: OFX ──────────────────────────────────────────────────────────────
#ifndef LDT_FACTORY_NO_OFX

#include "ofxsImageEffect.h"

namespace ldt {
namespace factory {

// Hidden persistent flag, the only persistent param this feature adds.
static const char* const kInitFlagParam = "ldt_factory_initialized";

struct Options {
    std::string pluginId;               // the OFX plugin id (factory getID())
    std::string storeKey;               // folder name; plugins sharing a param set may share it
    std::string toolName;               // used in messages ("New <toolName> nodes ...")
    std::string pluginVersion = "1.0";  // recorded in saved files
    std::vector<std::string> excludeIds;      // exact ids never captured
    std::vector<std::string> excludePrefixes; // id prefixes never captured
};

// Define the init flag and the closed "Factory Settings" group, then build the
// registry for options.pluginId from everything defined on desc so far. Call
// LAST in describeInContext, after describeUuidParam. Idempotent per plugin id.
void describe(OFX::ImageEffectDescriptor& desc, OFX::PageParamDescriptor* page,
              const Options& options);

// Captured param ids for a plugin (empty when describe has not run).
std::vector<std::string> registeredParams(const std::string& pluginId);

class Controller {
public:
    typedef std::function<void(double time)> AfterApplyFn;

    // Construct in the tool ctor after every fetch, BEFORE the control channel
    // bind (bind mints the uuid, which the new-node test reads).
    Controller(OFX::ImageEffect& effect, const std::string& pluginId,
               AfterApplyFn afterApply = AfterApplyFn());
    ~Controller();

    // Fill the non-persistent menu, and give a brand-new node the default
    // starting point. Writes nothing persistent on an existing node.
    void onConstructed();

    // Call FIRST in the tool's changedParam; return when it returns true.
    // True for every factory_* param, and for any param while an apply is
    // running (so tool logic does not react to half-applied state).
    bool handleChangedParam(const OFX::InstanceChangedArgs& args, const std::string& name);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};

} // namespace factory
} // namespace ldt

#endif // LDT_FACTORY_NO_OFX
