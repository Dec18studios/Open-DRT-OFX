#pragma once
//
// LdtControl.h — file-based control plane shared by every Look Dev Tools factory.
//
// Root:      ~/.ofx-lookdevtools/          (created 0700 on demand)
// Sentinel:  ~/.ofx-lookdevtools/armed     (plain file)
//
// If the sentinel does NOT exist, every plugin in the suite does ZERO file IO
// and starts ZERO threads: each tool behaves as an ordinary param-driven node.
// Nothing in this header touches the filesystem until armed() returns true,
// except armed() itself (one stat per render).
//
// Every write is atomic: the payload is written to "<name>.tmp" in the same
// directory and then rename()d over the destination, so a reader never observes
// a half-written file.
//
// Ported from the BalanceX LGGO control plane (BalanceXIPC + BalanceXLGGO),
// generalised so a tool integrates by declaring a whitelist table and owning
// one Channel.
//
// Windows: everything compiles, but the channel is STUBBED — armed() always
// returns false, so no thread ever starts and no file is ever written. The
// hidden uuid param is still defined and bound so project files round-trip
// identically across platforms. Linux gets the full POSIX path.
//
#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstdio>   // snprintf, used by the inline formatters below
#include <cmath>    // llround, used by the inline decoder below

#ifndef LDT_CONTROL_SELFTEST
  #include <thread>
  #include <mutex>
  #include <atomic>
#endif

#include "ldt_json.h"

// Forward declarations only: this header never pulls in the OFX SDK, so the
// pure decode/format logic below can be unit-tested host-free.
namespace OFX {
    class ImageEffect;
    class ImageEffectDescriptor;
    class PageParamDescriptor;
    class DoubleParam;
    class IntParam;
    class BooleanParam;
    class ChoiceParam;
    class StringParam;
}

namespace ldt {
namespace control {

// ── Paths ────────────────────────────────────────────────────────────────────

// "~/.ofx-lookdevtools".  Empty string if HOME cannot be resolved.
const std::string& rootDir();

// "<dir>/<leaf>"; leaf may contain '/' separators.
std::string joinPath(const std::string& dir, const std::string& leaf);

// mkdir -p with mode 0700 for every created component.  Returns true on success
// (or if the directory already exists).  Always false on Windows.
bool ensureDir(const std::string& path);

// ── Sentinel ─────────────────────────────────────────────────────────────────

// One stat() of "<root>/armed".  Callers must invoke this at most once per
// render and cache the result for the rest of that call.  Always false on
// Windows (the channel is stubbed there).
bool armed();

// ── Atomic IO ────────────────────────────────────────────────────────────────

// Write `len` bytes to `path` atomically (tmp 0600 + rename).  Never throws.
bool atomicWrite(const std::string& path, const void* data, size_t len);

// Convenience overload for text payloads.
bool atomicWriteText(const std::string& path, const std::string& text);

// Read a whole file into `out`.  Returns false if it cannot be read.
bool readFile(const std::string& path, std::string& out);

// mtime in nanoseconds, or 0 when the file does not exist / cannot be stat'ed.
uint64_t mtimeNanos(const std::string& path);

// Update (or create) a zero-length file's mtime.  Cheap: utimes, falling back
// to a create-and-close only when the file is missing.
bool touchFile(const std::string& path);

// ── Identity ─────────────────────────────────────────────────────────────────

// A fresh lowercase RFC-4122 UUID string.
std::string generateUuid();

// Process-global set of live instance UUIDs (Alt-drag clone detection).
// claimUuid() returns false when the UUID is already held by another live
// instance in this process; the caller then generates a fresh one and retries.
bool claimUuid(const std::string& uuid);
void releaseUuid(const std::string& uuid);

// Unix seconds as a double (matches the "unix_time" / "time" JSON fields).
double unixTimeNow();

// ── Tap publishing ───────────────────────────────────────────────────────────

// Box-downscale a BOTTOM-UP float32 RGBA buffer to <= 512 px wide, flip it to
// top-down, and atomically write "<root>/tap/frame_<slot>.bin" followed by its
// ".meta" sidecar (always bin before meta).  The publish seq counter is
// process-global and shared across every slot and every caller, so a reader's
// min-seq freshness gate spans every publishing node alike.  Never throws;
// returns false on any failure.  Callers gate on armed() themselves.
bool publishTapFrame(const float* srcBottomUp, int width, int height,
                     double time, int slot);

// ── Whitelist ────────────────────────────────────────────────────────────────

// One controllable parameter.  Tables are static const, one per tool, living
// next to that tool's kXX_* script-name constants so there is no string
// duplication and the clamp range matches the descriptor's setRange().
//
// JSON key vs OFX name:
//   the grade.json key is ALWAYS `scriptName` verbatim;
//   the fetched OFX param name is ALWAYS `prefix + scriptName`.
// Standalone tools pass an empty prefix, so the two are identical.  The
// monolith pre-builds prefixed scriptName strings (e.g. "m2D_redSat") that
// outlive the Channel and passes an empty prefix, so its JSON keys are the
// prefixed names.
struct ParamSpec {
    const char* scriptName;                       // OFX param name WITHOUT any prefix
    enum Kind { Double, Int, Bool, Choice } kind;
    double      minV;                             // clamp range (ignored for Bool)
    double      maxV;
};

// ── Pure decode / format logic (host-free, unit-tested) ──────────────────────
//
// Defined inline here rather than in LdtControl.cpp so tools/control/
// ldt_control_selftest.cpp can exercise them with no OFX SDK on the include
// path (build it with -DLDT_CONTROL_SELFTEST, which drops the Channel below).
namespace detail {

// One validated assignment.  `value` is already clamped and, for Bool, is
// exactly 0.0 or 1.0.
struct DecodedValue {
    size_t specIndex;
    double value;
};

struct DecodeResult {
    bool                      ok    = false;
    long long                 seq   = -1;   // -1 when "seq" could not be read
    std::string               error;        // non-empty exactly when ok == false
    std::vector<DecodedValue> values;       // empty for a state read
};

inline double clampDouble(double v, double lo, double hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// Validate EVERY key in doc["params"] against the whitelist and decode ALL
// values before the caller touches a single param: an unknown key, a wrong
// type, or a param the node does not actually carry rejects the whole file and
// nothing is applied.
//
// `available` is an optional one-byte-per-spec mask (nullptr = every spec is
// available) matching Channel's bind() result.
//
// A missing or empty "params" object is a state read: ok == true with an empty
// `values`, which the caller answers by rewriting applied.json.
inline DecodeResult decodeGrade(const ldtjson::Value& doc,
                                const ParamSpec* specs, size_t count,
                                const unsigned char* available)
{
    DecodeResult res;

    if (!doc.isObject()) {
        res.error = "grade.json root is not an object";
        return res;
    }

    const ldtjson::Value* seqV = doc.find("seq");
    long long seq = 0;
    if (!seqV || !seqV->isNumber() || !seqV->asInt(seq)) {
        res.error = "grade.json is missing a numeric \"seq\"";
        return res;
    }
    res.seq = seq;

    // Absent, null, or empty "params" = state read: nothing to apply, the
    // caller still rewrites applied.json so the controller learns the state.
    const ldtjson::Value* paramsV = doc.find("params");
    if (!paramsV || paramsV->isNull()) {
        res.ok = true;
        return res;
    }
    if (!paramsV->isObject()) {
        res.error = "\"params\" must be an object";
        return res;
    }

    for (ldtjson::Object::const_iterator it = paramsV->obj.begin();
         it != paramsV->obj.end(); ++it) {
        const std::string&    key = it->first;
        const ldtjson::Value& v   = it->second;

        size_t idx = count;
        for (size_t i = 0; i < count; ++i) {
            if (specs[i].scriptName && key == specs[i].scriptName) { idx = i; break; }
        }
        if (idx == count) {
            res.values.clear();
            res.error = "unknown param \"" + key + "\"";
            return res;
        }
        if (available && !available[idx]) {
            res.values.clear();
            res.error = "param \"" + key + "\" is not available on this node";
            return res;
        }

        const ParamSpec& s = specs[idx];
        double out = 0.0;

        if (s.kind == ParamSpec::Bool) {
            if (v.isBool()) {
                out = v.boolean ? 1.0 : 0.0;
            } else if (v.isNumber() && (v.number == 0.0 || v.number == 1.0)) {
                out = v.number;
            } else {
                res.values.clear();
                res.error = "param \"" + key + "\" expects true/false or 0/1";
                return res;
            }
        } else {
            if (!v.isNumber()) {
                res.values.clear();
                res.error = "param \"" + key + "\" expects a number";
                return res;
            }
            if (s.kind == ParamSpec::Double) {
                out = clampDouble(v.number, s.minV, s.maxV);
            } else {
                // Int / Choice: round to the nearest integer, then clamp to the
                // integer range the descriptor declared.
                const long long lo = (long long)llround(s.minV);
                const long long hi = (long long)llround(s.maxV);
                long long iv = (long long)llround(v.number);
                if (iv < lo) iv = lo;
                if (iv > hi) iv = hi;
                out = (double)iv;
            }
        }

        DecodedValue dv;
        dv.specIndex = idx;
        dv.value     = out;
        res.values.push_back(dv);
    }

    res.ok = true;
    return res;
}

// applied.json body: EVERY bound spec's current value, not just what was sent.
// That full read-back is what lets a controller learn a node's whole state from
// one `{"seq":1,"params":{}}` write.  Doubles print %.9g, Int/Choice print as
// integers, Bool prints true/false.  Built into a std::string rather than a
// fixed snprintf buffer because a whitelist can run to 60+ entries.
inline std::string formatApplied(const char* toolId, long long seq, double unixTime,
                                 const ParamSpec* specs, size_t count,
                                 const double* values, const unsigned char* available)
{
    char nbuf[64];
    std::string out;
    out.reserve(64 + count * 32);

    snprintf(nbuf, sizeof(nbuf), "%lld", seq);
    out += "{\"seq\":";
    out += nbuf;
    snprintf(nbuf, sizeof(nbuf), "%.6f", unixTime);
    out += ",\"time\":";
    out += nbuf;
    out += ",\"tool\":\"";
    out += ldtjson::escape(toolId ? toolId : "");
    out += "\",\"values\":{";

    bool first = true;
    for (size_t i = 0; i < count; ++i) {
        if (available && !available[i]) continue;
        if (!first) out += ",";
        first = false;

        out += "\"";
        out += ldtjson::escape(specs[i].scriptName ? specs[i].scriptName : "");
        out += "\":";

        const double v = values ? values[i] : 0.0;
        switch (specs[i].kind) {
            case ParamSpec::Bool:
                out += (v != 0.0) ? "true" : "false";
                break;
            case ParamSpec::Int:
            case ParamSpec::Choice:
                snprintf(nbuf, sizeof(nbuf), "%lld", (long long)llround(v));
                out += nbuf;
                break;
            default:
                snprintf(nbuf, sizeof(nbuf), "%.9g", v);
                out += nbuf;
                break;
        }
    }

    out += "}}\n";
    return out;
}

// error.json body.  Nothing was applied when this is written.
inline std::string formatError(const char* toolId, long long seq, double unixTime,
                               const std::string& message)
{
    char nbuf[64];
    std::string out;

    snprintf(nbuf, sizeof(nbuf), "%lld", seq);
    out += "{\"seq\":";
    out += nbuf;
    snprintf(nbuf, sizeof(nbuf), "%.6f", unixTime);
    out += ",\"time\":";
    out += nbuf;
    out += ",\"tool\":\"";
    out += ldtjson::escape(toolId ? toolId : "");
    out += "\",\"message\":\"";
    out += ldtjson::escape(message);
    out += "\"}\n";
    return out;
}

} // namespace detail

#ifndef LDT_CONTROL_SELFTEST

// ── Channel ──────────────────────────────────────────────────────────────────
//
// One per plugin instance.  Owns the hidden uuid param, the poll thread, and
// the whole grade.json -> params -> applied.json lifecycle.  Integration is
// five lines in the tool:
//
//   describeInContext():  ldt::control::Channel::describeUuidParam(desc, page);
//   constructor (last):   _control.reset(new Channel(*this, "4B", kSpecs, N));
//                         _control->bind();
//   render() (first):     if (_control) _control->onRender();
//   destructor:           if (_control) _control->shutdown();
//
// No method of this class ever lets an exception escape.
class Channel
{
public:
    // Defines the hidden persistent uuid StringParam "ldt_instance_uuid".
    // setIsSecret / setIsPersistant / setAnimates / setEvaluateOnChange MUST be
    // called INSIDE the define block BEFORE page->addChild — a post-pass over
    // the finished descriptor is a silent no-op on Resolve.  Callers define it
    // LAST so existing param order never shifts (OFX restore order matters).
    static void describeUuidParam(OFX::ImageEffectDescriptor& desc,
                                  OFX::PageParamDescriptor* page);

    // toolId: short tag echoed in applied.json / error.json ("4B", "1F", "6A",
    //         "MONO", ...).  Must outlive the Channel (a string literal).
    // specs/count: the whitelist table; must outlive the Channel.
    // prefix: prepended to every scriptName when FETCHING (monolith children
    //         use "m2D_" etc).  The JSON key stays the unprefixed scriptName.
    // publishInputWhenArmed: v1 hook for tap publishing; see publishInput().
    Channel(OFX::ImageEffect& effect, const char* toolId,
            const ParamSpec* specs, size_t count,
            const std::string& prefix = std::string(),
            bool publishInputWhenArmed = false);

    ~Channel();   // calls shutdown()

    Channel(const Channel&)            = delete;
    Channel& operator=(const Channel&) = delete;

    // Call at the END of the plugin constructor, after its own fetches.
    // Fetches ldt_instance_uuid, establishes the uuid (claim the persisted one,
    // or mint a fresh one in <= 8 attempts, setValue in try/catch), then
    // fetches every spec by prefix+scriptName with the matching fetch*Param.
    // A param the host does not carry is marked unavailable and skipped, never
    // thrown.
    void bind();

    // First line of render().  Cheap when disarmed: exactly one stat.
    void onRender();

    // Optional: publish this node's input as tap slot `slot`.  No-op unless
    // armed AND the channel was constructed with publishInputWhenArmed.
    void publishInput(const float* srcBottomUp, int w, int h,
                      double time, int slot = 0);

    // Stop + join the poll thread and release the uuid claim.  Idempotent;
    // called from ~Channel and safe to call explicitly from the plugin's
    // destructor.
    void shutdown();

    const std::string& instanceUuid() const { return m_InstanceUuid; }

    // "<root>/<uuid>", or "" when the channel is unbound / HOME is unset.
    std::string instanceDir() const;

private:
    void establishInstanceUuid();
    void syncPollThread(bool armedNow);
    void pollLoop();
    void consumeControlFile();
    bool readCurrentValues(std::vector<double>& out) const;
    void writeApplied(long long seq);
    void writeError(long long seq, const std::string& message);
    void heartbeat();

    // One fetched param.  Exactly one pointer is non-null when available.
    struct Bound {
        OFX::DoubleParam*  d = nullptr;
        OFX::IntParam*     i = nullptr;
        OFX::BooleanParam* b = nullptr;
        OFX::ChoiceParam*  c = nullptr;
    };

    OFX::ImageEffect&  m_Effect;
    const char*        m_ToolId;
    const ParamSpec*   m_Specs;
    size_t             m_SpecCount;
    std::string        m_Prefix;
    bool               m_PublishInput;

    OFX::StringParam*  m_InstanceUuidParam;
    std::string        m_InstanceUuid;   // empty = identity could not be established

    std::vector<Bound>         m_Bound;
    std::vector<unsigned char> m_Available;   // 1 byte per spec

    std::mutex        m_ThreadMutex;
    std::thread       m_PollThread;
    std::atomic<bool> m_PollRunning;
    std::atomic<bool> m_PollStop;
    std::atomic<bool> m_ShutDown;
    uint64_t          m_LastGradeMtime;
};

#endif // LDT_CONTROL_SELFTEST

} // namespace control
} // namespace ldt
