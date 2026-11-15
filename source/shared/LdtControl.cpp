#include "LdtControl.h"

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <mutex>
#include <set>
#include <random>
#include <chrono>
#include <atomic>
#include <vector>
#include <algorithm>

#if !defined(_WIN32)
  #include <sys/stat.h>
  #include <sys/types.h>
  #include <sys/time.h>
  #include <unistd.h>
  #include <fcntl.h>
#endif

#if defined(__APPLE__)
  #include <uuid/uuid.h>
#endif

#include "ofxsImageEffect.h"
#include "ofxsParam.h"

namespace ldt {
namespace control {

// Poll cadence for grade.json, in milliseconds (slept in 20 ms slices so a
// disarm or a destructor is observed quickly).
static const int kPollIntervalMs = 100;

// ── Paths ────────────────────────────────────────────────────────────────────
const std::string& rootDir()
{
    static std::string s_root = []() -> std::string {
        const char* home = getenv("HOME");
        if (!home || !*home) return std::string();
        return std::string(home) + "/.ofx-lookdevtools";
    }();
    return s_root;
}

std::string joinPath(const std::string& dir, const std::string& leaf)
{
    if (dir.empty()) return leaf;
    if (!dir.empty() && dir[dir.size() - 1] == '/') return dir + leaf;
    return dir + "/" + leaf;
}

bool ensureDir(const std::string& path)
{
#if defined(_WIN32)
    (void)path;
    return false;   // channel is stubbed on Windows
#else
    if (path.empty()) return false;

    struct stat st;
    if (stat(path.c_str(), &st) == 0)
        return S_ISDIR(st.st_mode);

    // Create parents first.
    const size_t slash = path.find_last_of('/');
    if (slash != std::string::npos && slash > 0) {
        const std::string parent = path.substr(0, slash);
        if (stat(parent.c_str(), &st) != 0) {
            if (!ensureDir(parent)) return false;
        }
    }

    if (mkdir(path.c_str(), 0700) == 0) return true;
    // Lost a race with another process: fine as long as it is a directory now.
    return (stat(path.c_str(), &st) == 0) && S_ISDIR(st.st_mode);
#endif
}

// ── Sentinel ─────────────────────────────────────────────────────────────────
bool armed()
{
#if defined(_WIN32)
    return false;   // the whole channel is inert on Windows in v1
#else
    const std::string& root = rootDir();
    if (root.empty()) return false;
    struct stat st;
    return stat((root + "/armed").c_str(), &st) == 0;
#endif
}

// ── Atomic IO ────────────────────────────────────────────────────────────────
bool atomicWrite(const std::string& path, const void* data, size_t len)
{
#if defined(_WIN32)
    (void)path; (void)data; (void)len;
    return false;
#else
    if (path.empty()) return false;

    const std::string tmp = path + ".tmp";

    const int fd = open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return false;

    const char* p = static_cast<const char*>(data);
    size_t remaining = len;
    bool ok = true;
    while (remaining > 0) {
        const ssize_t n = write(fd, p, remaining);
        if (n <= 0) { ok = false; break; }
        p         += (size_t)n;
        remaining -= (size_t)n;
    }
    close(fd);

    if (!ok) { unlink(tmp.c_str()); return false; }

    if (rename(tmp.c_str(), path.c_str()) != 0) {
        unlink(tmp.c_str());
        return false;
    }
    return true;
#endif
}

bool atomicWriteText(const std::string& path, const std::string& text)
{
    return atomicWrite(path, text.data(), text.size());
}

bool readFile(const std::string& path, std::string& out)
{
    out.clear();
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    char buf[8192];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0)
        out.append(buf, n);

    const bool bad = ferror(f) != 0;
    fclose(f);
    if (bad) { out.clear(); return false; }
    return true;
}

uint64_t mtimeNanos(const std::string& path)
{
#if defined(_WIN32)
    (void)path;
    return 0;
#else
    struct stat st;
    if (stat(path.c_str(), &st) != 0) return 0;
  #if defined(__APPLE__)
    return (uint64_t)st.st_mtimespec.tv_sec * 1000000000ull + (uint64_t)st.st_mtimespec.tv_nsec;
  #else
    return (uint64_t)st.st_mtim.tv_sec * 1000000000ull + (uint64_t)st.st_mtim.tv_nsec;
  #endif
#endif
}

bool touchFile(const std::string& path)
{
#if defined(_WIN32)
    (void)path;
    return false;
#else
    if (utimes(path.c_str(), nullptr) == 0) return true;
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT, 0600);
    if (fd < 0) return false;
    close(fd);
    return true;
#endif
}

// ── Identity ─────────────────────────────────────────────────────────────────
std::string generateUuid()
{
#if defined(__APPLE__)
    uuid_t raw;
    uuid_generate(raw);
    char text[37];
    uuid_unparse_lower(raw, text);
    return std::string(text);
#else
    // Portable RFC-4122 v4 fallback — avoids a libuuid link dependency.
    static std::mutex      s_mutex;
    static std::mt19937_64 s_rng(
        (uint64_t)std::chrono::steady_clock::now().time_since_epoch().count() ^
        ((uint64_t)std::random_device{}() << 32) ^
  #if defined(_WIN32)
        (uint64_t)std::random_device{}());
  #else
        (uint64_t)getpid());
  #endif

    uint64_t hi, lo;
    {
        std::lock_guard<std::mutex> lk(s_mutex);
        hi = s_rng();
        lo = s_rng();
    }
    hi = (hi & 0xFFFFFFFFFFFF0FFFull) | 0x0000000000004000ull;  // version 4
    lo = (lo & 0x3FFFFFFFFFFFFFFFull) | 0x8000000000000000ull;  // variant 1

    char text[37];
    snprintf(text, sizeof(text), "%08x-%04x-%04x-%04x-%012llx",
             (unsigned)(hi >> 32),
             (unsigned)((hi >> 16) & 0xFFFF),
             (unsigned)(hi & 0xFFFF),
             (unsigned)((lo >> 48) & 0xFFFF),
             (unsigned long long)(lo & 0xFFFFFFFFFFFFull));
    return std::string(text);
#endif
}

namespace {
std::mutex&            uuidMutex() { static std::mutex m; return m; }
std::set<std::string>& uuidSet()   { static std::set<std::string> s; return s; }
}

bool claimUuid(const std::string& uuid)
{
    if (uuid.empty()) return false;
    std::lock_guard<std::mutex> lk(uuidMutex());
    return uuidSet().insert(uuid).second;
}

void releaseUuid(const std::string& uuid)
{
    if (uuid.empty()) return;
    std::lock_guard<std::mutex> lk(uuidMutex());
    uuidSet().erase(uuid);
}

double unixTimeNow()
{
    using namespace std::chrono;
    return duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
}

// ── Tap publishing ───────────────────────────────────────────────────────────

// Longest edge of a published snapshot, in pixels.
static const int kMaxPublishWidth = 512;

// Process-global monotonically increasing publish counter, shared by every slot
// and every publisher, so a reader's min-seq freshness gate spans the whole
// suite.
static std::atomic<unsigned long long> s_PublishSeq(0);

// The OFX buffer on this host is bottom-up (row 0 is the bottom scanline); the
// published .bin is top-down, so the vertical mapping is inverted here.
bool publishTapFrame(const float* srcBottomUp, int width, int height,
                     double time, int slot)
{
    if (!srcBottomUp || width <= 0 || height <= 0) return false;

    const std::string& root = rootDir();
    if (root.empty()) return false;

    const std::string tapDir = joinPath(root, "tap");
    if (!ensureDir(tapDir)) return false;

    if (slot < 0) slot = 0;
    if (slot > 3) slot = 3;

    // --- output geometry ----------------------------------------------------
    const int outW = std::min(width, kMaxPublishWidth);
    if (outW < 1) return false;
    int outH = (int)std::floor((double)height * (double)outW / (double)width + 0.5);
    if (outH < 1)      outH = 1;
    if (outH > height) outH = height;

    std::vector<float> out((size_t)outW * (size_t)outH * 4, 0.0f);

    // --- box average --------------------------------------------------------
    for (int oy = 0; oy < outH; ++oy) {
        // Rows counted from the TOP of the image.
        int ty0 = (int)((long long)oy       * height / outH);
        int ty1 = (int)((long long)(oy + 1) * height / outH);
        if (ty1 <= ty0) ty1 = ty0 + 1;
        if (ty1 > height) ty1 = height;

        for (int ox = 0; ox < outW; ++ox) {
            int sx0 = (int)((long long)ox       * width / outW);
            int sx1 = (int)((long long)(ox + 1) * width / outW);
            if (sx1 <= sx0) sx1 = sx0 + 1;
            if (sx1 > width) sx1 = width;

            double ar = 0.0, ag = 0.0, ab = 0.0, aa = 0.0;
            long long n = 0;

            for (int ty = ty0; ty < ty1; ++ty) {
                // Top-down row ty lives at buffer row (height - 1 - ty).
                const int by = height - 1 - ty;
                const float* row = srcBottomUp + (size_t)by * width * 4;
                for (int sx = sx0; sx < sx1; ++sx) {
                    const int i = sx * 4;
                    ar += row[i + 0];
                    ag += row[i + 1];
                    ab += row[i + 2];
                    aa += row[i + 3];
                    ++n;
                }
            }
            if (n == 0) continue;

            const double inv = 1.0 / (double)n;
            float* o = &out[((size_t)oy * outW + ox) * 4];
            o[0] = (float)(ar * inv);
            o[1] = (float)(ag * inv);
            o[2] = (float)(ab * inv);
            o[3] = (float)(aa * inv);
        }
    }

    // --- write .bin, then .meta (always in that order) ----------------------
    char nameBuf[64];
    snprintf(nameBuf, sizeof(nameBuf), "frame_%d.bin", slot);
    const std::string binPath = joinPath(tapDir, nameBuf);

    if (!atomicWrite(binPath, out.data(), out.size() * sizeof(float)))
        return false;   // no .bin means no .meta: a reader never sees a stale pairing

    const unsigned long long seq = s_PublishSeq.fetch_add(1) + 1;

    char meta[320];
    const int n = snprintf(meta, sizeof(meta),
        "{\"seq\":%llu,\"width\":%d,\"height\":%d,\"time\":%.6f,\"unix_time\":%.6f}\n",
        seq, outW, outH, time, unixTimeNow());
    if (n <= 0) return false;

    snprintf(nameBuf, sizeof(nameBuf), "frame_%d.meta", slot);
    return atomicWrite(joinPath(tapDir, nameBuf), meta, (size_t)n);
}

// ============================================================================
// Channel
// ============================================================================

// Hidden persistent identity param. One per plugin instance; the monolith
// defines exactly one for the whole node, not one per embedded child.
static const char* const kInstanceUuidParam = "ldt_instance_uuid";

// ── describe ─────────────────────────────────────────────────────────────────
// setIsSecret / setIsPersistant MUST be called here, inside the define block and
// before the descriptor is finalised — a post-pass over the finished descriptor
// is a silent no-op on this host. Define it LAST so it never shifts the restore
// order of the tool's existing params.
void Channel::describeUuidParam(OFX::ImageEffectDescriptor& p_Desc,
                                OFX::PageParamDescriptor* p_Page)
{
    OFX::StringParamDescriptor* uuidParam = p_Desc.defineStringParam(kInstanceUuidParam);
    uuidParam->setLabels("Instance UUID", "Instance UUID", "Instance UUID");
    uuidParam->setScriptName(kInstanceUuidParam);
    uuidParam->setHint("Internal Look Dev Tools instance identity. Not user-editable.");
    uuidParam->setStringType(OFX::eStringTypeSingleLine);
    uuidParam->setDefault("");
    uuidParam->setIsSecret(true);
    uuidParam->setIsPersistant(true);
    uuidParam->setAnimates(false);
    uuidParam->setEvaluateOnChange(false);
    if (p_Page) p_Page->addChild(*uuidParam);
}

// ── lifetime ─────────────────────────────────────────────────────────────────
Channel::Channel(OFX::ImageEffect& p_Effect, const char* p_ToolId,
                 const ParamSpec* p_Specs, size_t p_Count,
                 const std::string& p_Prefix, bool p_PublishInputWhenArmed)
    : m_Effect(p_Effect)
    , m_ToolId(p_ToolId ? p_ToolId : "")
    , m_Specs(p_Specs)
    , m_SpecCount(p_Specs ? p_Count : 0)
    , m_Prefix(p_Prefix)
    , m_PublishInput(p_PublishInputWhenArmed)
    , m_InstanceUuidParam(nullptr)
    , m_PollRunning(false)
    , m_PollStop(false)
    , m_ShutDown(false)
    , m_LastGradeMtime(0)
{
}

Channel::~Channel()
{
    shutdown();
}

void Channel::shutdown()
{
    if (m_ShutDown.exchange(true)) return;   // idempotent

    m_PollStop.store(true);
    try {
        std::lock_guard<std::mutex> lk(m_ThreadMutex);
        if (m_PollThread.joinable()) m_PollThread.join();
    } catch (...) {
        // A join failure must never propagate out of a destructor path.
    }
    releaseUuid(m_InstanceUuid);
    m_InstanceUuid.clear();
}

// ── bind ─────────────────────────────────────────────────────────────────────
void Channel::bind()
{
    m_Bound.assign(m_SpecCount, Bound());
    m_Available.assign(m_SpecCount, 0);

    try {
        m_InstanceUuidParam = m_Effect.fetchStringParam(kInstanceUuidParam);
    } catch (...) {
        m_InstanceUuidParam = nullptr;   // tool forgot describeUuidParam(): stay inert
    }

    establishInstanceUuid();

    // A param the host does not carry (an older project file, or a tool whose
    // whitelist got ahead of its descriptor) is skipped, never thrown.
    for (size_t i = 0; i < m_SpecCount; ++i) {
        if (!m_Specs[i].scriptName) continue;
        const std::string name = m_Prefix + m_Specs[i].scriptName;
        try {
            switch (m_Specs[i].kind) {
                case ParamSpec::Double:
                    m_Bound[i].d = m_Effect.fetchDoubleParam(name);
                    m_Available[i] = m_Bound[i].d ? 1 : 0;
                    break;
                case ParamSpec::Int:
                    m_Bound[i].i = m_Effect.fetchIntParam(name);
                    m_Available[i] = m_Bound[i].i ? 1 : 0;
                    break;
                case ParamSpec::Bool:
                    m_Bound[i].b = m_Effect.fetchBooleanParam(name);
                    m_Available[i] = m_Bound[i].b ? 1 : 0;
                    break;
                case ParamSpec::Choice:
                    m_Bound[i].c = m_Effect.fetchChoiceParam(name);
                    m_Available[i] = m_Bound[i].c ? 1 : 0;
                    break;
            }
        } catch (...) {
            m_Bound[i]     = Bound();
            m_Available[i] = 0;
        }
    }
}

// ----------------------------------------------------------------------------
// Identity: read the persisted UUID; mint one when empty; mint a fresh one when
// the persisted value is already claimed by a live instance in this process
// (that is an Alt-drag clone of an existing node).
// ----------------------------------------------------------------------------
void Channel::establishInstanceUuid()
{
    if (!m_InstanceUuidParam) return;

    std::string stored;
    try {
        m_InstanceUuidParam->getValue(stored);
    } catch (...) {
        stored.clear();
    }

    if (!stored.empty() && claimUuid(stored)) {
        m_InstanceUuid = stored;
        return;
    }

    // Empty, or a clone of a UUID already in use: mint a fresh one.
    for (int attempt = 0; attempt < 8; ++attempt) {
        const std::string fresh = generateUuid();
        if (claimUuid(fresh)) {
            m_InstanceUuid = fresh;
            try {
                m_InstanceUuidParam->setValue(fresh);
            } catch (...) {
                // Host refused the write (e.g. during a describe-time probe).
                // The in-memory UUID still works for this session.
            }
            return;
        }
    }
}

std::string Channel::instanceDir() const
{
    if (m_InstanceUuid.empty()) return std::string();
    const std::string& root = rootDir();
    if (root.empty()) return std::string();
    return joinPath(root, m_InstanceUuid);
}

// ── render hook ──────────────────────────────────────────────────────────────
void Channel::onRender()
{
    // Exactly one stat of the sentinel per render call. Everything else below
    // is skipped while disarmed.
    const bool armedNow = armed();
    syncPollThread(armedNow);
    if (armedNow) heartbeat();
}

void Channel::publishInput(const float* p_Src, int p_W, int p_H,
                           double p_Time, int p_Slot)
{
    if (!m_PublishInput || !p_Src) return;
    if (!armed()) return;
    try {
        publishTapFrame(p_Src, p_W, p_H, p_Time, p_Slot);
    } catch (...) {
        // Best effort: IO never fails a render.
    }
}

// ── poll-thread lifecycle ────────────────────────────────────────────────────
void Channel::syncPollThread(bool p_Armed)
{
    if (m_InstanceUuid.empty() || m_ShutDown.load()) return;

    if (p_Armed) {
        if (m_PollRunning.load()) return;   // already going: nothing to do

        std::lock_guard<std::mutex> lk(m_ThreadMutex);
        if (m_PollRunning.load()) return;
        if (m_PollThread.joinable()) m_PollThread.join();   // reap a finished thread

        m_PollStop.store(false);
        m_PollRunning.store(true);
        m_LastGradeMtime = 0;   // force a re-read of grade.json on re-arm
        try {
            m_PollThread = std::thread(&Channel::pollLoop, this);
        } catch (...) {
            m_PollRunning.store(false);
        }
    } else {
        // Disarmed: the thread exits on its own next tick; reap it here.
        if (m_PollRunning.load()) m_PollStop.store(true);

        std::lock_guard<std::mutex> lk(m_ThreadMutex);
        if (!m_PollRunning.load() && m_PollThread.joinable())
            m_PollThread.join();
    }
}

void Channel::pollLoop()
{
    while (!m_PollStop.load()) {
        if (!armed()) break;   // sentinel gone: stop all file IO

        const std::string dir = instanceDir();
        if (!dir.empty()) {
            const uint64_t mt = mtimeNanos(joinPath(dir, "grade.json"));
            if (mt != 0 && mt != m_LastGradeMtime) {
                m_LastGradeMtime = mt;
                try {
                    consumeControlFile();
                } catch (...) {
                    // Never let anything escape the thread body.
                }
            }
        }

        // Sleep in short slices so a disarm or a shutdown is observed quickly.
        for (int slept = 0; slept < kPollIntervalMs && !m_PollStop.load(); slept += 20)
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    m_PollRunning.store(false);
}

// ── grade.json -> params ─────────────────────────────────────────────────────
void Channel::consumeControlFile()
{
    const std::string dir = instanceDir();
    if (dir.empty()) return;

    const std::string path = joinPath(dir, "grade.json");

    std::string text;
    if (!readFile(path, text)) {
        writeError(-1, "grade.json could not be read");
        return;
    }

    ldtjson::Value doc;
    std::string    err;
    if (!ldtjson::parse(text, doc, err)) {
        writeError(-1, std::string("grade.json parse error: ") + err);
        return;
    }

    // Validate + decode EVERYTHING before touching a single param: a malformed
    // field must leave the node exactly as it was, with no partial apply.
    const detail::DecodeResult decoded = detail::decodeGrade(
        doc, m_Specs, m_SpecCount,
        m_Available.empty() ? nullptr : &m_Available[0]);

    if (!decoded.ok) {
        writeError(decoded.seq, decoded.error);
        return;
    }

    // --- apply --------------------------------------------------------------
    // Spontaneous setValue does not schedule a re-render on this host; the
    // external controller is responsible for nudging Resolve. That is expected.
    if (!decoded.values.empty()) {
        try {
            m_Effect.beginEditBlock("LookDevTools remote grade");

            for (size_t k = 0; k < decoded.values.size(); ++k) {
                const size_t  idx = decoded.values[k].specIndex;
                const double  v   = decoded.values[k].value;
                const Bound&  b   = m_Bound[idx];

                switch (m_Specs[idx].kind) {
                    case ParamSpec::Double: if (b.d) b.d->setValue(v);                    break;
                    case ParamSpec::Int:    if (b.i) b.i->setValue((int)llround(v));      break;
                    case ParamSpec::Bool:   if (b.b) b.b->setValue(v != 0.0);             break;
                    case ParamSpec::Choice: if (b.c) b.c->setValue((int)llround(v));      break;
                }
            }

            m_Effect.endEditBlock();
        } catch (...) {
            writeError(decoded.seq, "host rejected the parameter write");
            return;
        }
    }

    writeApplied(decoded.seq);
}

// ── applied.json / error.json / heartbeat ────────────────────────────────────
bool Channel::readCurrentValues(std::vector<double>& p_Out) const
{
    p_Out.assign(m_SpecCount, 0.0);

    for (size_t i = 0; i < m_SpecCount; ++i) {
        if (i >= m_Available.size() || !m_Available[i]) continue;
        const Bound& b = m_Bound[i];
        try {
            // Time 0 semantics: the channel never animates these params.
            switch (m_Specs[i].kind) {
                case ParamSpec::Double: { double d = 0.0; if (b.d) b.d->getValue(d); p_Out[i] = d;                break; }
                case ParamSpec::Int:    { int    v = 0;   if (b.i) b.i->getValue(v); p_Out[i] = (double)v;        break; }
                case ParamSpec::Bool:   { bool   v = false; if (b.b) b.b->getValue(v); p_Out[i] = v ? 1.0 : 0.0;  break; }
                case ParamSpec::Choice: { int    v = 0;   if (b.c) b.c->getValue(v); p_Out[i] = (double)v;        break; }
            }
        } catch (...) {
            return false;
        }
    }
    return true;
}

void Channel::writeApplied(long long p_Seq)
{
    const std::string dir = instanceDir();
    if (dir.empty()) return;
    if (!ensureDir(dir)) return;

    std::vector<double> values;
    if (!readCurrentValues(values)) return;

    const std::string body = detail::formatApplied(
        m_ToolId, p_Seq, unixTimeNow(), m_Specs, m_SpecCount,
        values.empty() ? nullptr : &values[0],
        m_Available.empty() ? nullptr : &m_Available[0]);

    atomicWriteText(joinPath(dir, "applied.json"), body);
}

void Channel::writeError(long long p_Seq, const std::string& p_Message)
{
    const std::string dir = instanceDir();
    if (dir.empty()) return;
    if (!ensureDir(dir)) return;

    atomicWriteText(joinPath(dir, "error.json"),
                    detail::formatError(m_ToolId, p_Seq, unixTimeNow(), p_Message));
}

void Channel::heartbeat()
{
    const std::string dir = instanceDir();
    if (dir.empty()) return;
    if (!ensureDir(dir)) return;
    touchFile(joinPath(dir, "heartbeat"));
}

} // namespace control
} // namespace ldt
