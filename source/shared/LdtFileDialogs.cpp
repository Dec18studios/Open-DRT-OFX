// LdtFileDialogs.cpp - Windows and Linux file pickers for Look Dev Tools
// (see LdtFileDialogs.h). macOS uses LdtFileDialogs.mm instead; on Apple this
// file compiles to nothing unless LDT_DIALOGS_POSIX_ON_APPLE is defined (used
// only to syntax-check the POSIX branch on a Mac).
//
// Windows: Common Item Dialog (IFileSaveDialog / IFileOpenDialog) in a COM
//   single-threaded apartment. If the calling thread is already in a
//   multi-threaded apartment, the dialog runs on a short-lived STA thread and
//   the caller waits for it. Link: ole32, shell32, uuid.
// Linux (and other POSIX): spawn zenity, then kdialog, with an argv vector
//   (posix_spawnp, never a shell string, never system()). stdout is read
//   through a pipe; the child is reaped with a waitpid loop that retries on
//   EINTR (hosts install SIGCHLD handlers). A helper that is not installed is
//   skipped; when none is available the call returns "".

#include "LdtFileDialogs.h"

#if defined(_WIN32)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <objbase.h>
#include <shellapi.h>
#include <shlobj.h>
#include <shobjidl.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>

namespace fs = std::filesystem;

namespace ldt {
namespace dialog {

namespace {

std::wstring widen(const std::string& s)
{
    if (s.empty()) return std::wstring();
    const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), nullptr, 0);
    if (n <= 0) return std::wstring();
    std::wstring w((size_t)n, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int)s.size(), &w[0], n);
    return w;
}

std::string narrow(const wchar_t* w)
{
    if (!w || !*w) return std::string();
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 1) return std::string();
    std::string s((size_t)(n - 1), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w, -1, &s[0], n, nullptr, nullptr);
    return s;
}

std::string withExtension(const std::string& name, const std::string& ext)
{
    if (ext.empty()) return name;
    const std::string dotExt = "." + ext;
    if (name.size() >= dotExt.size()) {
        std::string tail = name.substr(name.size() - dotExt.size());
        for (char& c : tail) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (tail == dotExt) return name;
    }
    return name + dotExt;
}

// Runs fn in an STA. Inline when this thread can be (or already is) an STA,
// otherwise on a temporary STA thread.
template <typename Fn>
std::string inSta(Fn fn)
{
    const HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    if (SUCCEEDED(hr)) {
        std::string r = fn();
        CoUninitialize();
        return r;
    }
    if (hr != RPC_E_CHANGED_MODE) return std::string();
    std::string result;
    std::thread t([&]() {
        const HRESULT h2 = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
        if (FAILED(h2)) return;
        result = fn();
        CoUninitialize();
    });
    t.join();
    return result;
}

void setStartFolder(IFileDialog* dlg, const std::string& startDir)
{
    std::string dir = startDir.empty() ? defaultDocumentsDirectory() : startDir;
    if (dir.empty()) return;
    IShellItem* item = nullptr;
    if (SUCCEEDED(SHCreateItemFromParsingName(widen(dir).c_str(), nullptr, IID_PPV_ARGS(&item))) && item) {
        dlg->SetFolder(item);
        item->Release();
    }
}

std::string resultPath(IFileDialog* dlg)
{
    IShellItem* item = nullptr;
    if (FAILED(dlg->GetResult(&item)) || !item) return std::string();
    PWSTR w = nullptr;
    std::string out;
    if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &w)) && w) {
        out = narrow(w);
        CoTaskMemFree(w);
    }
    item->Release();
    return out;
}

std::string runDialog(bool save, const std::string& title, const std::string& suggestedName,
                      const std::string& extension, const std::string& startDir)
{
    IFileDialog* dlg = nullptr;
    HRESULT hr = save
        ? CoCreateInstance(CLSID_FileSaveDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS((IFileSaveDialog**)&dlg))
        : CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS((IFileOpenDialog**)&dlg));
    if (FAILED(hr) || !dlg) return std::string();

    DWORD flags = 0;
    dlg->GetOptions(&flags);
    flags |= FOS_FORCEFILESYSTEM | FOS_NOCHANGEDIR;
    if (save) flags |= FOS_OVERWRITEPROMPT;
    else      flags |= FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST;
    dlg->SetOptions(flags);

    const std::wstring wTitle = widen(title);
    if (!wTitle.empty()) dlg->SetTitle(wTitle.c_str());

    const std::wstring wPattern = widen(extension.empty() ? std::string("*.*") : "*." + extension);
    const std::wstring wLabel = widen(extension.empty() ? std::string("All files")
                                                        : "Preset files (*." + extension + ")");
    COMDLG_FILTERSPEC specs[2] = {
        { wLabel.c_str(), wPattern.c_str() },
        { L"All files", L"*.*" },
    };
    dlg->SetFileTypes(extension.empty() ? 1u : 2u, specs);
    dlg->SetFileTypeIndex(1);
    const std::wstring wExt = widen(extension);
    if (!wExt.empty()) dlg->SetDefaultExtension(wExt.c_str());
    setStartFolder(dlg, startDir);
    if (save) {
        const std::wstring wName =
            widen(withExtension(suggestedName.empty() ? std::string("Preset") : suggestedName, extension));
        dlg->SetFileName(wName.c_str());
    }

    std::string out;
    if (SUCCEEDED(dlg->Show(GetForegroundWindow()))) out = resultPath(dlg);
    dlg->Release();
    return out;
}

} // namespace

std::string defaultDocumentsDirectory()
{
    std::error_code ec;
    const wchar_t* prof = _wgetenv(L"USERPROFILE");
    if (!prof || !*prof) return std::string();
    const fs::path root(prof);
    if (fs::is_directory(root / "Desktop", ec)) return (root / "Desktop").u8string();
    if (fs::is_directory(root / "Documents", ec)) return (root / "Documents").u8string();
    return root.u8string();
}

std::string chooseSavePath(const std::string& title, const std::string& suggestedName,
                           const std::string& extension, const std::string& startDir)
{
    return inSta([&]() { return runDialog(true, title, suggestedName, extension, startDir); });
}

std::string chooseOpenPath(const std::string& title, const std::string& extension, const std::string& startDir)
{
    return inSta([&]() { return runDialog(false, title, std::string(), extension, startDir); });
}

bool revealFolder(const std::string& dir)
{
    if (dir.empty()) return false;
    std::error_code ec;
    const fs::path p = fs::u8path(dir);
    fs::create_directories(p, ec);
    if (!fs::is_directory(p, ec)) return false;
    const HINSTANCE h = ShellExecuteW(nullptr, L"open", p.wstring().c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    return (INT_PTR)h > 32;
}

} // namespace dialog
} // namespace ldt

#elif !defined(__APPLE__) || defined(LDT_DIALOGS_POSIX_ON_APPLE)

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <spawn.h>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

extern char** environ;

namespace fs = std::filesystem;

namespace ldt {
namespace dialog {

namespace {

// waitpid that survives EINTR from a host SIGCHLD handler.
pid_t waitpidRetry(pid_t pid, int* status, int options)
{
    for (;;) {
        const pid_t r = waitpid(pid, status, options);
        if (r == -1 && errno == EINTR) continue;
        return r;
    }
}

std::string withExtension(const std::string& name, const std::string& ext)
{
    if (ext.empty()) return name;
    const std::string dotExt = "." + ext;
    if (name.size() >= dotExt.size()) {
        std::string tail = name.substr(name.size() - dotExt.size());
        for (char& c : tail) if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
        if (tail == dotExt) return name;
    }
    return name + dotExt;
}

enum class RunResult { NotAvailable, Cancelled, Ok };

// Spawn argv[0] from PATH with stdin and stderr on /dev/null, collect stdout.
RunResult runHelper(const std::vector<std::string>& args, std::string& out)
{
    out.clear();
    if (args.empty()) return RunResult::NotAvailable;
    int fds[2] = { -1, -1 };
    if (pipe(fds) != 0) return RunResult::NotAvailable;
    fcntl(fds[0], F_SETFD, FD_CLOEXEC);

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    posix_spawn_file_actions_addclose(&fa, fds[1]);

    std::vector<char*> argv;
    argv.reserve(args.size() + 1);
    for (const auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);

    pid_t pid = -1;
    const int rc = posix_spawnp(&pid, argv[0], &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    close(fds[1]);
    if (rc != 0) { close(fds[0]); return RunResult::NotAvailable; }

    char buf[4096];
    for (;;) {
        const ssize_t n = read(fds[0], buf, sizeof(buf));
        if (n > 0) { out.append(buf, (size_t)n); if (out.size() > 65536) break; continue; }
        if (n == -1 && errno == EINTR) continue;
        break;
    }
    close(fds[0]);

    int status = 0;
    if (waitpidRetry(pid, &status, 0) != pid) return RunResult::Cancelled;
    if (!WIFEXITED(status)) return RunResult::Cancelled;
    const int code = WEXITSTATUS(status);
    if (code == 127) return RunResult::NotAvailable;   // exec failed in the child
    if (code != 0) return RunResult::Cancelled;
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    const size_t nl = out.find('\n');
    if (nl != std::string::npos) out.resize(nl);
    return out.empty() ? RunResult::Cancelled : RunResult::Ok;
}

std::string pick(const std::vector<std::vector<std::string>>& candidates)
{
    for (const auto& args : candidates) {
        std::string out;
        const RunResult r = runHelper(args, out);
        if (r == RunResult::Ok) return out;
        if (r == RunResult::Cancelled) return std::string();
        // NotAvailable: try the next helper.
    }
    return std::string();
}

std::string startDirOrDefault(const std::string& startDir)
{
    std::string dir = startDir.empty() ? defaultDocumentsDirectory() : startDir;
    std::error_code ec;
    if (!dir.empty() && !fs::is_directory(fs::u8path(dir), ec)) dir = defaultDocumentsDirectory();
    return dir;
}

} // namespace

std::string defaultDocumentsDirectory()
{
    std::string home;
    if (const char* h = std::getenv("HOME")) home = h;
    if (home.empty()) return std::string();
    std::error_code ec;
    const fs::path root = fs::u8path(home);
    if (fs::is_directory(root / "Desktop", ec)) return (root / "Desktop").u8string();
    if (fs::is_directory(root / "Documents", ec)) return (root / "Documents").u8string();
    return root.u8string();
}

std::string chooseSavePath(const std::string& title, const std::string& suggestedName,
                           const std::string& extension, const std::string& startDir)
{
    const std::string dir = startDirOrDefault(startDir);
    const std::string name = withExtension(suggestedName.empty() ? std::string("Preset") : suggestedName, extension);
    const std::string full = dir.empty() ? name : (fs::u8path(dir) / fs::u8path(name)).u8string();
    const std::string t = title.empty() ? std::string("Save") : title;

    std::vector<std::string> z { "zenity", "--file-selection", "--save", "--confirm-overwrite",
                                 "--title=" + t, "--filename=" + full };
    std::vector<std::string> k { "kdialog", "--title", t, "--getsavefilename", full };
    if (!extension.empty()) {
        z.push_back("--file-filter=Preset files | *." + extension);
        z.push_back("--file-filter=All files | *");
        k.push_back("*." + extension + "|Preset files");
    }
    std::string path = pick({ z, k });
    if (!path.empty()) path = withExtension(path, extension);
    return path;
}

std::string chooseOpenPath(const std::string& title, const std::string& extension, const std::string& startDir)
{
    std::string dir = startDirOrDefault(startDir);
    if (!dir.empty() && dir.back() != '/') dir += '/';
    const std::string t = title.empty() ? std::string("Open") : title;

    std::vector<std::string> z { "zenity", "--file-selection", "--title=" + t };
    if (!dir.empty()) z.push_back("--filename=" + dir);
    std::vector<std::string> k { "kdialog", "--title", t, "--getopenfilename", dir.empty() ? std::string(".") : dir };
    if (!extension.empty()) {
        z.push_back("--file-filter=Preset files | *." + extension);
        z.push_back("--file-filter=All files | *");
        k.push_back("*." + extension + "|Preset files");
    }
    return pick({ z, k });
}

bool revealFolder(const std::string& dir)
{
    if (dir.empty()) return false;
    std::error_code ec;
    const fs::path p = fs::u8path(dir);
    fs::create_directories(p, ec);
    if (!fs::is_directory(p, ec)) return false;
    const std::string path = p.u8string();

    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 1, "/dev/null", O_WRONLY, 0);
    posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
    std::string prog = "xdg-open";
    char* argv[] = { const_cast<char*>(prog.c_str()), const_cast<char*>(path.c_str()), nullptr };
    pid_t pid = -1;
    const int rc = posix_spawnp(&pid, argv[0], &fa, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) return false;
    // xdg-open can stay alive as long as the file browser does. Reap it on a
    // detached thread so the caller never blocks and no zombie is left.
    std::thread([pid]() { int st = 0; waitpidRetry(pid, &st, 0); }).detach();
    return true;
}

} // namespace dialog
} // namespace ldt

#endif
