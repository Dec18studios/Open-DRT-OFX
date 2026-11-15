// LdtFileDialogs.mm - macOS file pickers for Look Dev Tools (see LdtFileDialogs.h).
//
// Manual retain/release: the suite builds Objective-C++ without -fobjc-arc.
// Panels come from +savePanel / +openPanel (autoreleased) inside an explicit
// autorelease pool, so nothing here needs a release call except the dispatch
// semaphore. No new Objective-C classes are declared.
//
// Threading: AppKit panels must run on the main thread.
//   - Called on the main thread (the usual case for a host param-changed
//     callback): run the panel inline with -runModal.
//   - Called on any other thread: post a block to the main queue and wait.
//     A plain dispatch_sync would deadlock if the main thread is itself
//     blocked waiting for this callback, so the request is claimed with an
//     atomic flag. If the main thread has not started it within
//     kStartTimeoutSeconds, the caller withdraws it and returns "" (the block
//     then sees the withdrawn flag and shows nothing). Once the panel has
//     started, the caller waits for the user without a time limit.

#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#include <dispatch/dispatch.h>

#include "LdtFileDialogs.h"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace ldt {
namespace dialog {

namespace {

const double kStartTimeoutSeconds = 3.0;

NSString* toNS(const std::string& s)
{
    NSString* out = [NSString stringWithUTF8String:s.c_str()];
    return out ? out : @"";
}

std::string fromNS(NSString* s)
{
    if (!s) return std::string();
    const char* c = [s UTF8String];
    return c ? std::string(c) : std::string();
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

struct Request {
    std::atomic<int> phase { 0 };   // 0 pending, 1 running, 2 withdrawn
    dispatch_semaphore_t started = nullptr;
    dispatch_semaphore_t done = nullptr;
    std::string result;
    std::function<std::string()> work;

    Request()
    {
        started = dispatch_semaphore_create(0);
        done = dispatch_semaphore_create(0);
    }
    ~Request()
    {
        // MRC: dispatch objects are released by hand.
        if (started) dispatch_release(started);
        if (done) dispatch_release(done);
    }
};

// Run work() on the main thread and return its result. See the file comment.
std::string runOnMain(std::function<std::string()> work)
{
    if ([NSThread isMainThread]) {
        @autoreleasepool {
            return work();
        }
    }
    std::shared_ptr<Request> req = std::make_shared<Request>();
    req->work = std::move(work);
    std::shared_ptr<Request> keep = req;   // the block's own reference
    dispatch_async(dispatch_get_main_queue(), ^{
        int expected = 0;
        if (!keep->phase.compare_exchange_strong(expected, 1)) return;   // withdrawn
        dispatch_semaphore_signal(keep->started);
        @autoreleasepool {
            keep->result = keep->work();
        }
        dispatch_semaphore_signal(keep->done);
    });

    const dispatch_time_t deadline =
        dispatch_time(DISPATCH_TIME_NOW, (int64_t)(kStartTimeoutSeconds * (double)NSEC_PER_SEC));
    if (dispatch_semaphore_wait(req->started, deadline) != 0) {
        int expected = 0;
        if (req->phase.compare_exchange_strong(expected, 2)) return std::string();
        // Lost the race: the block started just now. Fall through and wait.
    }
    dispatch_semaphore_wait(req->done, DISPATCH_TIME_FOREVER);
    return req->result;
}

void applyStartDir(NSSavePanel* panel, const std::string& startDir)
{
    std::string dir = startDir.empty() ? defaultDocumentsDirectory() : startDir;
    if (dir.empty()) return;
    std::error_code ec;
    if (!fs::is_directory(fs::u8path(dir), ec)) return;
    [panel setDirectoryURL:[NSURL fileURLWithPath:toNS(dir) isDirectory:YES]];
}

void applyExtension(NSSavePanel* panel, const std::string& ext)
{
    if (ext.empty()) return;
    // allowedFileTypes is deprecated from macOS 12 in favour of
    // allowedContentTypes, which needs UniformTypeIdentifiers. The old call
    // still works everywhere the suite runs and keeps the link line unchanged.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    [panel setAllowedFileTypes:@[ toNS(ext) ]];
#pragma clang diagnostic pop
}

} // namespace

std::string defaultDocumentsDirectory()
{
    std::string home;
    if (const char* h = std::getenv("HOME")) home = h;
    if (home.empty()) home = fromNS(NSHomeDirectory());
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
    return runOnMain([=]() -> std::string {
        NSSavePanel* panel = [NSSavePanel savePanel];   // autoreleased
        if (!panel) return std::string();
        if (!title.empty()) {
            [panel setTitle:toNS(title)];
            [panel setMessage:toNS(title)];
        }
        [panel setCanCreateDirectories:YES];
        [panel setExtensionHidden:NO];
        applyExtension(panel, extension);
        applyStartDir(panel, startDir);
        const std::string name = withExtension(suggestedName.empty() ? std::string("Preset") : suggestedName,
                                               extension);
        [panel setNameFieldStringValue:toNS(name)];
        if ([panel runModal] != NSModalResponseOK) return std::string();
        NSURL* url = [panel URL];
        if (!url || ![url isFileURL]) return std::string();
        return fromNS([url path]);
    });
}

std::string chooseOpenPath(const std::string& title, const std::string& extension,
                           const std::string& startDir)
{
    return runOnMain([=]() -> std::string {
        NSOpenPanel* panel = [NSOpenPanel openPanel];   // autoreleased
        if (!panel) return std::string();
        if (!title.empty()) {
            [panel setTitle:toNS(title)];
            [panel setMessage:toNS(title)];
        }
        [panel setCanChooseFiles:YES];
        [panel setCanChooseDirectories:NO];
        [panel setAllowsMultipleSelection:NO];
        [panel setResolvesAliases:YES];
        applyExtension(panel, extension);
        applyStartDir(panel, startDir);
        if ([panel runModal] != NSModalResponseOK) return std::string();
        NSArray* urls = [panel URLs];
        if (!urls || [urls count] == 0) return std::string();
        NSURL* url = [urls objectAtIndex:0];
        if (!url || ![url isFileURL]) return std::string();
        return fromNS([url path]);
    });
}

bool revealFolder(const std::string& dir)
{
    if (dir.empty()) return false;
    std::error_code ec;
    const fs::path p = fs::u8path(dir);
    fs::create_directories(p, ec);
    if (!fs::is_directory(p, ec)) return false;
    const std::string path = p.u8string();
    // NSWorkspace, never a shell string. Opening Finder does not need to block
    // the caller, so hop to main without waiting.
    void (^openIt)(void) = ^{
        @autoreleasepool {
            [[NSWorkspace sharedWorkspace] selectFile:nil inFileViewerRootedAtPath:toNS(path)];
        }
    };
    if ([NSThread isMainThread]) openIt();
    else dispatch_async(dispatch_get_main_queue(), openIt);
    return true;
}

} // namespace dialog
} // namespace ldt
