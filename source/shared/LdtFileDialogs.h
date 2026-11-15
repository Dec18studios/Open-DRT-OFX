#pragma once
// LdtFileDialogs.h - native file pickers and "show folder" for Look Dev Tools.
//
// Implementations:
//   macOS:         LdtFileDialogs.mm  (NSSavePanel / NSOpenPanel / NSWorkspace, MRC)
//   Windows/Linux: LdtFileDialogs.cpp (IFileSaveDialog / IFileOpenDialog;
//                                      zenity then kdialog via argv)
//
// Every call blocks until the user is done and returns a UTF-8 path, or an
// empty string on cancel, on error, or when no picker is available. Safe to
// call from a host param-changed callback. On macOS a call from a background
// thread hops to the main thread; if the main thread does not pick the request
// up within a couple of seconds (it is blocked waiting on the caller), the
// call gives up and returns empty instead of deadlocking.
//
// No OFX includes, no host calls.

#include <string>

namespace ldt {
namespace dialog {

// Ask where to save. suggestedName is the file name shown in the field (the
// extension is added when missing). extension is without the dot ("json").
// startDir may be empty (Desktop, then Documents, then home).
std::string chooseSavePath(const std::string& title, const std::string& suggestedName,
                           const std::string& extension, const std::string& startDir = std::string());

// Ask for one existing file. extension filters the list ("json"); empty
// shows everything. startDir as above.
std::string chooseOpenPath(const std::string& title, const std::string& extension,
                           const std::string& startDir = std::string());

// Open a folder in the platform file browser. Creates the folder first when
// it is missing. Returns false when nothing could be launched. Does not wait
// for the file browser.
bool revealFolder(const std::string& dir);

// Desktop, then Documents, then home, then "" (UTF-8). Shared default folder
// for export and import pickers.
std::string defaultDocumentsDirectory();

} // namespace dialog
} // namespace ldt
