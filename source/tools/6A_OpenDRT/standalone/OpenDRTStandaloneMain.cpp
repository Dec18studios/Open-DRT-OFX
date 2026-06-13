// OpenDRTStandaloneMain.cpp — single-factory OFX entry point for the FREE,
// standalone OpenDRT.ofx build (the GPL-3.0 free download).
//
// This is the dual-build twin of src/LookDevTools.cpp: it registers ONLY the
// OpenDRT factory, from the SAME tools/6A_OpenDRT source. It is compiled with
// LDT_OPENDRT_STANDALONE defined (see CMake option LDT_STANDALONE_OPENDRT), so
// OpenDRT.h resolves OPENDRT_PLUGIN_ID to "com.dec18studios.OpenDRT" and the
// factory advertises label "OpenDRT" / group "Dec. 18 Studios" — a distinct OFX
// plugin identifier from the suite's "com.dec18studios.lookdev.openDRT", so the
// free build and the suite bundle can co-install without colliding.
//
// OpenDRT is Jed Smith's open display transform, GPL-3.0. See LICENSE / NOTICE.

#include "ofxsImageEffect.h"

#include "../OpenDRT.h"

namespace OFX { namespace Plugin {
    void getPluginIDs(PluginFactoryArray& ids) {
        static OpenDRTFactory openDRT;
        ids.push_back(&openDRT);
    }
}}
