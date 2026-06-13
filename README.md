# OpenDRT for OFX

A free, open-source **OpenDRT** display rendering transform packaged as an
**OpenFX** plug-in for DaVinci Resolve, Fusion, Nuke and other OFX hosts —
GPU-accelerated with **Metal** on macOS and **CUDA** on Windows / Linux.

OpenDRT is [Jed Smith's open display transform](https://github.com/jedypod/open-display-transform).
This repository wraps it as an OFX plug-in and provides cross-platform builds.
It is licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE)
and [NOTICE](NOTICE).

> This is the same OpenDRT that ships as the **"6A. OpenDRT"** tool inside the
> Dec. 18 Studios *Look Dev Tools* suite. The transform is identical; this build
> is standalone and free.

---

## Install (pre-built)

Download the build for your platform from the
[**Releases**](https://github.com/Dec18studios/Open-DRT-OFX/releases) page, then
copy the `OpenDRT.ofx.bundle` into your host's OFX plug-in directory:

| Platform | Asset | Install path |
|----------|-------|--------------|
| macOS (universal) | `OpenDRT_<ver>_macOS_universal.zip` | `/Library/OFX/Plugins` |
| Windows (CUDA) | `OpenDRT_<ver>_windows_x86_64_cuda.zip` | `C:\Program Files\Common Files\OFX\Plugins` |
| Linux (CUDA) | `OpenDRT_<ver>_linux_x86_64_cuda.tar.gz` | `/usr/OFX/Plugins` |

Unzip so the final layout is e.g. `/Library/OFX/Plugins/OpenDRT.ofx.bundle/…`,
then restart your host. The effect appears under the **Dec. 18 Studios** group
as **OpenDRT**.

---

## Build from source

The OFX SDK is cloned on demand (not vendored). From the repo root:

```sh
cd source
git clone --depth 1 https://github.com/AcademySoftwareFoundation/openfx.git ofx-sdk
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

The staged plug-in is written to `source/bundle/OpenDRT.ofx.bundle`. Copy it to
the OFX plug-in directory for your platform (see the table above).

**Requirements**

- CMake ≥ 3.20 and a C++17 compiler.
- **macOS:** Xcode command-line tools (Metal backend, no extra deps).
- **Windows / Linux:** the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  for the GPU backend. CMake auto-detects it; without CUDA the plug-in still
  builds but the CUDA path is omitted.

CI builds all three platforms on every `v*` tag — see
[`.github/workflows/`](.github/workflows/).

---

## License

GPL-3.0. OpenDRT is © Jed Smith; the OFX wrapper, CUDA and Metal backends, and
build system are © Dec. 18 Studios, also under GPL-3.0. Full attribution in
[NOTICE](NOTICE).
