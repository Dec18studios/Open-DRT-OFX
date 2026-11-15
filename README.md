# OpenDRT for OFX

A free, open-source **OpenDRT** display rendering transform packaged as an
**OpenFX** plug-in for DaVinci Resolve, Fusion, Nuke and other OFX hosts.
GPU-accelerated with **Metal** on macOS and **CUDA** or **OpenCL** on Windows
and Linux.

OpenDRT is [Jed Smith's open display transform](https://github.com/jedypod/open-display-transform).
This repository wraps it as an OFX plug-in and provides cross-platform builds.
It is licensed under the **GNU General Public License v3.0** — see [LICENSE](LICENSE)
and [NOTICE](NOTICE).

> This is the same OpenDRT that ships as the **"6A. OpenDRT"** tool inside the
> Dec. 18 Studios *Look Dev Tools* suite. The transform is identical; this build
> is standalone and free.

---

## What's in 2.5.0

- **Model control** with five models: **Current**, **v1.0.1**, **v1.1**,
  **Tesseract** and **Chromagnon**. v1.1, Tesseract and Chromagnon are ports of
  transforms from the same upstream project.
- **My Presets**: save your own settings as presets and share them as files
  with other users (export and import).
- **Factory Settings**: save your own starting points and pick which one new
  OpenDRT nodes open with (or keep the built-in Factory defaults).
- **GPU backends**: Metal on macOS, CUDA and OpenCL on Windows and Linux. The
  OpenCL driver is loaded at run time, so the plug-in still loads on a machine
  without one.

---

## Install (pre-built)

Download the build for your platform from the
[**Releases**](https://github.com/Dec18studios/Open-DRT-OFX/releases) page, then
copy the `OpenDRT.ofx.bundle` into your host's OFX plug-in directory:

| Platform | Asset | Install path |
|----------|-------|--------------|
| macOS (universal, Metal) | `OpenDRT_macOS_universal.zip` | `/Library/OFX/Plugins` |
| Windows (CUDA and OpenCL) | `OpenDRT_windows_cuda.zip` | `C:\Program Files\Common Files\OFX\Plugins` |
| Linux (CUDA and OpenCL) | `OpenDRT_linux_cuda.zip` | `/usr/OFX/Plugins` |

The Windows and Linux zips contain one plug-in with both GPU backends, CUDA and
OpenCL.

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

- CMake 3.20 or newer and a C++17 compiler (CUDA code is also built as C++17).
- **macOS:** Xcode command-line tools (Metal backend, no extra deps). The
  default build is universal (x86_64 and arm64) for macOS 13.0 and later.
- **Windows / Linux:** the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
  for the CUDA backend, and OpenCL headers for the OpenCL backend (the CUDA
  Toolkit ships them; on Debian or Ubuntu, `opencl-headers`). CMake detects
  both; if either is missing the plug-in still builds without that backend.
  OpenCL is never linked, only its headers are needed.

CI builds and checks all three platforms on pull requests and `v*` tags, see
[`.github/workflows/`](.github/workflows/). CI only verifies the build and
uploads the bundles as workflow artifacts; it does not publish releases.

---

## License

GPL-3.0. OpenDRT is © Jed Smith; the OFX wrapper, CUDA, Metal and OpenCL backends, and
build system are © Dec. 18 Studios, also under GPL-3.0. Full attribution in
[NOTICE](NOTICE).
