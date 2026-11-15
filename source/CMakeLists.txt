cmake_minimum_required(VERSION 3.20)

# ─────────────────────────────────────────────────────────────────────────────
# OpenDRT.ofx — free, GPL-3.0 single-factory build of Jed Smith's OpenDRT
# display transform for OFX hosts (DaVinci Resolve, Fusion, Nuke, …).
#
# This is the public, self-building export of the OpenDRT tool that also ships
# as factory "6A. OpenDRT" inside the proprietary Dec. 18 Studios "Look Dev
# Tools" suite. The algorithm source (tools/6A_OpenDRT) is identical; here it is
# compiled with LDT_OPENDRT_STANDALONE so the OFX identifier resolves to
# "com.dec18studios.OpenDRT" and the free build co-installs with the suite.
#
# Model control: Current, v1.0.1, v1.1, Tesseract, Chromagnon.
# GPU backends: Metal on macOS, CUDA and OpenCL on Windows and Linux.
#
# Build:
#   git clone --depth 1 https://github.com/AcademySoftwareFoundation/openfx.git ofx-sdk
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release
# Result: bundle/OpenDRT.ofx.bundle, copy it to your OFX plugins directory.
# ─────────────────────────────────────────────────────────────────────────────

project(OpenDRT VERSION 2.5.0 LANGUAGES CXX)

set(PLUGIN_BASENAME   "OpenDRT")
set(PLUGIN_IDENTIFIER "com.dec18studios.OpenDRT")
set(LOCAL_OFX_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ofx-sdk")

if(NOT EXISTS "${LOCAL_OFX_SDK_DIR}/include/ofxImageEffect.h")
  message(FATAL_ERROR "Missing OFX SDK at ${LOCAL_OFX_SDK_DIR}. "
    "Run: git clone --depth 1 https://github.com/AcademySoftwareFoundation/openfx.git ofx-sdk")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Guard on emptiness, not definedness: CMake 4.x pre-defines both cache entries
# as empty strings, so a NOT DEFINED guard silently skips (arm64 only, and the
# build machine's macOS as the minimum). Explicit -D values are non-empty and
# still win.
if(APPLE AND "${CMAKE_OSX_DEPLOYMENT_TARGET}" STREQUAL "")
  set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS deployment target" FORCE)
endif()
if(APPLE AND "${CMAKE_OSX_ARCHITECTURES}" STREQUAL "")
  set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "macOS architectures" FORCE)
endif()

include(CheckLanguage)
find_package(Threads REQUIRED)
find_package(CUDAToolkit QUIET)

# -- Restrict the bundle to the OFX C ABI --------------------------------------
# An OFX plugin only needs to export OfxGetNumberOfPlugins / OfxGetPlugin
# (/ OfxSetHost). By default a SHARED lib exports every external C++ symbol,
# including the OFX support library internals and the C++ type_info. This free
# build and the Look Dev Tools suite are cut from the same source, so they would
# export the same mangled symbols; when a host loads both, the duplicate
# type_info breaks OFX exception matching and the host can crash at startup.
# Exporting only the C entry points keeps every C++ symbol local, so the two
# bundles co-install cleanly.
# macOS: -exported_symbols_list. Linux: a --version-script. Windows already only
# exports the dllexport'd OFX entry points, so it needs nothing.
function(opendrt_restrict_ofx_exports tgt)
  if(APPLE)
    set(_ofx_exp "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ofx_symbols.exp")
    target_link_options(${tgt} PRIVATE "LINKER:-exported_symbols_list,${_ofx_exp}")
    set_property(TARGET ${tgt} APPEND PROPERTY LINK_DEPENDS "${_ofx_exp}")
  elseif(UNIX)
    set(_ofx_map "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ofx_symbols.map")
    target_link_options(${tgt} PRIVATE "LINKER:--version-script=${_ofx_map}")
    set_property(TARGET ${tgt} APPEND PROPERTY LINK_DEPENDS "${_ofx_map}")
    # The version script only hides what we export, not what we import. Some
    # hosts (DaVinci Resolve on Linux) load libraries that export a private copy
    # of libstdc++ internals ahead of the system one, so any std:: call we import
    # can bind to their copy and crash. Link the C++ runtime in statically; the
    # version script keeps it local.
    target_link_options(${tgt} PRIVATE -static-libstdc++ -static-libgcc)
  endif()
endfunction()

# -- Optional CUDA backend (Windows / Linux) -----------------------------------
set(OPENDRT_HAS_CUDA OFF)
if(CUDAToolkit_FOUND AND NOT APPLE)
  if((NOT CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER MATCHES "-NOTFOUND$")
     AND DEFINED CUDAToolkit_NVCC_EXECUTABLE
     AND EXISTS "${CUDAToolkit_NVCC_EXECUTABLE}")
    set(CMAKE_CUDA_COMPILER "${CUDAToolkit_NVCC_EXECUTABLE}" CACHE FILEPATH "" FORCE)
  endif()
  check_language(CUDA)
endif()
if(CUDAToolkit_FOUND AND NOT APPLE AND CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(DEFINED CUDAToolkit_VERSION AND CUDAToolkit_VERSION VERSION_LESS "13.0")
      set(CMAKE_CUDA_ARCHITECTURES 61 75 80 86 89)
    else()
      set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
    endif()
  endif()
  if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
  endif()
  set(OPENDRT_HAS_CUDA ON)
endif()

# -- Optional OpenCL backend (Windows / Linux) ---------------------------------
# Headers only: the ICD loader is opened at run time with LoadLibrary / dlopen
# (see shared/LdtOpenCL.cpp), so the plugin never links OpenCL and still loads
# on a machine without an OpenCL driver. macOS stays on Metal and never compiles
# the OpenCL path.
set(OPENDRT_HAS_OPENCL OFF)
if(NOT APPLE)
  find_package(OpenCL QUIET)
endif()
if(NOT APPLE AND OpenCL_INCLUDE_DIRS)
  set(OPENDRT_HAS_OPENCL ON)
  message(STATUS "OpenDRT: OpenCL backend ON (headers: ${OpenCL_INCLUDE_DIRS})")
endif()

set(OFX_SUPPORT_SOURCES
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsCore.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsImageEffect.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsInteract.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsLog.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsMultiThread.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsParams.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsProperty.cpp
  ${LOCAL_OFX_SDK_DIR}/Support/Library/ofxsPropertyValidation.cpp
)

# -- Backend twins (one per model: Current/v1.0.1, v1.1, Tesseract, Chromagnon) -
set(OPENDRT_CUDA_SOURCES
  tools/6A_OpenDRT/cuda/OpenDRTCuda.cu
  tools/6A_OpenDRT/cuda/OpenDRTv11Cuda.cu
  tools/6A_OpenDRT/cuda/TesseractCuda.cu
  tools/6A_OpenDRT/cuda/ChromagnonCuda.cu)
set(OPENDRT_METAL_SOURCES
  tools/6A_OpenDRT/metal/OpenDRTMetal.mm
  tools/6A_OpenDRT/metal/OpenDRTv11Metal.mm
  tools/6A_OpenDRT/metal/TesseractMetal.mm
  tools/6A_OpenDRT/metal/ChromagnonMetal.mm)
set(OPENDRT_OPENCL_SOURCES
  shared/LdtOpenCL.cpp
  tools/6A_OpenDRT/opencl/OpenDRTOpenCL.cpp
  tools/6A_OpenDRT/opencl/OpenDRTv11OpenCL.cpp
  tools/6A_OpenDRT/opencl/TesseractOpenCL.cpp
  tools/6A_OpenDRT/opencl/ChromagnonOpenCL.cpp)

set(OPENDRT_BACKEND_SOURCES "")
if(APPLE)
  list(APPEND OPENDRT_BACKEND_SOURCES ${OPENDRT_METAL_SOURCES})
endif()

# -- OpenCL kernel embedding ---------------------------------------------------
# Each .cl file is embedded into a generated header as a C++ string
# (<Symbol>OpenCLKernelSource()), compiled at run time by the OpenCL driver.
set(OPENDRT_OPENCL_GENERATED_DIR "${CMAKE_BINARY_DIR}/generated")
set(OPENDRT_OPENCL_EMBED_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed-opencl-kernels.cmake")
function(opendrt_embed_opencl_kernel cl_path symbol out_var)
  set(_in  "${CMAKE_CURRENT_SOURCE_DIR}/${cl_path}")
  set(_out "${OPENDRT_OPENCL_GENERATED_DIR}/${symbol}OpenCLKernelSource.h")
  add_custom_command(
    OUTPUT  "${_out}"
    COMMAND ${CMAKE_COMMAND} -DLDT_CL_INPUT=${_in} -DLDT_CL_OUTPUT=${_out}
            -DLDT_CL_SYMBOL=${symbol} -P "${OPENDRT_OPENCL_EMBED_SCRIPT}"
    DEPENDS "${_in}" "${OPENDRT_OPENCL_EMBED_SCRIPT}"
    COMMENT "Embedding OpenCL kernel ${symbol}"
    VERBATIM)
  set(${out_var} "${_out}" PARENT_SCOPE)
endfunction()
set(OPENDRT_OPENCL_GENERATED "")
if(OPENDRT_HAS_OPENCL)
  file(MAKE_DIRECTORY "${OPENDRT_OPENCL_GENERATED_DIR}")
  foreach(_pair
      "tools/6A_OpenDRT/opencl/OpenDRTOpenCLKernels.cl|OpenDRT"
      "tools/6A_OpenDRT/opencl/OpenDRTv11OpenCLKernels.cl|OpenDRTv11"
      "tools/6A_OpenDRT/opencl/TesseractOpenCLKernels.cl|Tesseract"
      "tools/6A_OpenDRT/opencl/ChromagnonOpenCLKernels.cl|Chromagnon")
    string(REPLACE "|" ";" _parts "${_pair}")
    list(GET _parts 0 _cl)
    list(GET _parts 1 _sym)
    opendrt_embed_opencl_kernel("${_cl}" "${_sym}" _hdr)
    list(APPEND OPENDRT_OPENCL_GENERATED "${_hdr}")
  endforeach()
  add_custom_target(OpenDRTOpenCLKernels DEPENDS ${OPENDRT_OPENCL_GENERATED})
endif()

# My Presets: native export / import pickers and Show Preset Folder.
if(APPLE)
  set(OPENDRT_FILE_DIALOG_SOURCES shared/LdtFileDialogs.mm)
else()
  set(OPENDRT_FILE_DIALOG_SOURCES shared/LdtFileDialogs.cpp)
endif()

add_library(${PLUGIN_BASENAME} SHARED
  tools/6A_OpenDRT/standalone/OpenDRTStandaloneMain.cpp
  tools/6A_OpenDRT/OpenDRT.cpp
  shared/LdtControl.cpp
  shared/LdtFactorySettings.cpp
  shared/LdtPresetStore.cpp
  ${OPENDRT_FILE_DIALOG_SOURCES}
  $<$<BOOL:${OPENDRT_HAS_CUDA}>:${OPENDRT_CUDA_SOURCES}>
  $<$<BOOL:${OPENDRT_HAS_OPENCL}>:${OPENDRT_OPENCL_SOURCES}>
  ${OPENDRT_BACKEND_SOURCES}
  ${OFX_SUPPORT_SOURCES}
)

target_compile_features(${PLUGIN_BASENAME} PRIVATE cxx_std_17)
# Free single-factory build, so OPENDRT_PLUGIN_ID = "com.dec18studios.OpenDRT".
target_compile_definitions(${PLUGIN_BASENAME} PRIVATE LDT_OPENDRT_STANDALONE)

target_include_directories(${PLUGIN_BASENAME} PRIVATE
  ${LOCAL_OFX_SDK_DIR}/include
  ${LOCAL_OFX_SDK_DIR}/Support/include
  ${LOCAL_OFX_SDK_DIR}/Support/Library
  ${CMAKE_CURRENT_SOURCE_DIR}/shared
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/6A_OpenDRT
)

target_link_libraries(${PLUGIN_BASENAME} PRIVATE Threads::Threads)
if(WIN32)
  # LdtFileDialogs.cpp: IFileSaveDialog / IFileOpenDialog and ShellExecuteW.
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE ole32 shell32 uuid)
endif()

if(OPENDRT_HAS_CUDA)
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE LDT_HAS_CUDA)
  target_include_directories(${PLUGIN_BASENAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  # Static runtime: the bundle must not need cudart64_*.dll / libcudart.so on
  # the user's machine (the driver API lives in the display driver).
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE CUDA::cudart_static)
  set_target_properties(${PLUGIN_BASENAME} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    # OFF: each .cu is self-contained (no cross-TU __device__ calls), so we do
    # NOT want a device-link step. With it ON, nvlink merge_elf can fail across
    # multiple archs on CUDA 13 (nvlink fatal : merge_elf failed).
    CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(OPENDRT_HAS_OPENCL)
  # Headers only, never linked (see the detection block above).
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE LDT_HAS_OPENCL)
  add_dependencies(${PLUGIN_BASENAME} OpenDRTOpenCLKernels)
  target_include_directories(${PLUGIN_BASENAME} PRIVATE
    ${OpenCL_INCLUDE_DIRS}
    ${CMAKE_BINARY_DIR}/generated)
  # dlopen of the ICD loader (part of libc on glibc 2.34+, libdl before).
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE ${CMAKE_DL_LIBS})
endif()

if(APPLE)
  enable_language(OBJCXX)
  # No custom panel, but My Presets uses NSSavePanel / NSOpenPanel /
  # NSWorkspace (shared/LdtFileDialogs.mm, MRC), so AppKit is linked too.
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE
    "-framework Foundation"
    "-framework Metal"
    "-framework AppKit")
endif()

if(MSVC)
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE
    _CRT_SECURE_NO_WARNINGS NOMINMAX _USE_MATH_DEFINES)
  target_compile_options(${PLUGIN_BASENAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W3 /bigobj /MP /EHsc>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W3,-bigobj,-EHsc>)
endif()

set_target_properties(${PLUGIN_BASENAME} PROPERTIES
  PREFIX ""
  SUFFIX ".ofx"
  OUTPUT_NAME "${PLUGIN_BASENAME}")

# Export only the OFX C ABI so the free build co-installs with the suite.
opendrt_restrict_ofx_exports(${PLUGIN_BASENAME})

# -- Bundle staging to bundle/OpenDRT.ofx.bundle -------------------------------
set(OFX_STAGE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/bundle/${PLUGIN_BASENAME}.ofx.bundle")
if(WIN32)
  set(OFX_OUTPUT_DIR    "${OFX_STAGE_ROOT}/Contents/Win64")
  set(OFX_RESOURCES_DIR "${OFX_STAGE_ROOT}/Contents/Resources")
elseif(APPLE)
  set(OFX_BUNDLE_CONTENTS "${OFX_STAGE_ROOT}/Contents")
  set(OFX_OUTPUT_DIR    "${OFX_BUNDLE_CONTENTS}/MacOS")
  set(OFX_RESOURCES_DIR "${OFX_BUNDLE_CONTENTS}/Resources")
else()
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _ARCH)
  if(_ARCH STREQUAL "x86_64" OR _ARCH STREQUAL "amd64")
    set(_LINUX_ARCH_DIR "Linux-x86-64")
  else()
    set(_LINUX_ARCH_DIR "Linux-${_ARCH}")
  endif()
  set(OFX_OUTPUT_DIR    "${OFX_STAGE_ROOT}/Contents/${_LINUX_ARCH_DIR}")
  set(OFX_RESOURCES_DIR "${OFX_STAGE_ROOT}/Contents/Resources")
endif()

set(BUILD_ARTIFACT_DIR "${CMAKE_CURRENT_BINARY_DIR}/artifacts/opendrt")
file(MAKE_DIRECTORY "${OFX_OUTPUT_DIR}" "${OFX_RESOURCES_DIR}" "${BUILD_ARTIFACT_DIR}")
set_target_properties(${PLUGIN_BASENAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY         "${BUILD_ARTIFACT_DIR}"
  RUNTIME_OUTPUT_DIRECTORY_RELEASE "${BUILD_ARTIFACT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY         "${BUILD_ARTIFACT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY_RELEASE "${BUILD_ARTIFACT_DIR}")

if(APPLE)
  file(MAKE_DIRECTORY "${OFX_BUNDLE_CONTENTS}")
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Info.plist.in"
                 "${CMAKE_CURRENT_BINARY_DIR}/Info.plist" @ONLY)
  add_custom_command(TARGET ${PLUGIN_BASENAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_CURRENT_BINARY_DIR}/Info.plist"
      "${OFX_BUNDLE_CONTENTS}/Info.plist"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:${PLUGIN_BASENAME}>"
      "${OFX_OUTPUT_DIR}/${PLUGIN_BASENAME}.ofx")
else()
  add_custom_command(TARGET ${PLUGIN_BASENAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "$<TARGET_FILE:${PLUGIN_BASENAME}>"
      "${OFX_OUTPUT_DIR}/${PLUGIN_BASENAME}.ofx")
endif()

# Host effect-list icon, named after the OFX identifier.
add_custom_command(TARGET ${PLUGIN_BASENAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_CURRENT_SOURCE_DIR}/resources/${PLUGIN_IDENTIFIER}.png"
    "${OFX_RESOURCES_DIR}/${PLUGIN_IDENTIFIER}.png"
  COMMENT "Staging OpenDRT effect-list icon into ${OFX_RESOURCES_DIR}")

message(STATUS "OpenDRT.ofx: bundle/${PLUGIN_BASENAME}.ofx.bundle, id ${PLUGIN_IDENTIFIER}, "
  "version ${PROJECT_VERSION}, CUDA=${OPENDRT_HAS_CUDA}, OpenCL=${OPENDRT_HAS_OPENCL}")
