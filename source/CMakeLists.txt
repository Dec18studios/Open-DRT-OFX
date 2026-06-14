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
# Build:
#   git clone --depth 1 https://github.com/AcademySoftwareFoundation/openfx.git ofx-sdk
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build --config Release
# Result: bundle/OpenDRT.ofx.bundle  →  copy to your OFX plugins directory.
# ─────────────────────────────────────────────────────────────────────────────

project(OpenDRT VERSION 2.0.0 LANGUAGES CXX)

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

if(APPLE AND NOT DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "Minimum macOS deployment target")
endif()
if(APPLE AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
  set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "macOS architectures")
endif()

include(CheckLanguage)
find_package(Threads REQUIRED)
find_package(CUDAToolkit QUIET)

# ── Optional CUDA backend (Windows / Linux) ───────────────────────────────────
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

# ── Backend twins ─────────────────────────────────────────────────────────────
set(OPENDRT_CUDA_SOURCES  tools/6A_OpenDRT/cuda/OpenDRTCuda.cu)
set(OPENDRT_METAL_SOURCES tools/6A_OpenDRT/metal/OpenDRTMetal.mm)

set(OPENDRT_BACKEND_SOURCES "")
if(APPLE)
  list(APPEND OPENDRT_BACKEND_SOURCES ${OPENDRT_METAL_SOURCES})
endif()

add_library(${PLUGIN_BASENAME} SHARED
  tools/6A_OpenDRT/standalone/OpenDRTStandaloneMain.cpp
  tools/6A_OpenDRT/OpenDRT.cpp
  $<$<BOOL:${OPENDRT_HAS_CUDA}>:${OPENDRT_CUDA_SOURCES}>
  ${OPENDRT_BACKEND_SOURCES}
  ${OFX_SUPPORT_SOURCES}
)

target_compile_features(${PLUGIN_BASENAME} PRIVATE cxx_std_17)
# Free single-factory build → OPENDRT_PLUGIN_ID = "com.dec18studios.OpenDRT".
target_compile_definitions(${PLUGIN_BASENAME} PRIVATE LDT_OPENDRT_STANDALONE)

target_include_directories(${PLUGIN_BASENAME} PRIVATE
  ${LOCAL_OFX_SDK_DIR}/include
  ${LOCAL_OFX_SDK_DIR}/Support/include
  ${LOCAL_OFX_SDK_DIR}/Support/Library
  ${CMAKE_CURRENT_SOURCE_DIR}/shared
  ${CMAKE_CURRENT_SOURCE_DIR}/tools/6A_OpenDRT
)

target_link_libraries(${PLUGIN_BASENAME} PRIVATE Threads::Threads)

if(OPENDRT_HAS_CUDA)
  target_compile_definitions(${PLUGIN_BASENAME} PRIVATE LDT_HAS_CUDA)
  target_include_directories(${PLUGIN_BASENAME} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE CUDA::cudart)
  set_target_properties(${PLUGIN_BASENAME} PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    # OFF: the kernel TU is self-contained (no cross-TU __device__ calls), so we
    # do NOT want a device-link step. With it ON, nvlink merge_elf can fail across
    # multiple archs on CUDA 13 (nvlink fatal : merge_elf failed).
    CUDA_SEPARABLE_COMPILATION OFF)
endif()

if(APPLE)
  enable_language(OBJCXX)
  # OpenDRT has no custom panel — Foundation + Metal only (no AppKit).
  target_link_libraries(${PLUGIN_BASENAME} PRIVATE
    "-framework Foundation"
    "-framework Metal")
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

# ── Bundle staging → bundle/OpenDRT.ofx.bundle ────────────────────────────────
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

message(STATUS "OpenDRT.ofx → bundle/${PLUGIN_BASENAME}.ofx.bundle, id ${PLUGIN_IDENTIFIER}, "
  "CUDA=${OPENDRT_HAS_CUDA}")
