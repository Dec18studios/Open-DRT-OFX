# embed-opencl-kernels.cmake: turn one tool's <Tool>OpenCLKernels.cl into a C++
# header the plugin can compile in.
#
# OpenCL has no offline compile step in this build: the device source is handed
# to the vendor driver at render time via clCreateProgramWithSource. Baking the
# text into the binary avoids shipping loose .cl files next to the .ofx bundle.
#
# The .cl file stays the single source of truth, so syntax-check-opencl.sh
# checks exactly what ships.
#
# Emitted as an array of chunks joined at runtime rather than one big literal:
# MSVC caps a single string literal at 65535 bytes.
#
# Invoke with:
#   cmake -DLDT_CL_INPUT=<in.cl> -DLDT_CL_OUTPUT=<out.h> -DLDT_CL_SYMBOL=<Tool>
#         -P cmake/embed-opencl-kernels.cmake
#
# which writes
#   const char* const k<Tool>OpenCLKernelChunks[]   (anonymous namespace)
#   inline const char* <Tool>OpenCLKernelSource()

if(NOT DEFINED LDT_CL_INPUT OR NOT DEFINED LDT_CL_OUTPUT OR NOT DEFINED LDT_CL_SYMBOL)
  message(FATAL_ERROR "embed-opencl-kernels: LDT_CL_INPUT, LDT_CL_OUTPUT and LDT_CL_SYMBOL are required")
endif()

if(NOT LDT_CL_SYMBOL MATCHES "^[A-Za-z_][A-Za-z0-9_]*$")
  message(FATAL_ERROR "embed-opencl-kernels: LDT_CL_SYMBOL '${LDT_CL_SYMBOL}' is not a C++ identifier")
endif()

if(NOT EXISTS "${LDT_CL_INPUT}")
  message(FATAL_ERROR "embed-opencl-kernels: ${LDT_CL_INPUT} not found")
endif()

file(READ "${LDT_CL_INPUT}" LDT_CL_TEXT)

# The raw-string delimiter must not occur in the kernel source.
string(FIND "${LDT_CL_TEXT}" ")LDCL\"" LDT_CL_DELIM_CLASH)
if(NOT LDT_CL_DELIM_CLASH EQUAL -1)
  message(FATAL_ERROR "embed-opencl-kernels: ${LDT_CL_INPUT} contains the raw-string delimiter")
endif()

string(LENGTH "${LDT_CL_TEXT}" LDT_CL_LENGTH)

set(LDT_CL_CHUNK 8000)
set(LDT_CL_BODY "")
set(LDT_CL_OFFSET 0)
while(LDT_CL_OFFSET LESS LDT_CL_LENGTH)
  math(EXPR LDT_CL_REMAIN "${LDT_CL_LENGTH} - ${LDT_CL_OFFSET}")
  if(LDT_CL_REMAIN LESS LDT_CL_CHUNK)
    set(LDT_CL_TAKE ${LDT_CL_REMAIN})
  else()
    set(LDT_CL_TAKE ${LDT_CL_CHUNK})
  endif()
  string(SUBSTRING "${LDT_CL_TEXT}" ${LDT_CL_OFFSET} ${LDT_CL_TAKE} LDT_CL_PIECE)
  string(APPEND LDT_CL_BODY "R\"LDCL(${LDT_CL_PIECE})LDCL\",\n")
  math(EXPR LDT_CL_OFFSET "${LDT_CL_OFFSET} + ${LDT_CL_TAKE}")
endwhile()

get_filename_component(LDT_CL_INPUT_NAME "${LDT_CL_INPUT}" NAME)

# Written to a temp file and copied only when different, so an unchanged .cl
# does not rebuild every tool that includes the header.
file(WRITE "${LDT_CL_OUTPUT}.tmp"
"// GENERATED FILE. Do not edit.
// Produced by cmake/embed-opencl-kernels.cmake from ${LDT_CL_INPUT_NAME}.
// Edit the .cl, not this.
#pragma once

#include <string>

namespace {

const char* const k${LDT_CL_SYMBOL}OpenCLKernelChunks[] = {
${LDT_CL_BODY}};

}  // namespace

// Full OpenCL C device source for ${LDT_CL_SYMBOL}, joined once on first use
// and kept for the life of the process.
inline const char* ${LDT_CL_SYMBOL}OpenCLKernelSource()
{
    static const std::string source = [] {
        std::string joined;
        for (const char* chunk : k${LDT_CL_SYMBOL}OpenCLKernelChunks) {
            joined += chunk;
        }
        return joined;
    }();
    return source.c_str();
}
")
execute_process(COMMAND "${CMAKE_COMMAND}" -E copy_if_different "${LDT_CL_OUTPUT}.tmp" "${LDT_CL_OUTPUT}")
file(REMOVE "${LDT_CL_OUTPUT}.tmp")

message(STATUS "embed-opencl-kernels: wrote ${LDT_CL_OUTPUT} (${LDT_CL_LENGTH} bytes of kernel source)")
