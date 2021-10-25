cmake_minimum_required(VERSION 3.16)
project(benchmark LANGUAGES CUDA CXX C)

cmake_policy(SET CMP0104 OLD)

set(CMAKE_VERBOSE_MAKEFILE:BOOL ON)

set(CUDA_ARCHS 86)
#set(CUDA_ARCHITECTURES ${CUDA_ARCHS})
#set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHS})

# Here CUDA is properly found and variable are correctly set
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
else ()
    message(FATAL_ERROR "CUDA required, failed to CUDA, try setting CUDACXX and resetting the CMakeCache")
endif ()

add_subdirectory(third_party)
message(STATUS "gencode ${CUDA_GENCODE_FLAGS}")

# Helper to create CUDA gencode flags.
function(create_cuda_gencode_flags out archs_args)
  set(archs ${archs_args} ${ARGN})
  set(tmp "")
  foreach(arch IN LISTS archs)
    set(tmp "${tmp} -gencode arch=compute_${arch},code=sm_${arch}")
  endforeach(arch)
  set(${out} ${tmp} PARENT_SCOPE)
endfunction()

create_cuda_gencode_flags(CUDA_GENCODE_FLAGS ${CUDA_ARCHS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODE_FLAGS}")
message(STATUS "gencode ${CMAKE_CUDA_FLAGS}")

# Build `kernel_python_bindings` python module
set(PYBIND11_PYTHON_VERSION 3.6 3.9)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" CACHE STRING "" FORCE)# you re-configure the project

set(CUDA_SRCS src/benchmark_cpp/kernel_python_bindings.cu)
set_source_files_properties(${CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
pybind11_add_module(kernel_python_bindings ${CUDA_SRCS})
target_compile_options(kernel_python_bindings INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS}>)

#set_target_properties(kernel_python_bindings PROPERTIES CUDA_ARCHITECTURES "${CUDA_ARCHS}")
add_kernel_sputnik(kernel_python_bindings)
