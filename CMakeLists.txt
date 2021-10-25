cmake_minimum_required(VERSION 3.16)
project(benchmark LANGUAGES CUDA CXX C)

cmake_policy(SET CMP0104 OLD)

#
#   Use -DCUDA_ARCHS to specify which GPUs to compile for
#
IF(NOT DEFINED CUDA_ARCHS )
    set(CUDA_ARCHS 86)
ENDIF()

# Helper to create CUDA gencode flags.
function(create_cuda_gencode_flags out archs_args)
    set(archs ${archs_args} ${ARGN})
    set(tmp "")
    foreach(arch IN LISTS archs)
        set(tmp "${tmp} -gencode arch=compute_${arch},code=sm_${arch}")
    endforeach(arch)
    set(${out} ${tmp} PARENT_SCOPE)
endfunction()

# Here CUDA is properly found and variable are correctly set
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
else ()
    message(FATAL_ERROR "CUDA required, failed to CUDA, try setting CUDACXX and resetting the CMakeCache")
endif ()

create_cuda_gencode_flags(CUDA_GENCODE_FLAGS ${CUDA_ARCHS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODE_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}" CACHE STRING "" FORCE)

add_subdirectory(third_party)

set(CUDA_SRCS src/benchmark_cpp/kernel_python_bindings.cu)
set_source_files_properties(${CUDA_SRCS} PROPERTIES LANGUAGE CUDA)
pybind11_add_module(kernel_python_bindings ${CUDA_SRCS})
target_compile_options(kernel_python_bindings INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS}>)
add_kernel_sputnik(kernel_python_bindings)
