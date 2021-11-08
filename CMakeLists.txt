cmake_minimum_required(VERSION 3.16)
project(benchmark LANGUAGES CUDA CXX C)

cmake_policy(SET CMP0104 OLD)

#
#   Use -DCUDA_ARCHS to specify which GPUs to compile for
#
IF(NOT DEFINED CUDA_ARCHS)
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

# Collect common
file(GLOB COMMON_CPP_SRCS ${PROJECT_SOURCE_DIR}/common/**/*.cc ${PROJECT_SOURCE_DIR}/common/*.cc)
file(GLOB COMMON_CUDA_SRCS ${PROJECT_SOURCE_DIR}/common/**/*.cu ${PROJECT_SOURCE_DIR}/common/*.cu)
set_source_files_properties(${COMMON_CUDA_SRCS} PROPERTIES LANGUAGE CUDA)

file(GLOB COMMON_UTILS_CPP_SRCS ${PROJECT_SOURCE_DIR}/common/utils/*.cc)

message(STATUS ${COMMON_CPP_SRCS})
message(STATUS ${COMMON_CUDA_SRCS})

macro(add_common)
    message(STATUS "[${ARGV0}] Adding common")
    target_sources(${ARGV0} PUBLIC ${COMMON_CUDA_SRCS})
    target_sources(${ARGV0} PUBLIC ${COMMON_CPP_SRCS})
    target_include_directories(${ARGV0} PUBLIC ${PROJECT_SOURCE_DIR})
endmacro()

macro(add_common_utils)
    message(STATUS "[${ARGV0}] Adding common utils")
    target_sources(${ARGV0} PUBLIC ${COMMON_UTILS_CPP_SRCS})
    target_include_directories(${ARGV0} PUBLIC ${PROJECT_SOURCE_DIR})
endmacro()

###############################################
#   Benchmark
###############################################

# Create python bindings target

# set(CUDA_SRCS src/benchmark_cpp/kernel_python_bindings.cu)
# pybind11_add_module(kernel_python_bindings ${CUDA_SRCS})

# target_compile_options(kernel_python_bindings INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS}>)
# add_kernel_sputnik(kernel_python_bindings)
# add_common(kernel_python_bindings)


###############################################
#   Tools
###############################################

## Create kernel driver target for profiling

set(CUDA_SRCS tools/kernel_driver.cu)
add_executable(kernel_driver ${CUDA_SRCS})

target_compile_options(kernel_driver INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS}>)
add_kernel_sputnik(kernel_driver)
add_common(kernel_driver)

## Row similarity

add_executable(row_similarity tools/row_similarity.cc)
add_common_utils(row_similarity)

find_package(Boost REQUIRED)
target_link_libraries(row_similarity PUBLIC Boost::boost)

## Stride check

add_executable(stride_check tools/stride_check.cc)
add_common_utils(stride_check)

find_package(Boost REQUIRED)
target_link_libraries(stride_check PUBLIC Boost::boost)

## Synthetic codelets

set(CUDA_SRCS tools/synthetic_codlets.cu)
add_executable(synthetic_codelets ${CUDA_SRCS})
target_compile_options(synthetic_codelets INTERFACE $<$<COMPILE_LANGUAGE:CUDA>:${CMAKE_CUDA_FLAGS}>)
add_cupti_profiler(synthetic_codelets)
add_common_utils(synthetic_codelets)
add_kernel_sputnik(synthetic_codelets)