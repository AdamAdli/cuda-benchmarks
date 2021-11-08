message(STATUS ${CMAKE_CURRENT_LIST_DIR}/cmake)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(DefineLibHook)

add_subdirectory(kernels/aspt)

# `add_kernel_sputnik(<target>)`
add_subdirectory(kernels/sputnik)
define_lib_hook(PREFIX kernel
    NAME sputnik
    INCLUDE_PATH "kernels/sputnik"
    LIB_NAME "sputnik"
    LIB_PATH "kernels/sputnik/sputnik"
)

# `add_cupti_profiler(<target>)`
cuda_find_library(CUPTI_LIBRARY cupti_static)
cuda_find_library(CUDA_LIBRARY cuda)
define_lib_hook(
    NAME cupti_profiler
    INCLUDE_PATH "cupti_profiler/include"
    LIB_NAME "${CUPTI_LIBRARY};${CUDA_LIBRARY}"
)


# No hook needed, use pybind11_add_module
add_subdirectory(pybind11)
