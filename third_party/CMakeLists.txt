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

# No hook needed, use pybind11_add_module
add_subdirectory(pybind11)
