from src.utilities.environment import REPO_ROOT_PATH
import glob
import os
import sys
from .kernel_scipy_wrappers import scipy_spmm_kernel_decorator

_KNOWN_BUILD_FOLDERS = [
    # Prioritize release builds
    'build',
    'cmake-build-release',
    # Debug builds
    'cmake-build-debug'
]

cpython_version = "cpython-" + "".join([str(x) for x in sys.version_info[0:2]])
build_dir = None

for _build_dir in _KNOWN_BUILD_FOLDERS:
    if len(glob.glob(f'{REPO_ROOT_PATH}/{_build_dir}/*python_bindings.{cpython_version}*.so')):
        build_dir = _build_dir
        break

if build_dir is None:
    raise Exception(f'Failed to find a build directory containing python_bindings for {cpython_version}')
else:
    if os.path.exists(f'{build_dir}/CMakeCache.txt'):
        with open(f'{build_dir}/CMakeCache.txt') as cmake_cache:
            if "CMAKE_BUILD_TYPE:STRING=Debug" in cmake_cache.read():
                print("WARNING: using a debug build for benchmark_cpp, timings may not be accurate"
                      f', build folder: {build_dir}')

    build_dir_path = f'{REPO_ROOT_PATH}/{build_dir}'
    if build_dir_path not in sys.path:
        sys.path.append(build_dir_path)

    print("using", build_dir)
    from kernel_python_bindings import *
    # TODO Find a better way of doing this
    sgk_scipy = scipy_spmm_kernel_decorator(sgk)
